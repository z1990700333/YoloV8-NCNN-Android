package com.yolov8ncnn;

import android.annotation.SuppressLint;
import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.Service;
import android.content.Intent;
import android.content.pm.ServiceInfo;
import android.graphics.PixelFormat;
import android.hardware.display.DisplayManager;
import android.hardware.display.VirtualDisplay;
import android.media.Image;
import android.media.ImageReader;
import android.media.projection.MediaProjection;
import android.media.projection.MediaProjectionManager;
import android.os.Build;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.IBinder;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.WindowManager;

import java.nio.ByteBuffer;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * 纯 MediaProjection 截图服务（已移除 Root 模式）
 * 基于 ncnn 官方示例的简洁实现
 */
public class ScreenCaptureService extends Service {

    private static final String TAG = "ScreenCapture";
    private static final String CHANNEL_ID = "screen_capture_channel";
    private static final int NOTIFICATION_ID = 1001;

    public static final String EXTRA_RESULT_CODE = "result_code";
    public static final String EXTRA_DATA = "data";

    private static ScreenCaptureService sInstance = null;

    private MediaProjection mediaProjection;
    private VirtualDisplay virtualDisplay;
    private ImageReader imageReader;

    private HandlerThread inferThread;
    private Handler inferHandler;

    private SettingsManager settings;

    private final AtomicBoolean isRunning = new AtomicBoolean(false);
    private final AtomicBoolean inferBusy = new AtomicBoolean(false);
    private volatile DetectionCallback callback;

    private int screenWidth, screenHeight, screenDpi;
    private int captureWidth, captureHeight;

    private long frameCount = 0;
    private long fpsStartTime = 0;
    private float currentFps = 0;

    public interface DetectionCallback {
        void onDetectionResult(BoxInfo[] boxes, float inferTimeMs, float fps,
                               int captureW, int captureH);
    }

    public static ScreenCaptureService getInstance() { return sInstance; }
    public static boolean isServiceRunning() { return sInstance != null && sInstance.isRunning.get(); }
    public void setCallback(DetectionCallback cb) { this.callback = cb; }

    private void log(String msg) {
        Log.i(TAG, msg);
        MainActivity.appendLog(msg);
        OverlayService ov = OverlayService.getInstance();
        if (ov != null) ov.appendLog(msg);
    }

    @Override
    public void onCreate() {
        super.onCreate();
        sInstance = this;
        settings = new SettingsManager(this);

        // 获取屏幕参数
        WindowManager wm = (WindowManager) getSystemService(WINDOW_SERVICE);
        DisplayMetrics dm = new DisplayMetrics();
        wm.getDefaultDisplay().getRealMetrics(dm);
        screenWidth = dm.widthPixels;
        screenHeight = dm.heightPixels;
        screenDpi = dm.densityDpi;

        float scale = settings.getCaptureScale();
        captureWidth = ((int)(screenWidth * scale) / 2) * 2;
        captureHeight = ((int)(screenHeight * scale) / 2) * 2;

        log("屏幕:" + screenWidth + "x" + screenHeight + " 捕获:" + captureWidth + "x" + captureHeight);

        // 推理线程
        inferThread = new HandlerThread("InferThread", Thread.MAX_PRIORITY);
        inferThread.start();
        inferHandler = new Handler(inferThread.getLooper());

        createNotificationChannel();
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        if (intent == null) {
            log("intent=null, 停止");
            stopSelf();
            return START_NOT_STICKY;
        }

        // 必须在 getMediaProjection 之前调用 startForeground
        // Android 10+ 必须指定 FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(NOTIFICATION_ID, buildNotification(),
                    ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION);
        } else {
            startForeground(NOTIFICATION_ID, buildNotification());
        }

        int resultCode = intent.getIntExtra(EXTRA_RESULT_CODE, 0);
        Intent data = intent.getParcelableExtra(EXTRA_DATA);
        log("onStartCommand: resultCode=" + resultCode + " data=" + (data != null ? "OK" : "null"));

        if (data == null) {
            log("MediaProjection data=null, 停止");
            stopSelf();
            return START_NOT_STICKY;
        }

        startMediaProjectionCapture(resultCode, data);
        return START_STICKY;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // MediaProjection 截图
    // ═══════════════════════════════════════════════════════════════════════

    @SuppressLint("WrongConstant")
    private void startMediaProjectionCapture(int resultCode, Intent data) {
        MediaProjectionManager mpManager =
                (MediaProjectionManager) getSystemService(MEDIA_PROJECTION_SERVICE);
        try {
            mediaProjection = mpManager.getMediaProjection(resultCode, data);
        } catch (Exception e) {
            log("getMediaProjection异常: " + e.getMessage());
            stopSelf();
            return;
        }

        if (mediaProjection == null) {
            log("MediaProjection=null, 授权可能失败");
            stopSelf();
            return;
        }
        log("MediaProjection创建成功");

        mediaProjection.registerCallback(new MediaProjection.Callback() {
            @Override
            public void onStop() {
                log("MediaProjection回调: 停止");
                stopCapture();
            }
        }, inferHandler);

        // 创建 ImageReader
        imageReader = ImageReader.newInstance(captureWidth, captureHeight, PixelFormat.RGBA_8888, 2);

        imageReader.setOnImageAvailableListener(reader -> {
            Image image = reader.acquireLatestImage();
            if (image == null) return;

            // 如果上一帧还在推理，丢弃当前帧
            if (!inferBusy.compareAndSet(false, true)) {
                image.close();
                return;
            }

            try {
                if (!YoloV8Ncnn.nativeIsLoaded()) {
                    image.close();
                    inferBusy.set(false);
                    return;
                }

                Image.Plane plane = image.getPlanes()[0];
                ByteBuffer buffer = plane.getBuffer();
                int rowStride = plane.getRowStride();
                int w = image.getWidth();
                int h = image.getHeight();

                if (frameCount == 0) {
                    log("首帧: " + w + "x" + h + " stride=" + rowStride
                            + " buf=" + buffer.remaining());
                }

                float[] rawResult = YoloV8Ncnn.nativeDetectBuffer(
                        buffer, w, h, rowStride,
                        settings.getConfThresh(), settings.getNmsThresh());

                image.close();

                if (rawResult != null) {
                    float inferMs = YoloV8Ncnn.getInferTime(rawResult);
                    BoxInfo[] boxes = YoloV8Ncnn.parseResult(rawResult);
                    updateFps();
                    handleAutoClick(boxes);

                    if (frameCount <= 3) {
                        log("推理: " + String.format("%.0f", inferMs) + "ms 检测:" + boxes.length
                                + " fps:" + String.format("%.1f", currentFps));
                    }

                    DetectionCallback cb = callback;
                    if (cb != null) cb.onDetectionResult(boxes, inferMs, currentFps, w, h);
                } else {
                    if (frameCount <= 3) log("推理返回null");
                }
            } catch (Exception e) {
                log("帧处理异常: " + e.getMessage());
                try { image.close(); } catch (Exception ignored) {}
            } finally {
                inferBusy.set(false);
            }
        }, inferHandler);

        // 创建 VirtualDisplay
        try {
            virtualDisplay = mediaProjection.createVirtualDisplay("YoloV8Capture",
                    captureWidth, captureHeight, screenDpi,
                    DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
                    imageReader.getSurface(), null, inferHandler);
            log("VirtualDisplay创建成功");
        } catch (Exception e) {
            log("VirtualDisplay创建失败: " + e.getMessage());
            stopSelf();
            return;
        }

        isRunning.set(true);
        fpsStartTime = System.currentTimeMillis();
        frameCount = 0;
        log("MediaProjection就绪 " + captureWidth + "x" + captureHeight);
    }

    // ═══════════════════════════════════════════════════════════════════════

    private void updateFps() {
        frameCount++;
        long now = System.currentTimeMillis();
        if (fpsStartTime == 0) fpsStartTime = now;
        long elapsed = now - fpsStartTime;
        if (elapsed >= 1000) {
            currentFps = frameCount * 1000.0f / elapsed;
            frameCount = 0;
            fpsStartTime = now;
        }
    }

    private void handleAutoClick(BoxInfo[] boxes) {
        if (!settings.getAutoClick() || boxes == null || boxes.length == 0) return;
        if (!AutoClickService.isRunning()) return;
        int tl = settings.getTargetLabel();
        int cx = settings.getClickX(), cy = settings.getClickY();
        if (cx < 0 || cy < 0) return;
        for (BoxInfo b : boxes) {
            if (tl == -1 || b.label == tl) {
                AutoClickService s = AutoClickService.getInstance();
                if (s != null) s.tap(cx, cy, settings.getClickDelay());
                break;
            }
        }
    }

    public void stopCapture() {
        isRunning.set(false);
        if (virtualDisplay != null) { virtualDisplay.release(); virtualDisplay = null; }
        if (mediaProjection != null) { mediaProjection.stop(); mediaProjection = null; }
        if (imageReader != null) { imageReader.close(); imageReader = null; }
        log("已停止");
        stopSelf();
    }

    @Override
    public void onDestroy() {
        stopCapture();
        if (inferThread != null) { inferThread.quitSafely(); inferThread = null; }
        sInstance = null;
        super.onDestroy();
    }

    @Override
    public IBinder onBind(Intent intent) { return null; }

    private void createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            NotificationChannel ch = new NotificationChannel(
                    CHANNEL_ID, "屏幕捕获", NotificationManager.IMPORTANCE_LOW);
            getSystemService(NotificationManager.class).createNotificationChannel(ch);
        }
    }

    private Notification buildNotification() {
        Notification.Builder b = Build.VERSION.SDK_INT >= Build.VERSION_CODES.O
                ? new Notification.Builder(this, CHANNEL_ID)
                : new Notification.Builder(this);
        return b.setContentTitle("YOLOv8 NCNN")
                .setContentText("推理中...")
                .setSmallIcon(android.R.drawable.ic_menu_camera)
                .setOngoing(true)
                .build();
    }
}
