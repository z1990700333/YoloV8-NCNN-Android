package com.yolov8ncnn;

import android.annotation.SuppressLint;
import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.Service;
import android.content.Intent;
import android.graphics.Bitmap;
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

import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * 屏幕捕获服务：支持 MediaProjection 和 Root screencap 两种模式
 */
public class ScreenCaptureService extends Service {

    private static final String TAG = "ScreenCapture";
    private static final String CHANNEL_ID = "screen_capture_channel";
    private static final int NOTIFICATION_ID = 1001;

    public static final String EXTRA_RESULT_CODE = "result_code";
    public static final String EXTRA_DATA = "data";
    public static final String EXTRA_USE_ROOT = "use_root";

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
    private boolean useRootCapture = false;

    private int screenWidth;
    private int screenHeight;
    private int screenDpi;
    private int captureWidth;
    private int captureHeight;

    // FPS
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
    public int getScreenWidth() { return screenWidth; }
    public int getScreenHeight() { return screenHeight; }

    @Override
    public void onCreate() {
        super.onCreate();
        sInstance = this;
        settings = new SettingsManager(this);

        WindowManager wm = (WindowManager) getSystemService(WINDOW_SERVICE);
        DisplayMetrics dm = new DisplayMetrics();
        wm.getDefaultDisplay().getRealMetrics(dm);
        screenWidth = dm.widthPixels;
        screenHeight = dm.heightPixels;
        screenDpi = dm.densityDpi;

        float scale = settings.getCaptureScale();
        captureWidth = ((int)(screenWidth * scale) / 2) * 2;
        captureHeight = ((int)(screenHeight * scale) / 2) * 2;

        Log.i(TAG, String.format("屏幕: %dx%d, 捕获: %dx%d, 缩放: %.2f",
                screenWidth, screenHeight, captureWidth, captureHeight, scale));

        inferThread = new HandlerThread("InferThread", Thread.MAX_PRIORITY);
        inferThread.start();
        inferHandler = new Handler(inferThread.getLooper());

        createNotificationChannel();
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        if (intent == null) { stopSelf(); return START_NOT_STICKY; }

        startForeground(NOTIFICATION_ID, buildNotification());

        useRootCapture = intent.getBooleanExtra(EXTRA_USE_ROOT, false);

        if (useRootCapture) {
            startRootCapture();
        } else {
            int resultCode = intent.getIntExtra(EXTRA_RESULT_CODE, -1);
            Intent data = intent.getParcelableExtra(EXTRA_DATA);
            if (resultCode == -1 || data == null) {
                Log.e(TAG, "无效的 MediaProjection 数据");
                stopSelf();
                return START_NOT_STICKY;
            }
            startMediaProjectionCapture(resultCode, data);
        }
        return START_STICKY;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Root screencap 模式
    // ═══════════════════════════════════════════════════════════════════════

    private void startRootCapture() {
        isRunning.set(true);
        fpsStartTime = System.currentTimeMillis();
        frameCount = 0;
        Log.i(TAG, "Root 截图模式启动");

        inferHandler.post(this::rootCaptureLoop);
    }

    private void rootCaptureLoop() {
        if (!isRunning.get()) return;

        try {
            // 使用 su screencap 截图到 raw 文件
            Bitmap bmp = captureScreenByRoot();
            if (bmp != null) {
                if (YoloV8Ncnn.nativeIsLoaded()) {
                    float confThresh = settings.getConfThresh();
                    float nmsThresh = settings.getNmsThresh();

                    float[] rawResult = YoloV8Ncnn.nativeDetectBitmap(bmp, confThresh, nmsThresh);
                    bmp.recycle();

                    if (rawResult != null) {
                        float inferTimeMs = YoloV8Ncnn.getInferTime(rawResult);
                        BoxInfo[] boxes = YoloV8Ncnn.parseResult(rawResult);

                        updateFps();
                        handleAutoClick(boxes);

                        DetectionCallback cb = callback;
                        if (cb != null) {
                            cb.onDetectionResult(boxes, inferTimeMs, currentFps, captureWidth, captureHeight);
                        }
                    }
                } else {
                    bmp.recycle();
                }
            }
        } catch (Exception e) {
            Log.e(TAG, "Root 截图推理出错", e);
        }

        // 继续下一帧
        if (isRunning.get()) {
            inferHandler.post(this::rootCaptureLoop);
        }
    }

    /**
     * 通过 Root 权限执行 screencap 截取屏幕
     */
    private Bitmap captureScreenByRoot() {
        try {
            String tmpPath = getCacheDir().getAbsolutePath() + "/screen.raw";

            // 方法1: 直接读取 screencap 的 stdout (更快，避免文件IO)
            Process process = Runtime.getRuntime().exec(new String[]{"su", "-c",
                    "screencap -p /dev/stdout"});

            Bitmap bmp = android.graphics.BitmapFactory.decodeStream(process.getInputStream());
            process.waitFor();

            if (bmp != null && (bmp.getWidth() != captureWidth || bmp.getHeight() != captureHeight)) {
                Bitmap scaled = Bitmap.createScaledBitmap(bmp, captureWidth, captureHeight, false);
                bmp.recycle();
                return scaled;
            }
            return bmp;
        } catch (Exception e) {
            Log.e(TAG, "Root screencap 失败", e);
            return null;
        }
    }

    /**
     * 检查是否有 Root 权限
     */
    public static boolean checkRootAccess() {
        try {
            Process process = Runtime.getRuntime().exec(new String[]{"su", "-c", "id"});
            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            String line = reader.readLine();
            process.waitFor();
            return line != null && line.contains("uid=0");
        } catch (Exception e) {
            return false;
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // MediaProjection 模式 (修复: 同步复制 buffer)
    // ═══════════════════════════════════════════════════════════════════════

    @SuppressLint("WrongConstant")
    private void startMediaProjectionCapture(int resultCode, Intent data) {
        MediaProjectionManager mpManager =
                (MediaProjectionManager) getSystemService(MEDIA_PROJECTION_SERVICE);
        mediaProjection = mpManager.getMediaProjection(resultCode, data);

        if (mediaProjection == null) {
            Log.e(TAG, "MediaProjection 为 null");
            stopSelf();
            return;
        }

        mediaProjection.registerCallback(new MediaProjection.Callback() {
            @Override
            public void onStop() {
                Log.i(TAG, "MediaProjection 已停止");
                stopCapture();
            }
        }, inferHandler);

        imageReader = ImageReader.newInstance(captureWidth, captureHeight, PixelFormat.RGBA_8888, 2);

        imageReader.setOnImageAvailableListener(reader -> {
            Image image = reader.acquireLatestImage();
            if (image == null) return;

            // 跳帧：如果上一帧还在推理
            if (!inferBusy.compareAndSet(false, true)) {
                image.close();
                return;
            }

            // ★ 关键修复: 在关闭 Image 之前同步复制像素数据
            final Image.Plane plane = image.getPlanes()[0];
            final ByteBuffer srcBuffer = plane.getBuffer();
            final int rowStride = plane.getRowStride();
            final int pixelStride = plane.getPixelStride();
            final int w = captureWidth;
            final int h = captureHeight;

            // 复制到 Bitmap (最安全的方式)
            final Bitmap bmp = Bitmap.createBitmap(
                    w + (rowStride - pixelStride * w) / pixelStride, h,
                    Bitmap.Config.ARGB_8888);
            bmp.copyPixelsFromBuffer(srcBuffer);
            image.close(); // 立即释放 Image

            // 裁剪掉 rowStride 多余的像素
            final Bitmap cropped;
            if (bmp.getWidth() != w) {
                cropped = Bitmap.createBitmap(bmp, 0, 0, w, h);
                bmp.recycle();
            } else {
                cropped = bmp;
            }

            inferHandler.post(() -> {
                try {
                    if (!YoloV8Ncnn.nativeIsLoaded()) return;

                    float confThresh = settings.getConfThresh();
                    float nmsThresh = settings.getNmsThresh();

                    float[] rawResult = YoloV8Ncnn.nativeDetectBitmap(cropped, confThresh, nmsThresh);
                    cropped.recycle();

                    if (rawResult != null) {
                        float inferTimeMs = YoloV8Ncnn.getInferTime(rawResult);
                        BoxInfo[] boxes = YoloV8Ncnn.parseResult(rawResult);

                        updateFps();
                        handleAutoClick(boxes);

                        DetectionCallback cb = callback;
                        if (cb != null) {
                            cb.onDetectionResult(boxes, inferTimeMs, currentFps, w, h);
                        }
                    }
                } catch (Exception e) {
                    Log.e(TAG, "推理出错", e);
                } finally {
                    if (!cropped.isRecycled()) cropped.recycle();
                    inferBusy.set(false);
                }
            });
        }, inferHandler);

        virtualDisplay = mediaProjection.createVirtualDisplay(
                "YoloV8Capture",
                captureWidth, captureHeight, screenDpi,
                DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
                imageReader.getSurface(), null, null);

        isRunning.set(true);
        fpsStartTime = System.currentTimeMillis();
        frameCount = 0;
        Log.i(TAG, "MediaProjection 屏幕捕获已启动");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 公共方法
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
        if (!settings.getAutoClick()) return;
        if (boxes == null || boxes.length == 0) return;
        if (!AutoClickService.isRunning()) return;

        int targetLabel = settings.getTargetLabel();
        int clickX = settings.getClickX();
        int clickY = settings.getClickY();
        if (clickX < 0 || clickY < 0) return;

        for (BoxInfo box : boxes) {
            if (targetLabel == -1 || box.label == targetLabel) {
                AutoClickService service = AutoClickService.getInstance();
                if (service != null) {
                    service.tap(clickX, clickY, settings.getClickDelay());
                }
                break;
            }
        }
    }

    public void stopCapture() {
        isRunning.set(false);
        if (virtualDisplay != null) { virtualDisplay.release(); virtualDisplay = null; }
        if (mediaProjection != null) { mediaProjection.stop(); mediaProjection = null; }
        if (imageReader != null) { imageReader.close(); imageReader = null; }
        Log.i(TAG, "屏幕捕获已停止");
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
            NotificationChannel channel = new NotificationChannel(
                    CHANNEL_ID, "屏幕捕获", NotificationManager.IMPORTANCE_LOW);
            channel.setDescription("YOLOv8 屏幕推理服务");
            getSystemService(NotificationManager.class).createNotificationChannel(channel);
        }
    }

    private Notification buildNotification() {
        Notification.Builder builder;
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            builder = new Notification.Builder(this, CHANNEL_ID);
        } else {
            builder = new Notification.Builder(this);
        }
        return builder
                .setContentTitle("YOLOv8 NCNN")
                .setContentText("屏幕推理运行中...")
                .setSmallIcon(android.R.drawable.ic_menu_camera)
                .setOngoing(true)
                .build();
    }
}
