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
import java.io.DataInputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.concurrent.atomic.AtomicBoolean;

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

    private int screenWidth, screenHeight, screenDpi;
    private int captureWidth, captureHeight;

    private long frameCount = 0;
    private long fpsStartTime = 0;
    private float currentFps = 0;

    // Root: persistent su shell for raw screencap
    private Process suProcess;
    private OutputStream suStdin;
    private InputStream suStdout;

    public interface DetectionCallback {
        void onDetectionResult(BoxInfo[] boxes, float inferTimeMs, float fps,
                               int captureW, int captureH);
    }

    public static ScreenCaptureService getInstance() { return sInstance; }
    public static boolean isServiceRunning() { return sInstance != null && sInstance.isRunning.get(); }
    public void setCallback(DetectionCallback cb) { this.callback = cb; }
    public int getScreenWidth() { return screenWidth; }
    public int getScreenHeight() { return screenHeight; }

    private void log(String msg) {
        Log.i(TAG, msg);
        OverlayService ov = OverlayService.getInstance();
        if (ov != null) ov.appendLog(msg);
    }

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

        log("屏幕:" + screenWidth + "x" + screenHeight + " 捕获:" + captureWidth + "x" + captureHeight);

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
                log("MediaProjection数据无效 code=" + resultCode);
                stopSelf();
                return START_NOT_STICKY;
            }
            startMediaProjectionCapture(resultCode, data);
        }
        return START_STICKY;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Root screencap: raw RGBA → 直接喂 C++ (零拷贝)
    // screencap 不加 -p 输出 raw 格式: 4字节width + 4字节height + 4字节format + RGBA数据
    // ═══════════════════════════════════════════════════════════════════════

    private void startRootCapture() {
        isRunning.set(true);
        fpsStartTime = System.currentTimeMillis();
        frameCount = 0;
        log("Root截图启动(raw模式)");
        inferHandler.post(this::rootCaptureLoop);
    }

    private void rootCaptureLoop() {
        if (!isRunning.get()) return;

        try {
            long t0 = System.currentTimeMillis();

            // 每帧执行 screencap (不加-p = raw RGBA输出)
            // 用单独进程，因为 screencap 输出完就退出，比持久shell更可靠
            Process proc = Runtime.getRuntime().exec(new String[]{"su", "-c", "screencap"});
            DataInputStream dis = new DataInputStream(proc.getInputStream());

            // 读取 header: width(4) + height(4) + format(4) = 12 bytes
            int w = Integer.reverseBytes(dis.readInt());
            int h = Integer.reverseBytes(dis.readInt());
            int fmt = Integer.reverseBytes(dis.readInt());

            int dataSize = w * h * 4;
            byte[] rawPixels = new byte[dataSize];

            // 读取全部像素数据
            int offset = 0;
            while (offset < dataSize) {
                int read = dis.read(rawPixels, offset, dataSize - offset);
                if (read <= 0) break;
                offset += read;
            }
            proc.destroy();

            long captureMs = System.currentTimeMillis() - t0;

            if (offset < dataSize) {
                log("截图不完整:" + offset + "/" + dataSize);
            } else {
                // 直接用 ByteBuffer 传给 C++, 零拷贝
                ByteBuffer directBuf = ByteBuffer.allocateDirect(dataSize);
                directBuf.put(rawPixels);
                directBuf.rewind();

                if (YoloV8Ncnn.nativeIsLoaded()) {
                    float confTh = settings.getConfThresh();
                    float nmsTh = settings.getNmsThresh();

                    float[] rawResult = YoloV8Ncnn.nativeDetectBuffer(
                            directBuf, w, h, w * 4, confTh, nmsTh);

                    if (rawResult != null) {
                        float inferMs = YoloV8Ncnn.getInferTime(rawResult);
                        BoxInfo[] boxes = YoloV8Ncnn.parseResult(rawResult);

                        updateFps();
                        handleAutoClick(boxes);

                        log("cap:" + captureMs + "ms inf:" + String.format("%.0f", inferMs)
                                + "ms det:" + boxes.length + " " + w + "x" + h);

                        DetectionCallback cb = callback;
                        if (cb != null) {
                            cb.onDetectionResult(boxes, inferMs, currentFps, w, h);
                        }
                    } else {
                        log("推理null cap:" + captureMs + "ms");
                    }
                } else {
                    log("模型未加载!");
                }
            }
        } catch (Exception e) {
            log("Root异常:" + e.getMessage());
        }

        if (isRunning.get()) {
            inferHandler.post(this::rootCaptureLoop);
        }
    }

    public static boolean checkRootAccess() {
        try {
            Process p = Runtime.getRuntime().exec(new String[]{"su", "-c", "id"});
            BufferedReader r = new BufferedReader(new InputStreamReader(p.getInputStream()));
            String line = r.readLine();
            p.waitFor();
            return line != null && line.contains("uid=0");
        } catch (Exception e) { return false; }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // MediaProjection: Image ByteBuffer → 直接传 C++ (零拷贝)
    // ═══════════════════════════════════════════════════════════════════════

    @SuppressLint("WrongConstant")
    private void startMediaProjectionCapture(int resultCode, Intent data) {
        log("MediaProjection启动 code=" + resultCode);

        MediaProjectionManager mpManager =
                (MediaProjectionManager) getSystemService(MEDIA_PROJECTION_SERVICE);

        try {
            mediaProjection = mpManager.getMediaProjection(resultCode, data);
        } catch (Exception e) {
            log("getMediaProjection异常:" + e.getMessage());
            stopSelf();
            return;
        }

        if (mediaProjection == null) {
            log("MediaProjection=null!");
            stopSelf();
            return;
        }

        log("MediaProjection获取成功");

        mediaProjection.registerCallback(new MediaProjection.Callback() {
            @Override
            public void onStop() {
                log("MediaProjection停止");
                stopCapture();
            }
        }, inferHandler);

        imageReader = ImageReader.newInstance(captureWidth, captureHeight, PixelFormat.RGBA_8888, 3);
        log("ImageReader: " + captureWidth + "x" + captureHeight);

        imageReader.setOnImageAvailableListener(reader -> {
            Image image = reader.acquireLatestImage();
            if (image == null) return;

            if (!inferBusy.compareAndSet(false, true)) {
                image.close();
                return;
            }

            try {
                if (!YoloV8Ncnn.nativeIsLoaded()) {
                    image.close();
                    inferBusy.set(false);
                    log("模型未加载!");
                    return;
                }

                Image.Plane plane = image.getPlanes()[0];
                ByteBuffer buffer = plane.getBuffer();
                int rowStride = plane.getRowStride();
                int w = captureWidth;
                int h = captureHeight;

                float confTh = settings.getConfThresh();
                float nmsTh = settings.getNmsThresh();

                // 零拷贝: 直接把 Image 的 ByteBuffer 传给 C++
                float[] rawResult = YoloV8Ncnn.nativeDetectBuffer(
                        buffer, w, h, rowStride, confTh, nmsTh);

                image.close(); // 推理完再关闭

                if (rawResult != null) {
                    float inferMs = YoloV8Ncnn.getInferTime(rawResult);
                    BoxInfo[] boxes = YoloV8Ncnn.parseResult(rawResult);

                    updateFps();
                    handleAutoClick(boxes);

                    DetectionCallback cb = callback;
                    if (cb != null) {
                        cb.onDetectionResult(boxes, inferMs, currentFps, w, h);
                    }
                }
            } catch (Exception e) {
                log("MP推理异常:" + e.getMessage());
                try { image.close(); } catch (Exception ignored) {}
            } finally {
                inferBusy.set(false);
            }
        }, inferHandler);

        virtualDisplay = mediaProjection.createVirtualDisplay(
                "YoloV8Capture",
                captureWidth, captureHeight, screenDpi,
                DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
                imageReader.getSurface(), null, inferHandler);

        isRunning.set(true);
        fpsStartTime = System.currentTimeMillis();
        frameCount = 0;
        log("VirtualDisplay+ImageReader就绪");
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
        int targetLabel = settings.getTargetLabel();
        int clickX = settings.getClickX(), clickY = settings.getClickY();
        if (clickX < 0 || clickY < 0) return;
        for (BoxInfo box : boxes) {
            if (targetLabel == -1 || box.label == targetLabel) {
                AutoClickService svc = AutoClickService.getInstance();
                if (svc != null) svc.tap(clickX, clickY, settings.getClickDelay());
                break;
            }
        }
    }

    public void stopCapture() {
        isRunning.set(false);
        if (virtualDisplay != null) { virtualDisplay.release(); virtualDisplay = null; }
        if (mediaProjection != null) { mediaProjection.stop(); mediaProjection = null; }
        if (imageReader != null) { imageReader.close(); imageReader = null; }
        destroySu();
        log("捕获已停止");
        stopSelf();
    }

    private void destroySu() {
        try { if (suStdin != null) { suStdin.write("exit\n".getBytes()); suStdin.flush(); suStdin.close(); } } catch (Exception ignored) {}
        try { if (suProcess != null) suProcess.destroy(); } catch (Exception ignored) {}
        suProcess = null; suStdin = null; suStdout = null;
    }

    @Override
    public void onDestroy() {
        stopCapture();
        if (inferThread != null) { inferThread.quitSafely(); inferThread = null; }
        sInstance = null;
        super.onDestroy();
    }

    @Override public IBinder onBind(Intent intent) { return null; }

    private void createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            NotificationChannel ch = new NotificationChannel(CHANNEL_ID, "屏幕捕获", NotificationManager.IMPORTANCE_LOW);
            ch.setDescription("YOLOv8 屏幕推理服务");
            getSystemService(NotificationManager.class).createNotificationChannel(ch);
        }
    }

    private Notification buildNotification() {
        Notification.Builder b = Build.VERSION.SDK_INT >= Build.VERSION_CODES.O
                ? new Notification.Builder(this, CHANNEL_ID)
                : new Notification.Builder(this);
        return b.setContentTitle("YOLOv8 NCNN").setContentText("屏幕推理运行中...")
                .setSmallIcon(android.R.drawable.ic_menu_camera).setOngoing(true).build();
    }
}
