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
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
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

    private void overlayLog(String msg) {
        Log.i(TAG, msg);
        OverlayService ov = OverlayService.getInstance();
        if (ov != null) ov.appendLog(msg);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Root screencap 模式 (持久化 su shell，避免每帧 fork 新进程)
    // ═══════════════════════════════════════════════════════════════════════

    private Process persistentSuProcess;
    private OutputStream suStdin;
    private InputStream suStdout;

    private void startRootCapture() {
        isRunning.set(true);
        fpsStartTime = System.currentTimeMillis();
        frameCount = 0;
        overlayLog("Root截图启动");

        inferHandler.post(this::rootCaptureLoop);
    }

    private void rootCaptureLoop() {
        if (!isRunning.get()) return;

        try {
            long t0 = System.currentTimeMillis();
            Bitmap bmp = captureScreenByRoot();
            long captureMs = System.currentTimeMillis() - t0;

            if (bmp != null) {
                overlayLog("截图:" + bmp.getWidth() + "x" + bmp.getHeight() + " " + captureMs + "ms");
                if (YoloV8Ncnn.nativeIsLoaded()) {
                    float confThresh = settings.getConfThresh();
                    float nmsThresh = settings.getNmsThresh();

                    float[] rawResult = YoloV8Ncnn.nativeDetectBitmap(bmp, confThresh, nmsThresh);
                    bmp.recycle();

                    if (rawResult != null) {
                        float inferTimeMs = YoloV8Ncnn.getInferTime(rawResult);
                        BoxInfo[] boxes = YoloV8Ncnn.parseResult(rawResult);

                        overlayLog("推理:" + String.format("%.0f", inferTimeMs) + "ms 检测:" + boxes.length);
                        updateFps();
                        handleAutoClick(boxes);

                        DetectionCallback cb = callback;
                        if (cb != null) {
                            cb.onDetectionResult(boxes, inferTimeMs, currentFps, captureWidth, captureHeight);
                        }
                    } else {
                        overlayLog("推理返回null");
                    }
                } else {
                    bmp.recycle();
                    overlayLog("模型未加载!");
                }
            } else {
                overlayLog("截图失败(null) " + captureMs + "ms");
            }
        } catch (Exception e) {
            overlayLog("异常:" + e.getMessage());
            Log.e(TAG, "Root 截图推理出错", e);
        }

        if (isRunning.get()) {
            inferHandler.post(this::rootCaptureLoop);
        }
    }

    /**
     * 通过 Root 权限截取屏幕 — 使用持久化 su shell 避免每帧 fork
     * 策略: 每帧通过持久 su shell 执行 screencap 写入临时文件，再读取
     * 比每帧 Runtime.exec("su") 快 5-10 倍
     */
    private Bitmap captureScreenByRoot() {
        try {
            String tmpPath = getCacheDir().getAbsolutePath() + "/screen_cap.png";

            // 确保持久化 su shell 存活
            if (persistentSuProcess == null || !isSuProcessAlive()) {
                initPersistentSu();
            }

            if (persistentSuProcess == null) {
                // 回退到旧方法
                return captureScreenByRootFallback();
            }

            // 通过持久 su shell 执行截图命令
            // 使用 screencap -p 输出 PNG 到文件，然后用 echo 标记完成
            String cmd = "screencap -p " + tmpPath + " && echo CAPTURE_DONE\n";
            suStdin.write(cmd.getBytes());
            suStdin.flush();

            // 读取直到看到 CAPTURE_DONE
            byte[] buf = new byte[256];
            StringBuilder sb = new StringBuilder();
            long startTime = System.currentTimeMillis();
            boolean done = false;

            while (!done && (System.currentTimeMillis() - startTime) < 3000) {
                if (suStdout.available() > 0) {
                    int read = suStdout.read(buf);
                    if (read > 0) {
                        sb.append(new String(buf, 0, read));
                        if (sb.toString().contains("CAPTURE_DONE")) {
                            done = true;
                        }
                    }
                } else {
                    Thread.sleep(1);
                }
            }

            if (!done) {
                Log.w(TAG, "screencap 超时，重建 su shell");
                destroyPersistentSu();
                return null;
            }

            // 读取 PNG 文件
            File f = new File(tmpPath);
            if (!f.exists() || f.length() == 0) return null;

            Bitmap bmp = android.graphics.BitmapFactory.decodeFile(tmpPath);
            if (bmp != null && (bmp.getWidth() != captureWidth || bmp.getHeight() != captureHeight)) {
                Bitmap scaled = Bitmap.createScaledBitmap(bmp, captureWidth, captureHeight, false);
                bmp.recycle();
                return scaled;
            }
            return bmp;
        } catch (Exception e) {
            Log.e(TAG, "Root screencap 失败", e);
            destroyPersistentSu();
            return null;
        }
    }

    private void initPersistentSu() {
        try {
            destroyPersistentSu();
            persistentSuProcess = Runtime.getRuntime().exec("su");
            suStdin = persistentSuProcess.getOutputStream();
            suStdout = persistentSuProcess.getInputStream();
            Log.i(TAG, "持久化 su shell 已创建");
        } catch (Exception e) {
            Log.e(TAG, "创建持久化 su shell 失败", e);
            persistentSuProcess = null;
        }
    }

    private void destroyPersistentSu() {
        try {
            if (suStdin != null) {
                suStdin.write("exit\n".getBytes());
                suStdin.flush();
                suStdin.close();
            }
        } catch (Exception ignored) {}
        try {
            if (persistentSuProcess != null) persistentSuProcess.destroy();
        } catch (Exception ignored) {}
        persistentSuProcess = null;
        suStdin = null;
        suStdout = null;
    }

    private boolean isSuProcessAlive() {
        if (persistentSuProcess == null) return false;
        try {
            persistentSuProcess.exitValue();
            return false; // 如果能获取 exitValue，说明进程已结束
        } catch (IllegalThreadStateException e) {
            return true; // 进程仍在运行
        }
    }

    /**
     * 回退方法：每帧 fork 新 su 进程 (慢)
     */
    private Bitmap captureScreenByRootFallback() {
        try {
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
            Log.e(TAG, "Root screencap fallback 失败", e);
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

        overlayLog("获取MediaProjection code=" + resultCode);

        try {
            mediaProjection = mpManager.getMediaProjection(resultCode, data);
        } catch (Exception e) {
            overlayLog("getMediaProjection异常:" + e.getMessage());
            Log.e(TAG, "getMediaProjection failed", e);
            stopSelf();
            return;
        }

        if (mediaProjection == null) {
            overlayLog("MediaProjection为null!");
            stopSelf();
            return;
        }

        overlayLog("MediaProjection获取成功");

        mediaProjection.registerCallback(new MediaProjection.Callback() {
            @Override
            public void onStop() {
                overlayLog("MediaProjection已停止");
                stopCapture();
            }
        }, inferHandler);

        imageReader = ImageReader.newInstance(captureWidth, captureHeight, PixelFormat.RGBA_8888, 3);
        overlayLog("ImageReader创建 " + captureWidth + "x" + captureHeight);

        imageReader.setOnImageAvailableListener(reader -> {
            Image image = reader.acquireLatestImage();
            if (image == null) return;

            if (!inferBusy.compareAndSet(false, true)) {
                image.close();
                return;
            }

            try {
                final Image.Plane plane = image.getPlanes()[0];
                final ByteBuffer srcBuffer = plane.getBuffer();
                final int rowStride = plane.getRowStride();
                final int pixelStride = plane.getPixelStride();
                final int w = captureWidth;
                final int h = captureHeight;

                final Bitmap bmp = Bitmap.createBitmap(
                        w + (rowStride - pixelStride * w) / pixelStride, h,
                        Bitmap.Config.ARGB_8888);
                bmp.copyPixelsFromBuffer(srcBuffer);
                image.close();

                final Bitmap cropped;
                if (bmp.getWidth() != w) {
                    cropped = Bitmap.createBitmap(bmp, 0, 0, w, h);
                    bmp.recycle();
                } else {
                    cropped = bmp;
                }

                try {
                    if (!YoloV8Ncnn.nativeIsLoaded()) {
                        cropped.recycle();
                        overlayLog("模型未加载!");
                        return;
                    }

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
                    } else {
                        overlayLog("推理返回null");
                    }
                } catch (Exception e) {
                    overlayLog("推理异常:" + e.getMessage());
                    if (!cropped.isRecycled()) cropped.recycle();
                }
            } catch (Exception e) {
                overlayLog("图像处理异常:" + e.getMessage());
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

        overlayLog("VirtualDisplay创建完成");

        isRunning.set(true);
        fpsStartTime = System.currentTimeMillis();
        frameCount = 0;
        overlayLog("MediaProjection启动 " + captureWidth + "x" + captureHeight);
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
        destroyPersistentSu();
        if (virtualDisplay != null) { virtualDisplay.release(); virtualDisplay = null; }
        if (mediaProjection != null) { mediaProjection.stop(); mediaProjection = null; }
        if (imageReader != null) { imageReader.close(); imageReader = null; }
        Log.i(TAG, "屏幕捕获已停止");
        stopSelf();
    }

    @Override
    public void onDestroy() {
        stopCapture();
        destroyPersistentSu();
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
