package com.yolov8ncnn;

import android.annotation.SuppressLint;
import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.Service;
import android.content.Intent;
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

    // Root: persistent su shell
    private Process suProcess;
    private OutputStream suOut;
    private InputStream suIn;
    // 复用 buffer 避免 GC
    private byte[] rootPixelBuf;
    private ByteBuffer rootDirectBuf;

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
                log("MediaProjection数据无效");
                stopSelf();
                return START_NOT_STICKY;
            }
            startMediaProjectionCapture(resultCode, data);
        }
        return START_STICKY;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Root: 持久 su shell + screencap raw, 复用 buffer
    // ═══════════════════════════════════════════════════════════════════════

    private boolean initSuShell() {
        try {
            destroySu();
            suProcess = Runtime.getRuntime().exec("su");
            suOut = suProcess.getOutputStream();
            suIn = suProcess.getInputStream();
            // 测试 shell 是否可用
            suOut.write("echo READY\n".getBytes());
            suOut.flush();
            byte[] tmp = new byte[64];
            long t = System.currentTimeMillis();
            StringBuilder sb = new StringBuilder();
            while (System.currentTimeMillis() - t < 3000) {
                if (suIn.available() > 0) {
                    int n = suIn.read(tmp);
                    if (n > 0) sb.append(new String(tmp, 0, n));
                    if (sb.toString().contains("READY")) {
                        log("su shell就绪");
                        return true;
                    }
                } else {
                    Thread.sleep(5);
                }
            }
            log("su shell超时");
            return false;
        } catch (Exception e) {
            log("su shell失败:" + e.getMessage());
            return false;
        }
    }

    private void startRootCapture() {
        isRunning.set(true);
        fpsStartTime = System.currentTimeMillis();
        frameCount = 0;

        if (!initSuShell()) {
            log("无法创建su shell,停止");
            stopSelf();
            return;
        }

        log("Root截图启动(持久shell)");
        inferHandler.post(this::rootCaptureLoop);
    }

    private void rootCaptureLoop() {
        if (!isRunning.get()) return;

        try {
            long t0 = System.currentTimeMillis();

            // 通过持久 su shell 执行 screencap (raw格式)
            // screencap 输出: 4字节w + 4字节h + 4字节fmt + RGBA像素
            suOut.write("screencap\n".getBytes());
            suOut.flush();

            DataInputStream dis = new DataInputStream(suIn);

            int w = Integer.reverseBytes(dis.readInt());
            int h = Integer.reverseBytes(dis.readInt());
            int fmt = Integer.reverseBytes(dis.readInt());

            int dataSize = w * h * 4;

            // 复用 buffer
            if (rootPixelBuf == null || rootPixelBuf.length != dataSize) {
                rootPixelBuf = new byte[dataSize];
                rootDirectBuf = ByteBuffer.allocateDirect(dataSize);
                log("分配buffer:" + w + "x" + h + " " + (dataSize/1024) + "KB");
            }

            int offset = 0;
            while (offset < dataSize) {
                int n = dis.read(rootPixelBuf, offset, Math.min(65536, dataSize - offset));
                if (n <= 0) break;
                offset += n;
            }

            long captureMs = System.currentTimeMillis() - t0;

            if (offset < dataSize) {
                log("数据不完整:" + offset + "/" + dataSize + " 重建shell");
                initSuShell();
            } else if (YoloV8Ncnn.nativeIsLoaded()) {
                rootDirectBuf.rewind();
                rootDirectBuf.put(rootPixelBuf);
                rootDirectBuf.rewind();

                float[] rawResult = YoloV8Ncnn.nativeDetectBuffer(
                        rootDirectBuf, w, h, w * 4,
                        settings.getConfThresh(), settings.getNmsThresh());

                if (rawResult != null) {
                    float inferMs = YoloV8Ncnn.getInferTime(rawResult);
                    BoxInfo[] boxes = YoloV8Ncnn.parseResult(rawResult);
                    updateFps();
                    handleAutoClick(boxes);

                    log("cap:" + captureMs + " inf:" + String.format("%.0f", inferMs)
                            + " det:" + boxes.length + " fps:" + String.format("%.1f", currentFps));

                    DetectionCallback cb = callback;
                    if (cb != null) cb.onDetectionResult(boxes, inferMs, currentFps, w, h);
                } else {
                    log("推理null cap:" + captureMs + "ms");
                }
            } else {
                log("模型未加载!");
            }
        } catch (Exception e) {
            log("Root异常:" + e.getMessage());
            // 重建 shell
            try { initSuShell(); } catch (Exception ignored) {}
        }

        if (isRunning.get()) {
            inferHandler.post(this::rootCaptureLoop);
        }
    }

    public static boolean checkRootAccess() {
        try {
            Process p = Runtime.getRuntime().exec(new String[]{"su", "-c", "id"});
            BufferedReader r = new BufferedReader(new InputStreamReader(p.getInputStream()));
            String line = r.readLine(); p.waitFor();
            return line != null && line.contains("uid=0");
        } catch (Exception e) { return false; }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // MediaProjection: Image ByteBuffer → C++ 零拷贝
    // ═══════════════════════════════════════════════════════════════════════

    @SuppressLint("WrongConstant")
    private void startMediaProjectionCapture(int resultCode, Intent data) {
        log("MediaProjection启动");

        MediaProjectionManager mpManager =
                (MediaProjectionManager) getSystemService(MEDIA_PROJECTION_SERVICE);
        try {
            mediaProjection = mpManager.getMediaProjection(resultCode, data);
        } catch (Exception e) {
            log("getMediaProjection异常:" + e.getMessage());
            stopSelf(); return;
        }
        if (mediaProjection == null) {
            log("MediaProjection=null!");
            stopSelf(); return;
        }

        mediaProjection.registerCallback(new MediaProjection.Callback() {
            @Override public void onStop() { log("MP停止"); stopCapture(); }
        }, inferHandler);

        imageReader = ImageReader.newInstance(captureWidth, captureHeight, PixelFormat.RGBA_8888, 3);

        imageReader.setOnImageAvailableListener(reader -> {
            Image image = reader.acquireLatestImage();
            if (image == null) return;

            if (!inferBusy.compareAndSet(false, true)) {
                image.close(); return;
            }

            try {
                if (!YoloV8Ncnn.nativeIsLoaded()) {
                    image.close(); inferBusy.set(false);
                    return;
                }

                Image.Plane plane = image.getPlanes()[0];
                ByteBuffer buffer = plane.getBuffer();
                int rowStride = plane.getRowStride();

                float[] rawResult = YoloV8Ncnn.nativeDetectBuffer(
                        buffer, captureWidth, captureHeight, rowStride,
                        settings.getConfThresh(), settings.getNmsThresh());

                image.close();

                if (rawResult != null) {
                    float inferMs = YoloV8Ncnn.getInferTime(rawResult);
                    BoxInfo[] boxes = YoloV8Ncnn.parseResult(rawResult);
                    updateFps();
                    handleAutoClick(boxes);

                    DetectionCallback cb = callback;
                    if (cb != null) cb.onDetectionResult(boxes, inferMs, currentFps,
                            captureWidth, captureHeight);
                }
            } catch (Exception e) {
                log("MP异常:" + e.getMessage());
                try { image.close(); } catch (Exception ignored) {}
            } finally {
                inferBusy.set(false);
            }
        }, inferHandler);

        virtualDisplay = mediaProjection.createVirtualDisplay("YoloV8Capture",
                captureWidth, captureHeight, screenDpi,
                DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
                imageReader.getSurface(), null, inferHandler);

        isRunning.set(true);
        fpsStartTime = System.currentTimeMillis();
        frameCount = 0;
        log("MP就绪 " + captureWidth + "x" + captureHeight);
    }

    // ═══════════════════════════════════════════════════════════════════════

    private void updateFps() {
        frameCount++;
        long now = System.currentTimeMillis();
        if (fpsStartTime == 0) fpsStartTime = now;
        long elapsed = now - fpsStartTime;
        if (elapsed >= 1000) {
            currentFps = frameCount * 1000.0f / elapsed;
            frameCount = 0; fpsStartTime = now;
        }
    }

    private void handleAutoClick(BoxInfo[] boxes) {
        if (!settings.getAutoClick() || boxes == null || boxes.length == 0) return;
        if (!AutoClickService.isRunning()) return;
        int tl = settings.getTargetLabel(), cx = settings.getClickX(), cy = settings.getClickY();
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
        destroySu();
        log("已停止");
        stopSelf();
    }

    private void destroySu() {
        try { if (suOut != null) { suOut.write("exit\n".getBytes()); suOut.flush(); suOut.close(); } } catch (Exception ignored) {}
        try { if (suProcess != null) suProcess.destroy(); } catch (Exception ignored) {}
        suProcess = null; suOut = null; suIn = null;
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
            getSystemService(NotificationManager.class).createNotificationChannel(ch);
        }
    }

    private Notification buildNotification() {
        Notification.Builder b = Build.VERSION.SDK_INT >= Build.VERSION_CODES.O
                ? new Notification.Builder(this, CHANNEL_ID) : new Notification.Builder(this);
        return b.setContentTitle("YOLOv8 NCNN").setContentText("推理中...")
                .setSmallIcon(android.R.drawable.ic_menu_camera).setOngoing(true).build();
    }
}
