package com.yolov8ncnn;

import android.annotation.SuppressLint;
import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.Service;
import android.content.Context;
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

import java.nio.ByteBuffer;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Foreground service that captures the screen using MediaProjection
 * and runs YOLOv8 inference on each frame via NCNN (C++).
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

    // State
    private final AtomicBoolean isRunning = new AtomicBoolean(false);
    private final AtomicBoolean inferBusy = new AtomicBoolean(false);

    // Results callback
    private volatile DetectionCallback callback;

    // Screen dimensions
    private int screenWidth;
    private int screenHeight;
    private int screenDpi;
    private int captureWidth;
    private int captureHeight;

    // FPS tracking
    private long frameCount = 0;
    private long fpsStartTime = 0;
    private float currentFps = 0;

    public interface DetectionCallback {
        void onDetectionResult(BoxInfo[] boxes, float inferTimeMs, float fps,
                               int captureW, int captureH);
    }

    public static ScreenCaptureService getInstance() {
        return sInstance;
    }

    public static boolean isServiceRunning() {
        return sInstance != null && sInstance.isRunning.get();
    }

    public void setCallback(DetectionCallback cb) {
        this.callback = cb;
    }

    @Override
    public void onCreate() {
        super.onCreate();
        sInstance = this;
        settings = new SettingsManager(this);

        // Get screen metrics
        WindowManager wm = (WindowManager) getSystemService(WINDOW_SERVICE);
        DisplayMetrics dm = new DisplayMetrics();
        wm.getDefaultDisplay().getRealMetrics(dm);
        screenWidth = dm.widthPixels;
        screenHeight = dm.heightPixels;
        screenDpi = dm.densityDpi;

        // Capture at reduced resolution for speed
        float scale = settings.getCaptureScale();
        captureWidth = (int)(screenWidth * scale);
        captureHeight = (int)(screenHeight * scale);

        // Ensure even dimensions
        captureWidth = (captureWidth / 2) * 2;
        captureHeight = (captureHeight / 2) * 2;

        Log.i(TAG, String.format("Screen: %dx%d, Capture: %dx%d, Scale: %.2f",
                screenWidth, screenHeight, captureWidth, captureHeight, scale));

        // Create inference thread
        inferThread = new HandlerThread("InferThread", Thread.MAX_PRIORITY);
        inferThread.start();
        inferHandler = new Handler(inferThread.getLooper());

        createNotificationChannel();
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        if (intent == null) {
            stopSelf();
            return START_NOT_STICKY;
        }

        // Start foreground immediately
        startForeground(NOTIFICATION_ID, buildNotification());

        int resultCode = intent.getIntExtra(EXTRA_RESULT_CODE, -1);
        Intent data = intent.getParcelableExtra(EXTRA_DATA);

        if (resultCode == -1 || data == null) {
            Log.e(TAG, "Invalid MediaProjection data");
            stopSelf();
            return START_NOT_STICKY;
        }

        startCapture(resultCode, data);
        return START_STICKY;
    }

    @SuppressLint("WrongConstant")
    private void startCapture(int resultCode, Intent data) {
        MediaProjectionManager mpManager =
                (MediaProjectionManager) getSystemService(MEDIA_PROJECTION_SERVICE);
        mediaProjection = mpManager.getMediaProjection(resultCode, data);

        if (mediaProjection == null) {
            Log.e(TAG, "MediaProjection is null");
            stopSelf();
            return;
        }

        // Register stop callback
        mediaProjection.registerCallback(new MediaProjection.Callback() {
            @Override
            public void onStop() {
                Log.i(TAG, "MediaProjection stopped");
                stopCapture();
            }
        }, inferHandler);

        // Create ImageReader with reduced resolution
        imageReader = ImageReader.newInstance(
                captureWidth, captureHeight,
                PixelFormat.RGBA_8888, 2);

        imageReader.setOnImageAvailableListener(reader -> {
            Image image = reader.acquireLatestImage();
            if (image == null) return;

            // Skip if previous inference still running (drop frame)
            if (!inferBusy.compareAndSet(false, true)) {
                image.close();
                return;
            }

            // Get pixel data from image
            final Image.Plane plane = image.getPlanes()[0];
            final ByteBuffer buffer = plane.getBuffer();
            final int rowStride = plane.getRowStride();
            final int w = captureWidth;
            final int h = captureHeight;

            // Run inference on dedicated thread
            inferHandler.post(() -> {
                try {
                    if (!YoloV8Ncnn.nativeIsLoaded()) {
                        return;
                    }

                    float confThresh = settings.getConfThresh();
                    float nmsThresh = settings.getNmsThresh();

                    // Call native detection with direct buffer
                    float[] rawResult = YoloV8Ncnn.nativeDetectBuffer(
                            buffer, w, h, rowStride,
                            confThresh, nmsThresh);

                    if (rawResult != null) {
                        float inferTimeMs = YoloV8Ncnn.getInferTime(rawResult);
                        BoxInfo[] boxes = YoloV8Ncnn.parseResult(rawResult);

                        // Update FPS
                        frameCount++;
                        long now = System.currentTimeMillis();
                        if (fpsStartTime == 0) fpsStartTime = now;
                        long elapsed = now - fpsStartTime;
                        if (elapsed >= 1000) {
                            currentFps = frameCount * 1000.0f / elapsed;
                            frameCount = 0;
                            fpsStartTime = now;
                        }

                        // Auto-click logic
                        handleAutoClick(boxes);

                        // Notify callback
                        DetectionCallback cb = callback;
                        if (cb != null) {
                            cb.onDetectionResult(boxes, inferTimeMs, currentFps, w, h);
                        }
                    }
                } catch (Exception e) {
                    Log.e(TAG, "Inference error", e);
                } finally {
                    image.close();
                    inferBusy.set(false);
                }
            });
        }, inferHandler);

        // Create virtual display
        virtualDisplay = mediaProjection.createVirtualDisplay(
                "YoloV8Capture",
                captureWidth, captureHeight, screenDpi,
                DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
                imageReader.getSurface(),
                null, null);

        isRunning.set(true);
        fpsStartTime = System.currentTimeMillis();
        frameCount = 0;
        Log.i(TAG, "Screen capture started");
    }

    private void handleAutoClick(BoxInfo[] boxes) {
        if (!settings.getAutoClick()) return;
        if (boxes == null || boxes.length == 0) return;
        if (!AutoClickService.isRunning()) return;

        int targetLabel = settings.getTargetLabel();
        int clickX = settings.getClickX();
        int clickY = settings.getClickY();

        if (clickX < 0 || clickY < 0) return;

        // Check if any detection matches target label
        for (BoxInfo box : boxes) {
            if (targetLabel == -1 || box.label == targetLabel) {
                // Scale detection coords from capture resolution to screen resolution
                // (not needed for click coords - they're already in screen space)
                AutoClickService service = AutoClickService.getInstance();
                if (service != null) {
                    service.tap(clickX, clickY, settings.getClickDelay());
                }
                break; // Only click once per frame
            }
        }
    }

    public void stopCapture() {
        isRunning.set(false);

        if (virtualDisplay != null) {
            virtualDisplay.release();
            virtualDisplay = null;
        }
        if (mediaProjection != null) {
            mediaProjection.stop();
            mediaProjection = null;
        }
        if (imageReader != null) {
            imageReader.close();
            imageReader = null;
        }

        Log.i(TAG, "Screen capture stopped");
        stopSelf();
    }

    public void updateCaptureScale(float scale) {
        captureWidth = (int)(screenWidth * scale) / 2 * 2;
        captureHeight = (int)(screenHeight * scale) / 2 * 2;
        // Would need to recreate virtual display to apply
    }

    public int getScreenWidth() { return screenWidth; }
    public int getScreenHeight() { return screenHeight; }
    public float getCurrentFps() { return currentFps; }

    @Override
    public void onDestroy() {
        stopCapture();
        if (inferThread != null) {
            inferThread.quitSafely();
            inferThread = null;
        }
        sInstance = null;
        super.onDestroy();
    }

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }

    private void createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            NotificationChannel channel = new NotificationChannel(
                    CHANNEL_ID,
                    "Screen Capture",
                    NotificationManager.IMPORTANCE_LOW);
            channel.setDescription("YOLOv8 screen capture inference");
            NotificationManager nm = getSystemService(NotificationManager.class);
            nm.createNotificationChannel(channel);
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
                .setContentText("Screen inference running...")
                .setSmallIcon(android.R.drawable.ic_menu_camera)
                .setOngoing(true)
                .build();
    }
}
