package com.yolov8ncnn;

import android.annotation.SuppressLint;
import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.PixelFormat;
import android.graphics.RectF;
import android.os.Build;
import android.os.Handler;
import android.os.IBinder;
import android.os.Looper;
import android.util.Log;
import android.view.Gravity;
import android.view.LayoutInflater;
import android.view.MotionEvent;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.TextView;

import java.util.ArrayList;
import java.util.List;

public class OverlayService extends Service {

    private static final String TAG = "OverlayService";
    private static final String CHANNEL_ID = "overlay_channel";
    private static final int NOTIFICATION_ID = 1002;

    private static OverlayService sInstance = null;

    private WindowManager windowManager;
    private View panelView;
    private View touchCaptureView;
    private DetectionOverlayView detectionOverlayView;

    private TextView tvFps, tvInferTime, tvDetections, tvClickCoord, tvStatus, tvLog;
    private Button btnToggle, btnSetClick, btnAutoClick;

    private SettingsManager settings;
    private final Handler mainHandler = new Handler(Looper.getMainLooper());

    private boolean isCapturing = false;
    private boolean isPickingCoord = false;
    private boolean autoClickEnabled = false;
    private int captureW = 1, captureH = 1;

    // Log buffer (悬浮窗只显示最近2行简短提示)
    private final List<String> logLines = new ArrayList<>();
    private static final int MAX_LOG_LINES = 2;

    public static OverlayService getInstance() { return sInstance; }

    @Override
    public void onCreate() {
        super.onCreate();
        sInstance = this;
        settings = new SettingsManager(this);
        windowManager = (WindowManager) getSystemService(WINDOW_SERVICE);

        createNotificationChannel();
        startForeground(NOTIFICATION_ID, buildNotification());

        createPanelOverlay();
        createDetectionOverlay();
    }

    public void appendLog(String msg) {
        mainHandler.post(() -> {
            logLines.add(msg);
            while (logLines.size() > MAX_LOG_LINES) logLines.remove(0);
            StringBuilder sb = new StringBuilder();
            for (String l : logLines) sb.append(l).append("\n");
            if (tvLog != null) tvLog.setText(sb.toString().trim());
        });
    }

    @SuppressLint("ClickableViewAccessibility")
    private void createPanelOverlay() {
        panelView = LayoutInflater.from(this).inflate(R.layout.overlay_panel, null);

        tvFps = panelView.findViewById(R.id.tv_fps);
        tvInferTime = panelView.findViewById(R.id.tv_infer_time);
        tvDetections = panelView.findViewById(R.id.tv_detections);
        tvClickCoord = panelView.findViewById(R.id.tv_click_coord);
        tvStatus = panelView.findViewById(R.id.tv_status);
        tvLog = panelView.findViewById(R.id.tv_log);
        btnToggle = panelView.findViewById(R.id.btn_toggle);
        btnSetClick = panelView.findViewById(R.id.btn_set_click);
        btnAutoClick = panelView.findViewById(R.id.btn_auto_click);

        int cx = settings.getClickX();
        int cy = settings.getClickY();
        tvClickCoord.setText(cx >= 0 && cy >= 0 ?
                String.format("点击坐标: (%d, %d)", cx, cy) : "点击坐标: 未设置");

        btnToggle.setOnClickListener(v -> {
            if (isCapturing) {
                stopScreenCapture();
            } else {
                startScreenCaptureMediaProjection();
            }
        });

        btnSetClick.setOnClickListener(v -> {
            if (!isPickingCoord) startCoordPicker();
            else stopCoordPicker();
        });

        autoClickEnabled = settings.getAutoClick();
        updateAutoClickButton();
        btnAutoClick.setOnClickListener(v -> {
            autoClickEnabled = !autoClickEnabled;
            settings.setAutoClick(autoClickEnabled);
            updateAutoClickButton();
        });

        WindowManager.LayoutParams params = new WindowManager.LayoutParams(
                WindowManager.LayoutParams.WRAP_CONTENT,
                WindowManager.LayoutParams.WRAP_CONTENT,
                WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
                WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE,
                PixelFormat.TRANSLUCENT);
        params.gravity = Gravity.TOP | Gravity.START;
        params.x = 0;
        params.y = 100;

        panelView.setOnTouchListener(new View.OnTouchListener() {
            private int initialX, initialY;
            private float initialTouchX, initialTouchY;
            private boolean isDragging = false;

            @Override
            public boolean onTouch(View v, MotionEvent event) {
                switch (event.getAction()) {
                    case MotionEvent.ACTION_DOWN:
                        initialX = params.x; initialY = params.y;
                        initialTouchX = event.getRawX(); initialTouchY = event.getRawY();
                        isDragging = false;
                        return true;
                    case MotionEvent.ACTION_MOVE:
                        float dx = event.getRawX() - initialTouchX;
                        float dy = event.getRawY() - initialTouchY;
                        if (Math.abs(dx) > 10 || Math.abs(dy) > 10) isDragging = true;
                        if (isDragging) {
                            params.x = initialX + (int) dx;
                            params.y = initialY + (int) dy;
                            windowManager.updateViewLayout(panelView, params);
                        }
                        return true;
                    case MotionEvent.ACTION_UP:
                        return isDragging;
                }
                return false;
            }
        });

        windowManager.addView(panelView, params);
    }

    private void createDetectionOverlay() {
        detectionOverlayView = new DetectionOverlayView(this);
        WindowManager.LayoutParams params = new WindowManager.LayoutParams(
                WindowManager.LayoutParams.MATCH_PARENT,
                WindowManager.LayoutParams.MATCH_PARENT,
                WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
                WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE
                        | WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE
                        | WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN
                        | WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS,
                PixelFormat.TRANSLUCENT);
        params.gravity = Gravity.TOP | Gravity.START;
        windowManager.addView(detectionOverlayView, params);
    }

    @SuppressLint("ClickableViewAccessibility")
    private void startCoordPicker() {
        isPickingCoord = true;
        btnSetClick.setText("取消");
        tvStatus.setText("点击屏幕设置点击坐标...");
        tvStatus.setVisibility(View.VISIBLE);

        touchCaptureView = new View(this);
        touchCaptureView.setBackgroundColor(Color.argb(40, 255, 255, 0));

        WindowManager.LayoutParams params = new WindowManager.LayoutParams(
                WindowManager.LayoutParams.MATCH_PARENT,
                WindowManager.LayoutParams.MATCH_PARENT,
                WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
                WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE
                        | WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN,
                PixelFormat.TRANSLUCENT);
        params.gravity = Gravity.TOP | Gravity.START;

        touchCaptureView.setOnTouchListener((v, event) -> {
            if (event.getAction() == MotionEvent.ACTION_DOWN) {
                int x = (int) event.getRawX();
                int y = (int) event.getRawY();
                settings.setClickX(x);
                settings.setClickY(y);
                tvClickCoord.setText(String.format("点击坐标: (%d, %d)", x, y));
                stopCoordPicker();
                return true;
            }
            return false;
        });

        windowManager.addView(touchCaptureView, params);
    }

    private void stopCoordPicker() {
        isPickingCoord = false;
        btnSetClick.setText("设置坐标");
        tvStatus.setVisibility(View.GONE);
        if (touchCaptureView != null) {
            windowManager.removeView(touchCaptureView);
            touchCaptureView = null;
        }
    }

    private void updateAutoClickButton() {
        if (autoClickEnabled) {
            btnAutoClick.setText("自动点击: 开");
            btnAutoClick.setBackgroundColor(Color.parseColor("#4CAF50"));
        } else {
            btnAutoClick.setText("自动点击: 关");
            btnAutoClick.setBackgroundColor(Color.parseColor("#757575"));
        }
    }

    /**
     * MediaProjection 模式：启动透明 Activity 请求屏幕录制权限
     */
    private void startScreenCaptureMediaProjection() {
        if (!YoloV8Ncnn.nativeIsLoaded()) {
            appendLog("请先加载模型!");
            MainActivity.appendLog("请先加载模型!");
            return;
        }
        appendLog("启动MP...");
        MainActivity.appendLog("悬浮窗启动 MediaProjection 模式");
        Intent intent = new Intent(this, MediaProjectionRequestActivity.class);
        intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
        startActivity(intent);
    }

    public void updateDetections(BoxInfo[] boxes, float inferTimeMs, float fps,
                                  int captureW, int captureH) {
        this.captureW = captureW;
        this.captureH = captureH;
        mainHandler.post(() -> {
            tvFps.setText(String.format("帧率: %.1f", fps));
            tvInferTime.setText(String.format("推理: %.1fms", inferTimeMs));
            tvDetections.setText(String.format("检测: %d", boxes != null ? boxes.length : 0));
            if (detectionOverlayView != null && boxes != null) {
                detectionOverlayView.setDetections(boxes, captureW, captureH);
            }
        });
    }

    public void setCapturing(boolean capturing) {
        this.isCapturing = capturing;
        mainHandler.post(() -> {
            btnToggle.setText(capturing ? "停止" : "开始");
            if (!capturing) {
                tvFps.setText("帧率: --");
                tvInferTime.setText("推理: --");
                tvDetections.setText("检测: --");
            }
        });
    }

    private void stopScreenCapture() {
        ScreenCaptureService service = ScreenCaptureService.getInstance();
        if (service != null) service.stopCapture();
        setCapturing(false);
    }

    @Override
    public void onDestroy() {
        if (panelView != null) windowManager.removeView(panelView);
        if (detectionOverlayView != null) windowManager.removeView(detectionOverlayView);
        if (touchCaptureView != null) windowManager.removeView(touchCaptureView);
        sInstance = null;
        super.onDestroy();
    }

    @Override public IBinder onBind(Intent intent) { return null; }

    private void createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            NotificationChannel channel = new NotificationChannel(
                    CHANNEL_ID, "悬浮窗", NotificationManager.IMPORTANCE_LOW);
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
                .setContentTitle("YOLOv8 悬浮窗")
                .setContentText("检测悬浮窗运行中")
                .setSmallIcon(android.R.drawable.ic_menu_view)
                .setOngoing(true)
                .build();
    }

    // ═══════════════════════════════════════════════════════════════════════
    static class DetectionOverlayView extends View {
        private BoxInfo[] boxes = new BoxInfo[0];
        private int srcW = 1, srcH = 1;

        private final Paint boxPaint = new Paint();
        private final Paint textPaint = new Paint();
        private final Paint bgPaint = new Paint();
        private final Paint clickPaint = new Paint();
        private final RectF rect = new RectF();

        public DetectionOverlayView(Context context) {
            super(context);
            setWillNotDraw(false);
            boxPaint.setColor(Color.RED); boxPaint.setStyle(Paint.Style.STROKE);
            boxPaint.setStrokeWidth(3f); boxPaint.setAntiAlias(true);
            textPaint.setColor(Color.WHITE); textPaint.setTextSize(28f); textPaint.setAntiAlias(true);
            bgPaint.setColor(Color.argb(180, 255, 0, 0)); bgPaint.setStyle(Paint.Style.FILL);
            clickPaint.setColor(Color.argb(200, 0, 255, 0)); clickPaint.setStyle(Paint.Style.STROKE);
            clickPaint.setStrokeWidth(4f); clickPaint.setAntiAlias(true);
        }

        public void setDetections(BoxInfo[] newBoxes, int w, int h) {
            this.boxes = newBoxes; this.srcW = w; this.srcH = h;
            postInvalidate();
        }

        @Override
        protected void onDraw(Canvas canvas) {
            super.onDraw(canvas);
            float scaleX = (float) getWidth() / srcW;
            float scaleY = (float) getHeight() / srcH;

            for (BoxInfo box : boxes) {
                float l = box.x1 * scaleX, t = box.y1 * scaleY;
                float r = box.x2 * scaleX, b = box.y2 * scaleY;
                rect.set(l, t, r, b);
                canvas.drawRect(rect, boxPaint);
                String label = String.format("L%d %.0f%%", box.label, box.score * 100);
                float textW = textPaint.measureText(label);
                canvas.drawRect(l, t - 32f, l + textW + 8f, t, bgPaint);
                canvas.drawText(label, l + 4f, t - 6f, textPaint);
            }

            SettingsManager sm = new SettingsManager(getContext());
            int cx = sm.getClickX(); int cy = sm.getClickY();
            if (cx >= 0 && cy >= 0) {
                canvas.drawCircle(cx, cy, 20f, clickPaint);
                canvas.drawLine(cx - 30, cy, cx + 30, cy, clickPaint);
                canvas.drawLine(cx, cy - 30, cx, cy + 30, clickPaint);
            }
        }
    }
}
