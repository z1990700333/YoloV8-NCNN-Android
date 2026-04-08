package com.yolov8ncnn;

import android.accessibilityservice.AccessibilityService;
import android.accessibilityservice.GestureDescription;
import android.graphics.Path;
import android.os.Build;
import android.util.Log;
import android.view.accessibility.AccessibilityEvent;

/**
 * AccessibilityService for performing auto-clicks at specified coordinates.
 * Uses dispatchGesture() API (Android 7.0+).
 */
public class AutoClickService extends AccessibilityService {

    private static final String TAG = "AutoClickService";
    private static AutoClickService sInstance = null;
    private long lastClickTime = 0;

    public static AutoClickService getInstance() { return sInstance; }
    public static boolean isRunning() { return sInstance != null; }

    @Override
    public void onServiceConnected() {
        super.onServiceConnected();
        sInstance = this;
        Log.i(TAG, "AutoClickService connected");
        MainActivity.appendLog("无障碍服务已连接");
    }

    @Override
    public void onAccessibilityEvent(AccessibilityEvent event) {
        // Not used - we only need dispatchGesture
    }

    @Override
    public void onInterrupt() {
        Log.w(TAG, "AutoClickService interrupted");
    }

    @Override
    public void onDestroy() {
        sInstance = null;
        Log.i(TAG, "AutoClickService destroyed");
        super.onDestroy();
    }

    /**
     * Perform a tap at the given screen coordinates.
     */
    public boolean tap(float x, float y, int minIntervalMs) {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.N) return false;

        long now = System.currentTimeMillis();
        if (now - lastClickTime < minIntervalMs) return false;
        lastClickTime = now;

        Path clickPath = new Path();
        clickPath.moveTo(x, y);

        GestureDescription.StrokeDescription stroke =
                new GestureDescription.StrokeDescription(clickPath, 0, 50);
        GestureDescription gesture = new GestureDescription.Builder()
                .addStroke(stroke).build();

        return dispatchGesture(gesture, null, null);
    }
}
