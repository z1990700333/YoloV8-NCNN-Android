package com.yolov8ncnn;

import android.accessibilityservice.AccessibilityService;
import android.accessibilityservice.AccessibilityServiceInfo;
import android.accessibilityservice.GestureDescription;
import android.content.Intent;
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

    public static AutoClickService getInstance() {
        return sInstance;
    }

    public static boolean isRunning() {
        return sInstance != null;
    }

    @Override
    public void onServiceConnected() {
        super.onServiceConnected();
        sInstance = this;

        AccessibilityServiceInfo info = new AccessibilityServiceInfo();
        info.eventTypes = AccessibilityEvent.TYPES_ALL_MASK;
        info.feedbackType = AccessibilityServiceInfo.FEEDBACK_GENERIC;
        info.notificationTimeout = 100;
        info.flags = AccessibilityServiceInfo.FLAG_REQUEST_ENHANCED_WEB_ACCESSIBILITY
                | AccessibilityServiceInfo.FLAG_REPORT_VIEW_IDS;
        setServiceInfo(info);

        Log.i(TAG, "AutoClickService connected");
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
     * Returns true if the gesture was dispatched successfully.
     */
    public boolean tap(float x, float y, int minIntervalMs) {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.N) {
            Log.e(TAG, "dispatchGesture requires API 24+");
            return false;
        }

        long now = System.currentTimeMillis();
        if (now - lastClickTime < minIntervalMs) {
            return false; // Rate limit
        }
        lastClickTime = now;

        Path clickPath = new Path();
        clickPath.moveTo(x, y);

        GestureDescription.StrokeDescription stroke =
                new GestureDescription.StrokeDescription(clickPath, 0, 50);

        GestureDescription gesture = new GestureDescription.Builder()
                .addStroke(stroke)
                .build();

        boolean dispatched = dispatchGesture(gesture, new GestureResultCallback() {
            @Override
            public void onCompleted(GestureDescription gestureDescription) {
                Log.d(TAG, String.format("Tap completed at (%.0f, %.0f)", x, y));
            }

            @Override
            public void onCancelled(GestureDescription gestureDescription) {
                Log.w(TAG, String.format("Tap cancelled at (%.0f, %.0f)", x, y));
            }
        }, null);

        return dispatched;
    }

    /**
     * Perform a swipe gesture.
     */
    public boolean swipe(float x1, float y1, float x2, float y2, long durationMs) {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.N) return false;

        Path swipePath = new Path();
        swipePath.moveTo(x1, y1);
        swipePath.lineTo(x2, y2);

        GestureDescription.StrokeDescription stroke =
                new GestureDescription.StrokeDescription(swipePath, 0, durationMs);

        GestureDescription gesture = new GestureDescription.Builder()
                .addStroke(stroke)
                .build();

        return dispatchGesture(gesture, null, null);
    }
}
