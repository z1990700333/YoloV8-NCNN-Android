package com.yolov8ncnn;

import android.app.Activity;
import android.content.Intent;
import android.media.projection.MediaProjectionManager;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;

/**
 * Transparent Activity solely for requesting MediaProjection consent.
 * Can be launched from OverlayService (which cannot call startActivityForResult).
 * After user grants/denies, starts ScreenCaptureService and finishes itself.
 */
public class MediaProjectionRequestActivity extends Activity {

    private static final String TAG = "MPRequestActivity";
    private static final int REQUEST_CODE = 9999;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        Log.i(TAG, "onCreate - requesting MediaProjection consent");

        // Check model loaded
        if (!YoloV8Ncnn.nativeIsLoaded()) {
            Log.e(TAG, "Model not loaded!");
            logToOverlay("请先加载模型!");
            finish();
            return;
        }

        MediaProjectionManager mpManager =
                (MediaProjectionManager) getSystemService(MEDIA_PROJECTION_SERVICE);
        try {
            Intent captureIntent = mpManager.createScreenCaptureIntent();
            startActivityForResult(captureIntent, REQUEST_CODE);
            Log.i(TAG, "MediaProjection consent dialog launched");
            logToOverlay("正在请求屏幕录制权限...");
        } catch (Exception e) {
            Log.e(TAG, "Failed to create capture intent: " + e.getMessage());
            logToOverlay("创建截图请求失败: " + e.getMessage());
            finish();
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        Log.i(TAG, "onActivityResult: requestCode=" + requestCode
                + " resultCode=" + resultCode + " data=" + data);

        if (requestCode == REQUEST_CODE) {
            if (resultCode == RESULT_OK && data != null) {
                Log.i(TAG, "MediaProjection consent GRANTED");
                logToOverlay("屏幕录制授权成功!");

                // Start ScreenCaptureService with MediaProjection data
                Intent serviceIntent = new Intent(this, ScreenCaptureService.class);
                serviceIntent.putExtra(ScreenCaptureService.EXTRA_RESULT_CODE, resultCode);
                serviceIntent.putExtra(ScreenCaptureService.EXTRA_DATA, data);

                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                    startForegroundService(serviceIntent);
                } else {
                    startService(serviceIntent);
                }

                // Wait for service to start and set callback
                new Thread(() -> {
                    for (int i = 0; i < 50; i++) {
                        try { Thread.sleep(100); } catch (InterruptedException ignored) {}
                        ScreenCaptureService svc = ScreenCaptureService.getInstance();
                        if (svc != null && ScreenCaptureService.isServiceRunning()) {
                            Log.i(TAG, "ScreenCaptureService ready after " + (i * 100) + "ms");
                            svc.setCallback((boxes, inferTimeMs, fps, captureW, captureH) -> {
                                OverlayService overlay = OverlayService.getInstance();
                                if (overlay != null) {
                                    overlay.updateDetections(boxes, inferTimeMs, fps, captureW, captureH);
                                }
                            });
                            OverlayService overlay = OverlayService.getInstance();
                            if (overlay != null) overlay.setCapturing(true);
                            logToOverlay("MediaProjection 截图已启动!");
                            return;
                        }
                    }
                    Log.e(TAG, "ScreenCaptureService did not start in time");
                    logToOverlay("服务启动超时!");
                }).start();

            } else {
                Log.w(TAG, "MediaProjection consent DENIED or cancelled");
                logToOverlay("用户拒绝了屏幕录制权限");
            }
        }

        finish();
    }

    private void logToOverlay(String msg) {
        OverlayService ov = OverlayService.getInstance();
        if (ov != null) ov.appendLog(msg);
    }
}
