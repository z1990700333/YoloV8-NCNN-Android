package com.yolov8ncnn;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import java.nio.ByteBuffer;

/**
 * JNI bridge to native NCNN YOLOv8 inference engine.
 * All heavy computation runs in C++ for maximum performance.
 */
public class YoloV8Ncnn {

    static {
        System.loadLibrary("yolov8ncnn");
    }

    // ── Native methods ──────────────────────────────────────────────────

    /** Initialize NCNN engine (call once at app start) */
    public static native void nativeInit();

    /** Destroy NCNN engine (call at app exit) */
    public static native void nativeDestroy();

    /** Load model from APK assets */
    public static native boolean nativeLoadModel(
            AssetManager assetManager,
            String paramPath, String binPath,
            int targetSize, boolean useGpu, int numThreads);

    /** Load model from file system path (for custom models) */
    public static native boolean nativeLoadModelPath(
            String paramPath, String binPath,
            int targetSize, boolean useGpu, int numThreads);

    /**
     * Detect objects in a Bitmap.
     * Returns float[]: [inferTimeMs, numBoxes, x1,y1,x2,y2,score,label, ...]
     */
    public static native float[] nativeDetectBitmap(
            Bitmap bitmap, float confThresh, float nmsThresh);

    /**
     * Detect objects from raw pixel buffer (fastest path).
     * pixelBuffer: direct ByteBuffer with RGBA pixels
     * Returns float[]: [inferTimeMs, numBoxes, x1,y1,x2,y2,score,label, ...]
     */
    public static native float[] nativeDetectBuffer(
            ByteBuffer pixelBuffer, int width, int height, int rowStride,
            float confThresh, float nmsThresh);

    /** Check if model is loaded */
    public static native boolean nativeIsLoaded();

    /** Check if GPU is available */
    public static native boolean nativeHasGpu();

    /**
     * Capture screen via /dev/graphics/fb0 (root required, mmap, very fast).
     * outBuffer: pre-allocated DirectByteBuffer for pixel data.
     * Returns int[3] = {width, height, bytesPerPixel} or null on failure.
     */
    public static native int[] nativeCaptureFramebuffer(ByteBuffer outBuffer);

    // ── Java helper methods ─────────────────────────────────────────────

    /**
     * Parse raw float array result into BoxInfo array.
     */
    public static BoxInfo[] parseResult(float[] raw) {
        if (raw == null || raw.length < 2) return new BoxInfo[0];

        int numBoxes = (int) raw[1];
        BoxInfo[] boxes = new BoxInfo[numBoxes];
        for (int i = 0; i < numBoxes; i++) {
            int offset = 2 + i * 6;
            boxes[i] = new BoxInfo(
                    raw[offset], raw[offset + 1],
                    raw[offset + 2], raw[offset + 3],
                    raw[offset + 4], (int) raw[offset + 5]);
        }
        return boxes;
    }

    /**
     * Get inference time from raw result.
     */
    public static float getInferTime(float[] raw) {
        if (raw == null || raw.length < 1) return -1;
        return raw[0];
    }
}
