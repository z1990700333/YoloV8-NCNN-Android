// yolov8ncnn_jni.cpp
// High-performance YOLOv8 NCNN inference engine with JNI bridge
// All inference logic in C++ for maximum speed

#include <jni.h>
#include <android/log.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/bitmap.h>

#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <mutex>

// NCNN headers
#include "net.h"
#include "cpu.h"

#if NCNN_VULKAN
#include "gpu.h"
#endif

#define TAG "YoloV8Ncnn"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

// ═══════════════════════════════════════════════════════════════════════════
// Data structures
// ═══════════════════════════════════════════════════════════════════════════

struct BoxInfo {
    float x1, y1, x2, y2;
    float score;
    int label;
};

// ═══════════════════════════════════════════════════════════════════════════
// NMS (Non-Maximum Suppression) - pure C++
// ═══════════════════════════════════════════════════════════════════════════

static inline float intersection_area(const BoxInfo& a, const BoxInfo& b) {
    float ix1 = std::max(a.x1, b.x1);
    float iy1 = std::max(a.y1, b.y1);
    float ix2 = std::min(a.x2, b.x2);
    float iy2 = std::min(a.y2, b.y2);
    float iw = std::max(0.0f, ix2 - ix1);
    float ih = std::max(0.0f, iy2 - iy1);
    return iw * ih;
}

static inline float box_area(const BoxInfo& b) {
    return (b.x2 - b.x1) * (b.y2 - b.y1);
}

static void nms_sorted(std::vector<BoxInfo>& boxes, float nms_threshold) {
    std::vector<BoxInfo> result;
    std::vector<bool> suppressed(boxes.size(), false);

    for (size_t i = 0; i < boxes.size(); i++) {
        if (suppressed[i]) continue;
        result.push_back(boxes[i]);

        float area_i = box_area(boxes[i]);
        for (size_t j = i + 1; j < boxes.size(); j++) {
            if (suppressed[j]) continue;
            if (boxes[i].label != boxes[j].label) continue;

            float inter = intersection_area(boxes[i], boxes[j]);
            float area_j = box_area(boxes[j]);
            float iou = inter / (area_i + area_j - inter + 1e-5f);

            if (iou > nms_threshold) {
                suppressed[j] = true;
            }
        }
    }
    boxes = std::move(result);
}

// ═══════════════════════════════════════════════════════════════════════════
// YoloV8 Detector class
// ═══════════════════════════════════════════════════════════════════════════

class YoloV8Detector {
public:
    YoloV8Detector() = default;
    ~YoloV8Detector() { net_.clear(); }

    bool loadModel(AAssetManager* mgr, const char* paramPath, const char* binPath,
                   int targetSize, bool useGpu, int numThreads) {
        std::lock_guard<std::mutex> lock(mutex_);

        net_.clear();
        blob_pool_.clear();
        workspace_pool_.clear();

        targetSize_ = targetSize;
        numThreads_ = numThreads;

        // Use big cores for performance
        ncnn::set_cpu_powersave(2);
        ncnn::set_omp_num_threads(numThreads);

        net_.opt = ncnn::Option();
        net_.opt.lightmode = true;
        net_.opt.num_threads = numThreads;
        net_.opt.blob_allocator = &blob_pool_;
        net_.opt.workspace_allocator = &workspace_pool_;
        net_.opt.use_packing_layout = true;

#if NCNN_VULKAN
        hasGpu_ = useGpu && (ncnn::get_gpu_count() > 0);
        net_.opt.use_vulkan_compute = hasGpu_;
        if (hasGpu_) {
            net_.opt.use_fp16_packed = true;
            net_.opt.use_fp16_storage = true;
            net_.opt.use_fp16_arithmetic = true;
            LOGI("Vulkan GPU acceleration enabled");
        }
#else
        hasGpu_ = false;
#endif

        // FP16 on CPU for ARM NEON acceleration
        net_.opt.use_fp16_packed = true;
        net_.opt.use_fp16_storage = true;

        int ret = net_.load_param(mgr, paramPath);
        if (ret != 0) {
            LOGE("load_param from assets failed: %d", ret);
            return false;
        }
        ret = net_.load_model(mgr, binPath);
        if (ret != 0) {
            LOGE("load_model from assets failed: %d", ret);
            return false;
        }

        loaded_ = true;
        LOGI("Model loaded from assets: %s / %s (target=%d, gpu=%d, threads=%d)",
             paramPath, binPath, targetSize, (int)hasGpu_, numThreads);
        return true;
    }

    bool loadModelFromPath(const char* paramPath, const char* binPath,
                           int targetSize, bool useGpu, int numThreads) {
        std::lock_guard<std::mutex> lock(mutex_);

        net_.clear();
        blob_pool_.clear();
        workspace_pool_.clear();

        targetSize_ = targetSize;
        numThreads_ = numThreads;

        ncnn::set_cpu_powersave(2);
        ncnn::set_omp_num_threads(numThreads);

        net_.opt = ncnn::Option();
        net_.opt.lightmode = true;
        net_.opt.num_threads = numThreads;
        net_.opt.blob_allocator = &blob_pool_;
        net_.opt.workspace_allocator = &workspace_pool_;
        net_.opt.use_packing_layout = true;

#if NCNN_VULKAN
        hasGpu_ = useGpu && (ncnn::get_gpu_count() > 0);
        net_.opt.use_vulkan_compute = hasGpu_;
        if (hasGpu_) {
            net_.opt.use_fp16_packed = true;
            net_.opt.use_fp16_storage = true;
            net_.opt.use_fp16_arithmetic = true;
            LOGI("Vulkan GPU acceleration enabled");
        }
#else
        hasGpu_ = false;
#endif

        net_.opt.use_fp16_packed = true;
        net_.opt.use_fp16_storage = true;

        int ret = net_.load_param(paramPath);
        if (ret != 0) {
            LOGE("load_param from path failed: %s, ret=%d", paramPath, ret);
            return false;
        }
        ret = net_.load_model(binPath);
        if (ret != 0) {
            LOGE("load_model from path failed: %s, ret=%d", binPath, ret);
            return false;
        }

        loaded_ = true;
        LOGI("Model loaded from path: %s / %s (target=%d, gpu=%d, threads=%d)",
             paramPath, binPath, targetSize, (int)hasGpu_, numThreads);
        return true;
    }

    // Main detection function - operates on raw RGBA pixel buffer
    // Returns detection results + inference time in ms
    std::vector<BoxInfo> detect(const unsigned char* pixels, int pixelType,
                                int imgW, int imgH,
                                float confThresh, float nmsThresh,
                                float& inferTimeMs) {
        std::vector<BoxInfo> results;
        if (!loaded_) return results;

        auto t0 = std::chrono::high_resolution_clock::now();

        // ── Pre-processing: letterbox resize ─────────────────────────────
        float scale = std::min((float)targetSize_ / imgW,
                               (float)targetSize_ / imgH);
        int scaledW = (int)(imgW * scale);
        int scaledH = (int)(imgH * scale);
        int padW = (targetSize_ - scaledW) / 2;
        int padH = (targetSize_ - scaledH) / 2;

        // Convert pixel format and resize in one step
        ncnn::Mat in = ncnn::Mat::from_pixels_resize(
            pixels, pixelType, imgW, imgH, scaledW, scaledH);

        // Letterbox padding
        ncnn::Mat in_pad;
        ncnn::copy_make_border(in, in_pad,
            padH, targetSize_ - scaledH - padH,
            padW, targetSize_ - scaledW - padW,
            ncnn::BORDER_CONSTANT, 114.0f);

        // Normalize to [0, 1]
        const float norm_vals[3] = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
        in_pad.substract_mean_normalize(nullptr, norm_vals);

        // ── Inference ────────────────────────────────────────────────────
        ncnn::Extractor ex = net_.create_extractor();
        ex.set_light_mode(true);
        ex.set_num_threads(numThreads_);

#if NCNN_VULKAN
        if (hasGpu_) ex.set_vulkan_compute(true);
#endif

        ex.input("images", in_pad);

        ncnn::Mat out;
        ex.extract("output0", out);

        // ── Post-processing ──────────────────────────────────────────────
        // YOLOv8 NCNN output: [num_classes+4, num_anchors]
        // out.h = 4 + num_classes, out.w = num_anchors (e.g. 8400)
        int numAnchors = out.w;
        int numClasses = out.h - 4;

        if (numClasses <= 0 || numAnchors <= 0) {
            LOGW("Invalid output shape: h=%d w=%d", out.h, out.w);
            auto t1 = std::chrono::high_resolution_clock::now();
            inferTimeMs = std::chrono::duration<float, std::milli>(t1 - t0).count();
            return results;
        }

        // Pre-fetch row pointers for faster access
        const float* ptr_cx = out.row(0);
        const float* ptr_cy = out.row(1);
        const float* ptr_w  = out.row(2);
        const float* ptr_h  = out.row(3);

        std::vector<const float*> cls_ptrs(numClasses);
        for (int c = 0; c < numClasses; c++) {
            cls_ptrs[c] = out.row(4 + c);
        }

        results.reserve(256);

        for (int i = 0; i < numAnchors; i++) {
            // Find best class score
            float maxScore = -1.0f;
            int bestClass = -1;
            for (int c = 0; c < numClasses; c++) {
                float s = cls_ptrs[c][i];
                if (s > maxScore) {
                    maxScore = s;
                    bestClass = c;
                }
            }

            if (maxScore < confThresh) continue;

            // Decode box (center format → corner format)
            float cx = ptr_cx[i];
            float cy = ptr_cy[i];
            float bw = ptr_w[i];
            float bh = ptr_h[i];

            // Convert from padded coords back to original image coords
            float x1 = (cx - bw * 0.5f - padW) / scale;
            float y1 = (cy - bh * 0.5f - padH) / scale;
            float x2 = (cx + bw * 0.5f - padW) / scale;
            float y2 = (cy + bh * 0.5f - padH) / scale;

            // Clamp to image bounds
            x1 = std::max(0.0f, std::min(x1, (float)imgW));
            y1 = std::max(0.0f, std::min(y1, (float)imgH));
            x2 = std::max(0.0f, std::min(x2, (float)imgW));
            y2 = std::max(0.0f, std::min(y2, (float)imgH));

            if (x2 <= x1 || y2 <= y1) continue;

            results.push_back({x1, y1, x2, y2, maxScore, bestClass});
        }

        // Sort by score descending
        std::sort(results.begin(), results.end(),
                  [](const BoxInfo& a, const BoxInfo& b) { return a.score > b.score; });

        // NMS
        nms_sorted(results, nmsThresh);

        auto t1 = std::chrono::high_resolution_clock::now();
        inferTimeMs = std::chrono::duration<float, std::milli>(t1 - t0).count();

        return results;
    }

    bool isLoaded() const { return loaded_; }
    bool hasGpu() const { return hasGpu_; }

private:
    ncnn::Net net_;
    ncnn::UnlockedPoolAllocator blob_pool_;
    ncnn::PoolAllocator workspace_pool_;
    std::mutex mutex_;

    int targetSize_ = 640;
    int numThreads_ = 4;
    bool hasGpu_ = false;
    bool loaded_ = false;
};

// ═══════════════════════════════════════════════════════════════════════════
// Global detector instance
// ═══════════════════════════════════════════════════════════════════════════

static YoloV8Detector g_detector;

// ═══════════════════════════════════════════════════════════════════════════
// JNI Functions
// ═══════════════════════════════════════════════════════════════════════════

extern "C" {

// Initialize NCNN (call once at app start)
JNIEXPORT void JNICALL
Java_com_yolov8ncnn_YoloV8Ncnn_nativeInit(JNIEnv* env, jclass clazz) {
#if NCNN_VULKAN
    ncnn::create_gpu_instance();
    LOGI("NCNN initialized with Vulkan, GPU count: %d", ncnn::get_gpu_count());
#else
    LOGI("NCNN initialized (CPU only, no Vulkan)");
#endif
}

// Destroy NCNN (call at app exit)
JNIEXPORT void JNICALL
Java_com_yolov8ncnn_YoloV8Ncnn_nativeDestroy(JNIEnv* env, jclass clazz) {
#if NCNN_VULKAN
    ncnn::destroy_gpu_instance();
#endif
    LOGI("NCNN destroyed");
}

// Load model from assets
JNIEXPORT jboolean JNICALL
Java_com_yolov8ncnn_YoloV8Ncnn_nativeLoadModel(
        JNIEnv* env, jclass clazz,
        jobject assetManager,
        jstring paramPath, jstring binPath,
        jint targetSize, jboolean useGpu, jint numThreads) {

    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);
    const char* param = env->GetStringUTFChars(paramPath, nullptr);
    const char* bin = env->GetStringUTFChars(binPath, nullptr);

    bool ok = g_detector.loadModel(mgr, param, bin, targetSize, useGpu, numThreads);

    env->ReleaseStringUTFChars(paramPath, param);
    env->ReleaseStringUTFChars(binPath, bin);
    return (jboolean)ok;
}

// Load model from file path (for user-selected custom models)
JNIEXPORT jboolean JNICALL
Java_com_yolov8ncnn_YoloV8Ncnn_nativeLoadModelPath(
        JNIEnv* env, jclass clazz,
        jstring paramPath, jstring binPath,
        jint targetSize, jboolean useGpu, jint numThreads) {

    const char* param = env->GetStringUTFChars(paramPath, nullptr);
    const char* bin = env->GetStringUTFChars(binPath, nullptr);

    bool ok = g_detector.loadModelFromPath(param, bin, targetSize, useGpu, numThreads);

    env->ReleaseStringUTFChars(paramPath, param);
    env->ReleaseStringUTFChars(binPath, bin);
    return (jboolean)ok;
}

// Detect from Bitmap (convenient but slightly slower)
JNIEXPORT jfloatArray JNICALL
Java_com_yolov8ncnn_YoloV8Ncnn_nativeDetectBitmap(
        JNIEnv* env, jclass clazz,
        jobject bitmap,
        jfloat confThresh, jfloat nmsThresh) {

    AndroidBitmapInfo info;
    AndroidBitmap_getInfo(env, bitmap, &info);

    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
        LOGE("Bitmap format not RGBA_8888: %d", info.format);
        return nullptr;
    }

    void* pixels = nullptr;
    AndroidBitmap_lockPixels(env, bitmap, &pixels);
    if (!pixels) {
        LOGE("Failed to lock bitmap pixels");
        return nullptr;
    }

    float inferTimeMs = 0;
    auto boxes = g_detector.detect(
        (const unsigned char*)pixels,
        ncnn::Mat::PIXEL_RGBA2RGB,
        info.width, info.height,
        confThresh, nmsThresh,
        inferTimeMs);

    AndroidBitmap_unlockPixels(env, bitmap);

    // Pack results into float array: [inferTimeMs, numBoxes, x1,y1,x2,y2,score,label, ...]
    int numBoxes = (int)boxes.size();
    int arraySize = 2 + numBoxes * 6;
    jfloatArray result = env->NewFloatArray(arraySize);
    std::vector<float> data(arraySize);
    data[0] = inferTimeMs;
    data[1] = (float)numBoxes;
    for (int i = 0; i < numBoxes; i++) {
        int offset = 2 + i * 6;
        data[offset + 0] = boxes[i].x1;
        data[offset + 1] = boxes[i].y1;
        data[offset + 2] = boxes[i].x2;
        data[offset + 3] = boxes[i].y2;
        data[offset + 4] = boxes[i].score;
        data[offset + 5] = (float)boxes[i].label;
    }
    env->SetFloatArrayRegion(result, 0, arraySize, data.data());
    return result;
}

// Detect from raw pixel buffer (fastest path - avoids Bitmap overhead)
// pixelBuffer: RGBA pixel data from ImageReader
JNIEXPORT jfloatArray JNICALL
Java_com_yolov8ncnn_YoloV8Ncnn_nativeDetectBuffer(
        JNIEnv* env, jclass clazz,
        jobject pixelBuffer, jint width, jint height, jint rowStride,
        jfloat confThresh, jfloat nmsThresh) {

    uint8_t* bufPtr = (uint8_t*)env->GetDirectBufferAddress(pixelBuffer);
    if (!bufPtr) {
        LOGE("Failed to get direct buffer address");
        return nullptr;
    }

    // Handle row stride padding: if rowStride != width*4, need to copy rows
    int pixelStride = 4; // RGBA
    int expectedStride = width * pixelStride;

    std::vector<uint8_t> compactBuf;
    const unsigned char* pixels = bufPtr;

    if (rowStride != expectedStride) {
        // Need to compact the buffer (remove row padding)
        compactBuf.resize(width * height * pixelStride);
        for (int y = 0; y < height; y++) {
            memcpy(compactBuf.data() + y * expectedStride,
                   bufPtr + y * rowStride,
                   expectedStride);
        }
        pixels = compactBuf.data();
    }

    float inferTimeMs = 0;
    auto boxes = g_detector.detect(
        pixels,
        ncnn::Mat::PIXEL_RGBA2RGB,
        width, height,
        confThresh, nmsThresh,
        inferTimeMs);

    // Pack results
    int numBoxes = (int)boxes.size();
    int arraySize = 2 + numBoxes * 6;
    jfloatArray result = env->NewFloatArray(arraySize);
    std::vector<float> data(arraySize);
    data[0] = inferTimeMs;
    data[1] = (float)numBoxes;
    for (int i = 0; i < numBoxes; i++) {
        int offset = 2 + i * 6;
        data[offset + 0] = boxes[i].x1;
        data[offset + 1] = boxes[i].y1;
        data[offset + 2] = boxes[i].x2;
        data[offset + 3] = boxes[i].y2;
        data[offset + 4] = boxes[i].score;
        data[offset + 5] = (float)boxes[i].label;
    }
    env->SetFloatArrayRegion(result, 0, arraySize, data.data());
    return result;
}

// Check if model is loaded
JNIEXPORT jboolean JNICALL
Java_com_yolov8ncnn_YoloV8Ncnn_nativeIsLoaded(JNIEnv* env, jclass clazz) {
    return (jboolean)g_detector.isLoaded();
}

// Check GPU availability
JNIEXPORT jboolean JNICALL
Java_com_yolov8ncnn_YoloV8Ncnn_nativeHasGpu(JNIEnv* env, jclass clazz) {
    return (jboolean)g_detector.hasGpu();
}

} // extern "C"
