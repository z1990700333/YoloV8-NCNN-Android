// yolov8ncnn_jni.cpp
// High-performance YOLOv8 NCNN inference engine with JNI bridge

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
#include <fstream>
#include <sstream>

#include "net.h"
#include "cpu.h"

#if NCNN_VULKAN
#include "gpu.h"
#endif

#define TAG "YoloV8Ncnn"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

struct BoxInfo {
    float x1, y1, x2, y2;
    float score;
    int label;
};

// ═══ NMS ═══════════════════════════════════════════════════════════════════

static inline float intersection_area(const BoxInfo& a, const BoxInfo& b) {
    float iw = std::max(0.0f, std::min(a.x2, b.x2) - std::max(a.x1, b.x1));
    float ih = std::max(0.0f, std::min(a.y2, b.y2) - std::max(a.y1, b.y1));
    return iw * ih;
}

static void nms_sorted(std::vector<BoxInfo>& boxes, float nms_threshold) {
    std::vector<BoxInfo> result;
    std::vector<bool> suppressed(boxes.size(), false);
    for (size_t i = 0; i < boxes.size(); i++) {
        if (suppressed[i]) continue;
        result.push_back(boxes[i]);
        float area_i = (boxes[i].x2 - boxes[i].x1) * (boxes[i].y2 - boxes[i].y1);
        for (size_t j = i + 1; j < boxes.size(); j++) {
            if (suppressed[j] || boxes[i].label != boxes[j].label) continue;
            float inter = intersection_area(boxes[i], boxes[j]);
            float area_j = (boxes[j].x2 - boxes[j].x1) * (boxes[j].y2 - boxes[j].y1);
            if (inter / (area_i + area_j - inter + 1e-5f) > nms_threshold)
                suppressed[j] = true;
        }
    }
    boxes = std::move(result);
}

// ═══ Sigmoid ══════════════════════════════════════════════════════════════

static inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// ═══ Auto-detect blob names from .param file ══════════════════════════════

static void detect_blob_names(const char* paramPath,
                               std::string& inputName, std::string& outputName) {
    // Default names
    inputName = "images";
    outputName = "output0";

    std::ifstream file(paramPath);
    if (!file.is_open()) {
        LOGW("Cannot open param file for blob detection: %s", paramPath);
        return;
    }

    std::string line;
    std::string firstBlobName;
    std::string lastBlobName;
    int lineNum = 0;

    // Parse .param format:
    // Line 0: magic number (7767517)
    // Line 1: layer_count blob_count
    // Line 2+: layer_type layer_name input_count output_count input_blobs... output_blobs... params...

    while (std::getline(file, line)) {
        lineNum++;
        if (lineNum <= 2) continue; // Skip header

        std::istringstream iss(line);
        std::string layerType, layerName;
        int inputCount, outputCount;

        if (!(iss >> layerType >> layerName >> inputCount >> outputCount)) continue;

        // Read input blob names
        std::vector<std::string> inputs(inputCount);
        for (int i = 0; i < inputCount; i++) iss >> inputs[i];

        // Read output blob names
        std::vector<std::string> outputs(outputCount);
        for (int i = 0; i < outputCount; i++) iss >> outputs[i];

        // First Input layer's output blob = model input
        if (layerType == "Input" && outputCount > 0) {
            inputName = outputs[0];
        }

        // Track last layer's output blob = model output
        if (outputCount > 0) {
            lastBlobName = outputs[outputCount - 1];
        }
    }

    if (!lastBlobName.empty()) {
        outputName = lastBlobName;
    }

    LOGI("Auto-detected blobs: input='%s', output='%s'", inputName.c_str(), outputName.c_str());
}

// ═══ YoloV8 Detector ═════════════════════════════════════════════════════

class YoloV8Detector {
public:
    YoloV8Detector() = default;
    ~YoloV8Detector() { net_.clear(); }

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
        net_.opt.use_fp16_packed = true;
        net_.opt.use_fp16_storage = true;

#if NCNN_VULKAN
        hasGpu_ = useGpu && (ncnn::get_gpu_count() > 0);
        net_.opt.use_vulkan_compute = hasGpu_;
        if (hasGpu_) {
            net_.opt.use_fp16_arithmetic = true;
            LOGI("Vulkan GPU enabled");
        }
#else
        hasGpu_ = false;
#endif

        // Auto-detect blob names BEFORE loading
        detect_blob_names(paramPath, inputBlobName_, outputBlobName_);

        int ret = net_.load_param(paramPath);
        if (ret != 0) { LOGE("load_param failed: %s ret=%d", paramPath, ret); return false; }
        ret = net_.load_model(binPath);
        if (ret != 0) { LOGE("load_model failed: %s ret=%d", binPath, ret); return false; }

        loaded_ = true;
        LOGI("Model loaded: %s (target=%d, gpu=%d, threads=%d, in='%s', out='%s')",
             paramPath, targetSize, (int)hasGpu_, numThreads,
             inputBlobName_.c_str(), outputBlobName_.c_str());
        return true;
    }

    bool loadModel(AAssetManager* mgr, const char* paramPath, const char* binPath,
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
        net_.opt.use_fp16_packed = true;
        net_.opt.use_fp16_storage = true;

#if NCNN_VULKAN
        hasGpu_ = useGpu && (ncnn::get_gpu_count() > 0);
        net_.opt.use_vulkan_compute = hasGpu_;
        if (hasGpu_) net_.opt.use_fp16_arithmetic = true;
#else
        hasGpu_ = false;
#endif

        // Use default blob names for assets
        inputBlobName_ = "images";
        outputBlobName_ = "output0";

        int ret = net_.load_param(mgr, paramPath);
        if (ret != 0) { LOGE("load_param failed: %d", ret); return false; }
        ret = net_.load_model(mgr, binPath);
        if (ret != 0) { LOGE("load_model failed: %d", ret); return false; }

        loaded_ = true;
        return true;
    }

    std::vector<BoxInfo> detect(const unsigned char* pixels, int pixelType,
                                int imgW, int imgH,
                                float confThresh, float nmsThresh,
                                float& inferTimeMs) {
        std::vector<BoxInfo> results;
        if (!loaded_) return results;

        auto t0 = std::chrono::high_resolution_clock::now();

        // ── Letterbox resize ─────────────────────────────────────────────
        float scale = std::min((float)targetSize_ / imgW, (float)targetSize_ / imgH);
        int scaledW = (int)(imgW * scale);
        int scaledH = (int)(imgH * scale);
        int padW = (targetSize_ - scaledW) / 2;
        int padH = (targetSize_ - scaledH) / 2;

        ncnn::Mat in = ncnn::Mat::from_pixels_resize(
            pixels, pixelType, imgW, imgH, scaledW, scaledH);

        ncnn::Mat in_pad;
        ncnn::copy_make_border(in, in_pad,
            padH, targetSize_ - scaledH - padH,
            padW, targetSize_ - scaledW - padW,
            ncnn::BORDER_CONSTANT, 114.0f);

        const float norm_vals[3] = {1.0f/255.0f, 1.0f/255.0f, 1.0f/255.0f};
        in_pad.substract_mean_normalize(nullptr, norm_vals);

        // ── Inference ────────────────────────────────────────────────────
        ncnn::Extractor ex = net_.create_extractor();
        ex.set_light_mode(true);
        ex.set_num_threads(numThreads_);
#if NCNN_VULKAN
        if (hasGpu_) ex.set_vulkan_compute(true);
#endif

        ex.input(inputBlobName_.c_str(), in_pad);

        ncnn::Mat out;
        ex.extract(outputBlobName_.c_str(), out);

        // ── Post-processing (auto-detect format) ─────────────────────────
        // Format A (ultralytics): out.dims=2, out.h = 4+num_classes, out.w = num_anchors
        //   → row(0..3) = cx,cy,w,h; row(4+c) = class score (already sigmoid)
        // Format B (transposed): out.dims=3, out.c=1, out.h = num_anchors, out.w = 4+num_classes
        //   → each row: cx,cy,w,h, cls0, cls1, ...
        // Format C (with objectness, YOLOv5-style): out.w = 5+num_classes
        //   → each row: cx,cy,w,h, obj_conf, cls0, cls1, ...

        LOGI("Output shape: dims=%d c=%d h=%d w=%d", out.dims, out.c, out.h, out.w);

        if (out.dims == 2 && out.h > out.w && out.h > 5) {
            // Format A: [4+nc, anchors] — standard ultralytics NCNN export
            parseFormatA(out, results, confThresh, padW, padH, scale, imgW, imgH);
        } else if (out.dims == 2 && out.w > out.h) {
            // Format B transposed in 2D: [anchors, 4+nc]
            parseFormatB(out, results, confThresh, padW, padH, scale, imgW, imgH);
        } else if (out.dims == 3) {
            // Format B/C in 3D: c=1, h=anchors, w=4+nc or 5+nc
            ncnn::Mat out2d = out.reshape(out.w, out.h);
            parseFormatB(out2d, results, confThresh, padW, padH, scale, imgW, imgH);
        } else {
            LOGW("Unknown output format: dims=%d c=%d h=%d w=%d", out.dims, out.c, out.h, out.w);
        }

        // Sort + NMS
        std::sort(results.begin(), results.end(),
                  [](const BoxInfo& a, const BoxInfo& b){ return a.score > b.score; });
        nms_sorted(results, nmsThresh);

        auto t1 = std::chrono::high_resolution_clock::now();
        inferTimeMs = std::chrono::duration<float, std::milli>(t1 - t0).count();

        LOGI("Detect: %d boxes, %.1fms", (int)results.size(), inferTimeMs);
        return results;
    }

    bool isLoaded() const { return loaded_; }
    bool hasGpu() const { return hasGpu_; }

private:
    // Format A: [4+nc, anchors] — rows are features, columns are anchors
    void parseFormatA(const ncnn::Mat& out, std::vector<BoxInfo>& results,
                      float confThresh, int padW, int padH, float scale, int imgW, int imgH) {
        int numAnchors = out.w;
        int numClasses = out.h - 4;
        if (numClasses <= 0) return;

        const float* ptr_cx = out.row(0);
        const float* ptr_cy = out.row(1);
        const float* ptr_w  = out.row(2);
        const float* ptr_h  = out.row(3);

        std::vector<const float*> cls_ptrs(numClasses);
        for (int c = 0; c < numClasses; c++) cls_ptrs[c] = out.row(4 + c);

        for (int i = 0; i < numAnchors; i++) {
            float maxScore = -1.0f;
            int bestClass = -1;
            for (int c = 0; c < numClasses; c++) {
                float s = cls_ptrs[c][i];
                if (s > maxScore) { maxScore = s; bestClass = c; }
            }
            if (maxScore < confThresh) continue;

            float cx = ptr_cx[i], cy = ptr_cy[i], bw = ptr_w[i], bh = ptr_h[i];
            float x1 = (cx - bw*0.5f - padW) / scale;
            float y1 = (cy - bh*0.5f - padH) / scale;
            float x2 = (cx + bw*0.5f - padW) / scale;
            float y2 = (cy + bh*0.5f - padH) / scale;
            x1 = std::max(0.f, std::min(x1, (float)imgW));
            y1 = std::max(0.f, std::min(y1, (float)imgH));
            x2 = std::max(0.f, std::min(x2, (float)imgW));
            y2 = std::max(0.f, std::min(y2, (float)imgH));
            if (x2 > x1 && y2 > y1)
                results.push_back({x1, y1, x2, y2, maxScore, bestClass});
        }
    }

    // Format B: [anchors, 4+nc] or [anchors, 5+nc] — rows are anchors
    void parseFormatB(const ncnn::Mat& out, std::vector<BoxInfo>& results,
                      float confThresh, int padW, int padH, float scale, int imgW, int imgH) {
        int numAnchors = out.h;
        int cols = out.w;

        // Detect if objectness column exists (YOLOv5-style: 5+nc vs YOLOv8: 4+nc)
        bool hasObj = false;
        int numClasses;

        // Heuristic: if cols-4 is a common class count (1,2,3,...,80,etc), it's YOLOv8 format
        // If cols-5 is, it's YOLOv5 format with objectness
        int nc4 = cols - 4;
        int nc5 = cols - 5;

        // Check first few rows to see if column 4 looks like objectness (0-1 range after sigmoid)
        if (nc5 > 0 && numAnchors > 0) {
            float sum4 = 0;
            int check = std::min(numAnchors, 100);
            for (int i = 0; i < check; i++) {
                float v = out.row(i)[4];
                // If raw values are large (not 0-1), they need sigmoid → likely objectness
                sum4 += fabsf(v);
            }
            float avg4 = sum4 / check;
            // YOLOv5 objectness is usually present; YOLOv8 class scores start at col 4
            // If average absolute value at col4 is very different from cols 5+, it's objectness
            hasObj = (nc5 >= 1 && nc5 <= 1000);
            // Simple heuristic: if nc4 is reasonable, use YOLOv8 format
            if (nc4 >= 1 && nc4 <= 1000) hasObj = false;
        }

        if (hasObj) {
            numClasses = nc5;
            LOGI("Format B with objectness: anchors=%d, classes=%d", numAnchors, numClasses);
        } else {
            numClasses = nc4;
            LOGI("Format B (YOLOv8): anchors=%d, classes=%d", numAnchors, numClasses);
        }

        if (numClasses <= 0) return;

        for (int i = 0; i < numAnchors; i++) {
            const float* row = out.row(i);
            float cx = row[0], cy = row[1], bw = row[2], bh = row[3];

            float objConf = 1.0f;
            int clsOffset = 4;
            if (hasObj) {
                objConf = sigmoid(row[4]);
                clsOffset = 5;
                if (objConf < confThresh) continue;
            }

            float maxScore = -1e9f;
            int bestClass = -1;
            for (int c = 0; c < numClasses; c++) {
                float s = row[clsOffset + c];
                if (s > maxScore) { maxScore = s; bestClass = c; }
            }

            // Apply sigmoid if scores look like raw logits (outside 0-1)
            if (maxScore > 1.0f || maxScore < 0.0f) {
                maxScore = sigmoid(maxScore);
            }
            float finalScore = maxScore * objConf;
            if (finalScore < confThresh) continue;

            float x1 = (cx - bw*0.5f - padW) / scale;
            float y1 = (cy - bh*0.5f - padH) / scale;
            float x2 = (cx + bw*0.5f - padW) / scale;
            float y2 = (cy + bh*0.5f - padH) / scale;
            x1 = std::max(0.f, std::min(x1, (float)imgW));
            y1 = std::max(0.f, std::min(y1, (float)imgH));
            x2 = std::max(0.f, std::min(x2, (float)imgW));
            y2 = std::max(0.f, std::min(y2, (float)imgH));
            if (x2 > x1 && y2 > y1)
                results.push_back({x1, y1, x2, y2, finalScore, bestClass});
        }
    }

    ncnn::Net net_;
    ncnn::UnlockedPoolAllocator blob_pool_;
    ncnn::PoolAllocator workspace_pool_;
    std::mutex mutex_;

    std::string inputBlobName_ = "images";
    std::string outputBlobName_ = "output0";
    int targetSize_ = 640;
    int numThreads_ = 4;
    bool hasGpu_ = false;
    bool loaded_ = false;
};

// ═══ Global ═══════════════════════════════════════════════════════════════

static YoloV8Detector g_detector;

// ═══ JNI ══════════════════════════════════════════════════════════════════

extern "C" {

JNIEXPORT void JNICALL
Java_com_yolov8ncnn_YoloV8Ncnn_nativeInit(JNIEnv* env, jclass clazz) {
#if NCNN_VULKAN
    ncnn::create_gpu_instance();
    LOGI("NCNN init, Vulkan GPU count: %d", ncnn::get_gpu_count());
#else
    LOGI("NCNN init (CPU only)");
#endif
}

JNIEXPORT void JNICALL
Java_com_yolov8ncnn_YoloV8Ncnn_nativeDestroy(JNIEnv* env, jclass clazz) {
#if NCNN_VULKAN
    ncnn::destroy_gpu_instance();
#endif
}

JNIEXPORT jboolean JNICALL
Java_com_yolov8ncnn_YoloV8Ncnn_nativeLoadModel(
        JNIEnv* env, jclass clazz, jobject assetManager,
        jstring paramPath, jstring binPath,
        jint targetSize, jboolean useGpu, jint numThreads) {
    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);
    const char* p = env->GetStringUTFChars(paramPath, nullptr);
    const char* b = env->GetStringUTFChars(binPath, nullptr);
    bool ok = g_detector.loadModel(mgr, p, b, targetSize, useGpu, numThreads);
    env->ReleaseStringUTFChars(paramPath, p);
    env->ReleaseStringUTFChars(binPath, b);
    return (jboolean)ok;
}

JNIEXPORT jboolean JNICALL
Java_com_yolov8ncnn_YoloV8Ncnn_nativeLoadModelPath(
        JNIEnv* env, jclass clazz,
        jstring paramPath, jstring binPath,
        jint targetSize, jboolean useGpu, jint numThreads) {
    const char* p = env->GetStringUTFChars(paramPath, nullptr);
    const char* b = env->GetStringUTFChars(binPath, nullptr);
    bool ok = g_detector.loadModelFromPath(p, b, targetSize, useGpu, numThreads);
    env->ReleaseStringUTFChars(paramPath, p);
    env->ReleaseStringUTFChars(binPath, b);
    return (jboolean)ok;
}

JNIEXPORT jfloatArray JNICALL
Java_com_yolov8ncnn_YoloV8Ncnn_nativeDetectBitmap(
        JNIEnv* env, jclass clazz, jobject bitmap,
        jfloat confThresh, jfloat nmsThresh) {
    AndroidBitmapInfo info;
    AndroidBitmap_getInfo(env, bitmap, &info);
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
        LOGE("Bitmap not RGBA_8888: %d", info.format);
        return nullptr;
    }
    void* pixels = nullptr;
    AndroidBitmap_lockPixels(env, bitmap, &pixels);
    if (!pixels) { LOGE("lockPixels failed"); return nullptr; }

    float inferTimeMs = 0;
    auto boxes = g_detector.detect((const unsigned char*)pixels,
        ncnn::Mat::PIXEL_RGBA2RGB, info.width, info.height,
        confThresh, nmsThresh, inferTimeMs);
    AndroidBitmap_unlockPixels(env, bitmap);

    int n = (int)boxes.size();
    int sz = 2 + n * 6;
    std::vector<float> data(sz);
    data[0] = inferTimeMs; data[1] = (float)n;
    for (int i = 0; i < n; i++) {
        int o = 2 + i*6;
        data[o]=boxes[i].x1; data[o+1]=boxes[i].y1;
        data[o+2]=boxes[i].x2; data[o+3]=boxes[i].y2;
        data[o+4]=boxes[i].score; data[o+5]=(float)boxes[i].label;
    }
    jfloatArray result = env->NewFloatArray(sz);
    env->SetFloatArrayRegion(result, 0, sz, data.data());
    return result;
}

JNIEXPORT jfloatArray JNICALL
Java_com_yolov8ncnn_YoloV8Ncnn_nativeDetectBuffer(
        JNIEnv* env, jclass clazz,
        jobject pixelBuffer, jint width, jint height, jint rowStride,
        jfloat confThresh, jfloat nmsThresh) {
    uint8_t* bufPtr = (uint8_t*)env->GetDirectBufferAddress(pixelBuffer);
    if (!bufPtr) { LOGE("GetDirectBufferAddress failed"); return nullptr; }

    int expected = width * 4;
    std::vector<uint8_t> compact;
    const unsigned char* pixels = bufPtr;
    if (rowStride != expected) {
        compact.resize(width * height * 4);
        for (int y = 0; y < height; y++)
            memcpy(compact.data() + y*expected, bufPtr + y*rowStride, expected);
        pixels = compact.data();
    }

    float inferTimeMs = 0;
    auto boxes = g_detector.detect(pixels, ncnn::Mat::PIXEL_RGBA2RGB,
        width, height, confThresh, nmsThresh, inferTimeMs);

    int n = (int)boxes.size();
    int sz = 2 + n*6;
    std::vector<float> data(sz);
    data[0] = inferTimeMs; data[1] = (float)n;
    for (int i = 0; i < n; i++) {
        int o = 2+i*6;
        data[o]=boxes[i].x1; data[o+1]=boxes[i].y1;
        data[o+2]=boxes[i].x2; data[o+3]=boxes[i].y2;
        data[o+4]=boxes[i].score; data[o+5]=(float)boxes[i].label;
    }
    jfloatArray result = env->NewFloatArray(sz);
    env->SetFloatArrayRegion(result, 0, sz, data.data());
    return result;
}

JNIEXPORT jboolean JNICALL
Java_com_yolov8ncnn_YoloV8Ncnn_nativeIsLoaded(JNIEnv* env, jclass clazz) {
    return (jboolean)g_detector.isLoaded();
}

JNIEXPORT jboolean JNICALL
Java_com_yolov8ncnn_YoloV8Ncnn_nativeHasGpu(JNIEnv* env, jclass clazz) {
    return (jboolean)g_detector.hasGpu();
}

} // extern "C"
