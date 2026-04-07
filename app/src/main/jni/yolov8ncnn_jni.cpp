// yolov8ncnn_jni.cpp - YOLOv8 DFL detection head + NCNN JNI bridge
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

struct BoxInfo { float x1, y1, x2, y2, score; int label; };

static inline float intersection_area(const BoxInfo& a, const BoxInfo& b) {
    float iw = std::max(0.f, std::min(a.x2, b.x2) - std::max(a.x1, b.x1));
    float ih = std::max(0.f, std::min(a.y2, b.y2) - std::max(a.y1, b.y1));
    return iw * ih;
}

static void nms_sorted(std::vector<BoxInfo>& boxes, float nms_thresh) {
    std::vector<BoxInfo> res;
    std::vector<bool> sup(boxes.size(), false);
    for (size_t i = 0; i < boxes.size(); i++) {
        if (sup[i]) continue;
        res.push_back(boxes[i]);
        float ai = (boxes[i].x2 - boxes[i].x1) * (boxes[i].y2 - boxes[i].y1);
        for (size_t j = i + 1; j < boxes.size(); j++) {
            if (sup[j] || boxes[i].label != boxes[j].label) continue;
            float inter = intersection_area(boxes[i], boxes[j]);
            float aj = (boxes[j].x2 - boxes[j].x1) * (boxes[j].y2 - boxes[j].y1);
            if (inter / (ai + aj - inter + 1e-5f) > nms_thresh) sup[j] = true;
        }
    }
    boxes = std::move(res);
}

static inline float sigmoid(float x) { return 1.f / (1.f + expf(-x)); }

// DFL: softmax over reg_max values then weighted sum -> distance
static inline float dfl_decode(const float* src, int reg_max) {
    float maxv = src[0];
    for (int i = 1; i < reg_max; i++) if (src[i] > maxv) maxv = src[i];
    float sum = 0.f, buf[16];
    for (int i = 0; i < reg_max; i++) { buf[i] = expf(src[i] - maxv); sum += buf[i]; }
    float val = 0.f;
    for (int i = 0; i < reg_max; i++) val += (buf[i] / sum) * i;
    return val;
}

// Auto-detect blob names from .param file
static void detect_blob_names(const char* path, std::string& inN, std::string& outN) {
    inN = "images"; outN = "output0";
    std::ifstream f(path); if (!f.is_open()) return;
    std::string line; int ln = 0; std::string last;
    while (std::getline(f, line)) {
        ln++; if (ln <= 2) continue;
        std::istringstream iss(line);
        std::string lt, nm; int ic, oc;
        if (!(iss >> lt >> nm >> ic >> oc)) continue;
        std::vector<std::string> ins(ic); for (int i=0;i<ic;i++) iss>>ins[i];
        std::vector<std::string> outs(oc); for (int i=0;i<oc;i++) iss>>outs[i];
        if (lt == "Input" && oc > 0) inN = outs[0];
        if (oc > 0) last = outs[oc-1];
    }
    if (!last.empty()) outN = last;
    LOGI("Blobs: in='%s' out='%s'", inN.c_str(), outN.c_str());
}

struct Anchor { float cx, cy; int stride; };

class YoloV8Detector {
public:
    YoloV8Detector() = default;
    ~YoloV8Detector() { net_.clear(); }

    bool loadModelFromPath(const char* pp, const char* bp, int ts, bool gpu, int nt) {
        std::lock_guard<std::mutex> lock(mtx_);
        net_.clear(); bpool_.clear(); wpool_.clear();
        tsize_ = ts; nthr_ = nt;
        ncnn::set_cpu_powersave(2); ncnn::set_omp_num_threads(nt);
        setupOpt(gpu);
        detect_blob_names(pp, inB_, outB_);
        if (net_.load_param(pp) != 0) { LOGE("load_param fail"); return false; }
        if (net_.load_model(bp) != 0) { LOGE("load_model fail"); return false; }
        loaded_ = true;
        LOGI("Model loaded: target=%d gpu=%d thr=%d", ts, (int)gpu_, nt);
        return true;
    }

    bool loadModel(AAssetManager* mgr, const char* pp, const char* bp, int ts, bool gpu, int nt) {
        std::lock_guard<std::mutex> lock(mtx_);
        net_.clear(); bpool_.clear(); wpool_.clear();
        tsize_ = ts; nthr_ = nt;
        ncnn::set_cpu_powersave(2); ncnn::set_omp_num_threads(nt);
        setupOpt(gpu);
        inB_ = "images"; outB_ = "output0";
        if (net_.load_param(mgr, pp) != 0) return false;
        if (net_.load_model(mgr, bp) != 0) return false;
        loaded_ = true; return true;
    }

    std::vector<BoxInfo> detect(const unsigned char* px, int pxT, int iW, int iH,
                                float cTh, float nTh, float& ms) {
        std::vector<BoxInfo> res;
        if (!loaded_) return res;
        auto t0 = std::chrono::high_resolution_clock::now();

        // Letterbox
        float sc = std::min((float)tsize_/iW, (float)tsize_/iH);
        int sw = (int)(iW*sc), sh = (int)(iH*sc);
        int pW = (tsize_-sw)/2, pH = (tsize_-sh)/2;
        ncnn::Mat in = ncnn::Mat::from_pixels_resize(px, pxT, iW, iH, sw, sh);
        ncnn::Mat inp;
        ncnn::copy_make_border(in, inp, pH, tsize_-sh-pH, pW, tsize_-sw-pW,
                               ncnn::BORDER_CONSTANT, 114.f);
        const float nv[3] = {1.f/255.f, 1.f/255.f, 1.f/255.f};
        inp.substract_mean_normalize(nullptr, nv);

        ncnn::Extractor ex = net_.create_extractor();
        ex.set_light_mode(true); ex.set_num_threads(nthr_);
#if NCNN_VULKAN
        if (gpu_) ex.set_vulkan_compute(true);
#endif
        ex.input(inB_.c_str(), inp);
        ncnn::Mat out; ex.extract(outB_.c_str(), out);

        LOGI("Out: dims=%d c=%d h=%d w=%d", out.dims, out.c, out.h, out.w);

        // Build anchor grid: strides 8,16,32
        std::vector<Anchor> anc;
        int strides[3] = {8, 16, 32};
        for (int s = 0; s < 3; s++) {
            int g = tsize_ / strides[s];
            for (int y = 0; y < g; y++)
                for (int x = 0; x < g; x++)
                    anc.push_back({x+0.5f, y+0.5f, strides[s]});
        }
        int totAnc = (int)anc.size();

        // Parse output shape
        int fd = 0, na = 0; bool tr = false;
        if (out.dims == 2) {
            if (out.h < out.w) { fd=out.h; na=out.w; tr=false; }
            else { fd=out.w; na=out.h; tr=true; }
        } else if (out.dims == 3) {
            fd=out.w; na=out.h; tr=true;
        } else { LOGW("dims=%d unsupported", out.dims); goto end; }

        LOGI("fd=%d na=%d tr=%d exp=%d", fd, na, (int)tr, totAnc);

        // Fix anchor count if mismatch
        if (na != totAnc) {
            anc.clear();
            int g0=tsize_/8, g1=tsize_/16, g2=tsize_/32;
            if (g0*g0+g1*g1+g2*g2 == na) {
                for(int y=0;y<g0;y++) for(int x=0;x<g0;x++) anc.push_back({x+.5f,y+.5f,8});
                for(int y=0;y<g1;y++) for(int x=0;x<g1;x++) anc.push_back({x+.5f,y+.5f,16});
                for(int y=0;y<g2;y++) for(int x=0;x<g2;x++) anc.push_back({x+.5f,y+.5f,32});
            } else { LOGW("anchor mismatch na=%d", na); goto end; }
        }

        {
            // Check DFL: featDim = 4*16 + nc
            int rm = 16, nc = fd - 4*rm;
            if (nc >= 1 && nc <= 1000) {
                // DFL format
                LOGI("DFL: rm=%d nc=%d", rm, nc);
                for (int i = 0; i < na && i < (int)anc.size(); i++) {
                    float ft[256];
                    if (!tr) { for(int f=0;f<fd;f++) ft[f]=out.row(f)[i]; }
                    else { const float*r=out.row(i); for(int f=0;f<fd;f++) ft[f]=r[f]; }

                    // Class scores (need sigmoid - raw from final Conv)
                    float mx = -1e9f; int bc = 0;
                    for (int c=0; c<nc; c++) {
                        float s = sigmoid(ft[4*rm+c]);
                        if (s > mx) { mx=s; bc=c; }
                    }
                    if (mx < cTh) continue;

                    // DFL bbox decode
                    float dl = dfl_decode(ft+0*rm, rm);
                    float dt = dfl_decode(ft+1*rm, rm);
                    float dr = dfl_decode(ft+2*rm, rm);
                    float db = dfl_decode(ft+3*rm, rm);

                    int st = anc[i].stride;
                    float x1 = (anc[i].cx - dl) * st;
                    float y1 = (anc[i].cy - dt) * st;
                    float x2 = (anc[i].cx + dr) * st;
                    float y2 = (anc[i].cy + db) * st;

                    x1=(x1-pW)/sc; y1=(y1-pH)/sc; x2=(x2-pW)/sc; y2=(y2-pH)/sc;
                    x1=std::max(0.f,std::min(x1,(float)iW));
                    y1=std::max(0.f,std::min(y1,(float)iH));
                    x2=std::max(0.f,std::min(x2,(float)iW));
                    y2=std::max(0.f,std::min(y2,(float)iH));
                    if (x2>x1 && y2>y1) res.push_back({x1,y1,x2,y2,mx,bc});
                }
            } else {
                // Simple: 4+nc or 5+nc
                nc = fd - 4; bool ho = false;
                if (nc<1||nc>1000) { nc=fd-5; ho=true; }
                if (nc<1||nc>1000) { LOGW("unknown fd=%d",fd); goto end; }
                LOGI("Simple: nc=%d obj=%d", nc, (int)ho);
                int co = ho ? 5 : 4;
                for (int i=0; i<na && i<(int)anc.size(); i++) {
                    float ft[2048];
                    if(!tr){for(int f=0;f<fd;f++)ft[f]=out.row(f)[i];}
                    else{const float*r=out.row(i);for(int f=0;f<fd;f++)ft[f]=r[f];}
                    float oc=1.f;
                    if(ho){oc=sigmoid(ft[4]);if(oc<cTh)continue;}
                    float mx=-1e9f;int bc=0;
                    for(int c=0;c<nc;c++){float s=ft[co+c];if(s>1.f||s<0.f)s=sigmoid(s);if(s>mx){mx=s;bc=c;}}
                    float fs=mx*oc; if(fs<cTh)continue;
                    float cx=ft[0],cy=ft[1],bw=ft[2],bh=ft[3];
                    float x1=(cx-bw*.5f-pW)/sc,y1=(cy-bh*.5f-pH)/sc;
                    float x2=(cx+bw*.5f-pW)/sc,y2=(cy+bh*.5f-pH)/sc;
                    x1=std::max(0.f,std::min(x1,(float)iW));
                    y1=std::max(0.f,std::min(y1,(float)iH));
                    x2=std::max(0.f,std::min(x2,(float)iW));
                    y2=std::max(0.f,std::min(y2,(float)iH));
                    if(x2>x1&&y2>y1) res.push_back({x1,y1,x2,y2,fs,bc});
                }
            }
        }

        end:
        std::sort(res.begin(), res.end(), [](const BoxInfo&a,const BoxInfo&b){return a.score>b.score;});
        nms_sorted(res, nTh);
        auto t1 = std::chrono::high_resolution_clock::now();
        ms = std::chrono::duration<float, std::milli>(t1-t0).count();
        LOGI("Detect: %d boxes %.1fms", (int)res.size(), ms);
        return res;
    }

    bool isLoaded() const { return loaded_; }
    bool hasGpu() const { return gpu_; }

private:
    void setupOpt(bool gpu) {
        net_.opt = ncnn::Option();
        net_.opt.lightmode = true;
        net_.opt.num_threads = nthr_;
        net_.opt.blob_allocator = &bpool_;
        net_.opt.workspace_allocator = &wpool_;
        net_.opt.use_packing_layout = true;
        net_.opt.use_fp16_packed = true;
        net_.opt.use_fp16_storage = true;
#if NCNN_VULKAN
        gpu_ = gpu && ncnn::get_gpu_count() > 0;
        net_.opt.use_vulkan_compute = gpu_;
        if (gpu_) { net_.opt.use_fp16_arithmetic = true; LOGI("Vulkan on"); }
#else
        gpu_ = false;
#endif
    }
    ncnn::Net net_;
    ncnn::UnlockedPoolAllocator bpool_;
    ncnn::PoolAllocator wpool_;
    std::mutex mtx_;
    std::string inB_="images", outB_="output0";
    int tsize_=640, nthr_=4;
    bool gpu_=false, loaded_=false;
};

static YoloV8Detector g_det;

extern "C" {

JNIEXPORT void JNICALL
Java_com_yolov8ncnn_YoloV8Ncnn_nativeInit(JNIEnv*, jclass) {
#if NCNN_VULKAN
    ncnn::create_gpu_instance();
    LOGI("NCNN init, GPUs: %d", ncnn::get_gpu_count());
#else
    LOGI("NCNN init CPU");
#endif
}

JNIEXPORT void JNICALL
Java_com_yolov8ncnn_YoloV8Ncnn_nativeDestroy(JNIEnv*, jclass) {
#if NCNN_VULKAN
    ncnn::destroy_gpu_instance();
#endif
}

JNIEXPORT jboolean JNICALL
Java_com_yolov8ncnn_YoloV8Ncnn_nativeLoadModel(
    JNIEnv* env, jclass, jobject am, jstring jp, jstring jb,
    jint ts, jboolean gpu, jint nt) {
    AAssetManager* mgr = AAssetManager_fromJava(env, am);
    const char* p=env->GetStringUTFChars(jp,0);
    const char* b=env->GetStringUTFChars(jb,0);
    bool ok = g_det.loadModel(mgr,p,b,ts,gpu,nt);
    env->ReleaseStringUTFChars(jp,p);
    env->ReleaseStringUTFChars(jb,b);
    return (jboolean)ok;
}

JNIEXPORT jboolean JNICALL
Java_com_yolov8ncnn_YoloV8Ncnn_nativeLoadModelPath(
    JNIEnv* env, jclass, jstring jp, jstring jb,
    jint ts, jboolean gpu, jint nt) {
    const char* p=env->GetStringUTFChars(jp,0);
    const char* b=env->GetStringUTFChars(jb,0);
    bool ok = g_det.loadModelFromPath(p,b,ts,gpu,nt);
    env->ReleaseStringUTFChars(jp,p);
    env->ReleaseStringUTFChars(jb,b);
    return (jboolean)ok;
}

JNIEXPORT jfloatArray JNICALL
Java_com_yolov8ncnn_YoloV8Ncnn_nativeDetectBitmap(
    JNIEnv* env, jclass, jobject bmp, jfloat cTh, jfloat nTh) {
    AndroidBitmapInfo info;
    AndroidBitmap_getInfo(env, bmp, &info);
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) return nullptr;
    void* px=nullptr;
    AndroidBitmap_lockPixels(env, bmp, &px);
    if (!px) return nullptr;
    float ms=0;
    auto boxes = g_det.detect((const unsigned char*)px, ncnn::Mat::PIXEL_RGBA2RGB,
                              info.width, info.height, cTh, nTh, ms);
    AndroidBitmap_unlockPixels(env, bmp);
    int n=(int)boxes.size(), sz=2+n*6;
    std::vector<float> d(sz);
    d[0]=ms; d[1]=(float)n;
    for(int i=0;i<n;i++){int o=2+i*6;
        d[o]=boxes[i].x1;d[o+1]=boxes[i].y1;
        d[o+2]=boxes[i].x2;d[o+3]=boxes[i].y2;
        d[o+4]=boxes[i].score;d[o+5]=(float)boxes[i].label;}
    jfloatArray r=env->NewFloatArray(sz);
    env->SetFloatArrayRegion(r,0,sz,d.data());
    return r;
}

JNIEXPORT jfloatArray JNICALL
Java_com_yolov8ncnn_YoloV8Ncnn_nativeDetectBuffer(
    JNIEnv* env, jclass, jobject buf, jint w, jint h, jint rs,
    jfloat cTh, jfloat nTh) {
    uint8_t* bp=(uint8_t*)env->GetDirectBufferAddress(buf);
    if (!bp) return nullptr;
    int exp=w*4; std::vector<uint8_t> cmp;
    const unsigned char* px=bp;
    if (rs!=exp) {
        cmp.resize(w*h*4);
        for(int y=0;y<h;y++) memcpy(cmp.data()+y*exp,bp+y*rs,exp);
        px=cmp.data();
    }
    float ms=0;
    auto boxes=g_det.detect(px,ncnn::Mat::PIXEL_RGBA2RGB,w,h,cTh,nTh,ms);
    int n=(int)boxes.size(),sz=2+n*6;
    std::vector<float> d(sz);
    d[0]=ms;d[1]=(float)n;
    for(int i=0;i<n;i++){int o=2+i*6;
        d[o]=boxes[i].x1;d[o+1]=boxes[i].y1;
        d[o+2]=boxes[i].x2;d[o+3]=boxes[i].y2;
        d[o+4]=boxes[i].score;d[o+5]=(float)boxes[i].label;}
    jfloatArray r=env->NewFloatArray(sz);
    env->SetFloatArrayRegion(r,0,sz,d.data());
    return r;
}

JNIEXPORT jboolean JNICALL
Java_com_yolov8ncnn_YoloV8Ncnn_nativeIsLoaded(JNIEnv*, jclass) {
    return (jboolean)g_det.isLoaded();
}

JNIEXPORT jboolean JNICALL
Java_com_yolov8ncnn_YoloV8Ncnn_nativeHasGpu(JNIEnv*, jclass) {
    return (jboolean)g_det.hasGpu();
}

} // extern "C"
