cmake_minimum_required(VERSION 3.22.1)
project(yolov8ncnn)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── NCNN prebuilt library ──────────────────────────────────────────────────
# Downloaded from https://github.com/Tencent/ncnn/releases (v20260113)
# Unzipped to: app/src/main/jni/ncnn-android-vulkan/
set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnn-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

# ── Build JNI shared library ──────────────────────────────────────────────
add_library(yolov8ncnn SHARED yolov8ncnn_jni.cpp)

target_link_libraries(yolov8ncnn
    ncnn
    android
    jnigraphics
    log
    z
)

target_compile_options(yolov8ncnn PRIVATE
    -O3
    -ffast-math
    -fno-exceptions
)
