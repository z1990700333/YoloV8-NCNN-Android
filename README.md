# YOLOv8 NCNN Android - Real-time Screen Detection

High-performance Android app for real-time screen object detection using YOLOv8 + NCNN.

## Features
- Real-time screen capture inference (30+ FPS target)
- C++ NCNN inference engine (maximum performance)
- Custom model loading (.param + .bin files)
- Floating overlay with FPS, inference time, detection boxes
- Auto-click at specified coordinates when target detected
- Configurable: confidence threshold, NMS threshold, input size, GPU, threads

## Prerequisites

### 1. Install JDK 17
Download from: https://adoptium.net/temurin/releases/
Set `JAVA_HOME` environment variable.

### 2. Install Android SDK Command Line Tools
Download from: https://developer.android.com/studio#command-line-tools-only

```bash
# Create SDK directory
mkdir C:\android-sdk
# Extract cmdline-tools to C:\android-sdk\cmdline-tools\latest\

# Set environment variables
set ANDROID_HOME=C:\android-sdk
set PATH=%PATH%;%ANDROID_HOME%\cmdline-tools\latest\bin;%ANDROID_HOME%\platform-tools

# Install required SDK components
sdkmanager "platforms;android-34" "build-tools;34.0.0" "ndk;26.1.10909125" "cmake;3.22.1"
sdkmanager --licenses
```

### 3. Create local.properties
Create `local.properties` in project root:
```
sdk.dir=C\:\\android-sdk
ndk.dir=C\:\\android-sdk\\ndk\\26.1.10909125
```

## Setup

### Windows
```cmd
setup.bat
```

### Linux/Mac
```bash
chmod +x setup.sh
./setup.sh
```

### Manual NCNN Download
If the script fails, manually download NCNN:
1. Go to: https://github.com/Tencent/ncnn/releases
2. Download: `ncnn-XXXXXXXX-android-vulkan.zip`
3. Extract to: `app/src/main/jni/ncnn-android-vulkan/`
4. Verify structure: `app/src/main/jni/ncnn-android-vulkan/arm64-v8a/lib/libncnn.a`

### Gradle Wrapper JAR
You need `gradle-wrapper.jar`. Easiest way:
1. Install Gradle 8.4 globally: https://gradle.org/releases/
2. Run: `gradle wrapper` in the project directory
3. This generates the proper `gradlew`, `gradlew.bat`, and `gradle-wrapper.jar`

## Build

```bash
# Debug APK
gradlew.bat assembleDebug

# Release APK
gradlew.bat assembleRelease
```

APK output: `app/build/outputs/apk/debug/app-debug.apk`

## Usage

1. **Install APK** on Android device
2. **Grant permissions** (tap each permission button):
   - Overlay (floating window)
   - Accessibility (auto-click)
   - Storage (model file access)
   - Notification (foreground service)
3. **Select model files** (.param and .bin)
4. **Configure settings** (confidence, NMS, input size, threads)
5. **Load Model** button
6. **Start Overlay** - shows floating control panel
7. **Start Screen Capture** - begins real-time inference

### Floating Panel Controls
- **Start/Stop** - toggle screen capture
- **Set Click Pos** - tap screen to set auto-click coordinate
- **AutoClick ON/OFF** - toggle auto-click when target detected

### Preparing YOLOv8 NCNN Model
Export your YOLOv8 model to NCNN format:
```python
# Install ultralytics
pip install ultralytics

# Export to NCNN
from ultralytics import YOLO
model = YOLO('your_model.pt')
model.export(format='ncnn')
# Output: your_model_ncnn_model/model.ncnn.param + model.ncnn.bin
```

## Architecture

```
C++ Layer (yolov8ncnn_jni.cpp):
  - YoloV8Detector class
  - Letterbox resize + normalize
  - NCNN inference (CPU/Vulkan GPU)
  - Post-processing + NMS
  - JNI bridge functions

Java Layer:
  - MainActivity: Settings UI + permission management
  - ScreenCaptureService: MediaProjection screen capture
  - OverlayService: Floating window + detection overlay
  - AutoClickService: AccessibilityService auto-click
  - YoloV8Ncnn: JNI bridge class
```

## Performance Tips
- Use **YOLOv8n** (nano) for fastest inference
- Set input size to **320** for maximum FPS
- Set capture scale to **50%** or lower
- Enable **GPU (Vulkan)** on supported devices
- Use **4 threads** on most devices
- Lower confidence threshold reduces post-processing time
