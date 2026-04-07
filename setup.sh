#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# YOLOv8 NCNN Android App - Setup Script
# Downloads NCNN prebuilt libraries and Gradle wrapper
# ═══════════════════════════════════════════════════════════════════════════

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================"
echo "  YOLOv8 NCNN Android App - Setup"
echo "================================================"

# ── 1. Download NCNN prebuilt Android Vulkan library ──────────────────────
NCNN_VERSION="20240410"
NCNN_URL="https://github.com/Tencent/ncnn/releases/download/${NCNN_VERSION}/ncnn-${NCNN_VERSION}-android-vulkan.zip"
NCNN_DIR="app/src/main/jni/ncnn-android-vulkan"

if [ ! -d "$NCNN_DIR" ]; then
    echo ""
    echo "[1/3] Downloading NCNN ${NCNN_VERSION} Android Vulkan..."
    echo "URL: ${NCNN_URL}"

    if command -v curl &> /dev/null; then
        curl -L -o ncnn-android-vulkan.zip "$NCNN_URL"
    elif command -v wget &> /dev/null; then
        wget -O ncnn-android-vulkan.zip "$NCNN_URL"
    else
        echo "ERROR: Neither curl nor wget found. Please install one."
        echo "Or manually download from: ${NCNN_URL}"
        exit 1
    fi

    echo "Extracting NCNN..."
    unzip -q ncnn-android-vulkan.zip -d app/src/main/jni/
    mv "app/src/main/jni/ncnn-${NCNN_VERSION}-android-vulkan" "$NCNN_DIR"
    rm ncnn-android-vulkan.zip
    echo "NCNN extracted to: ${NCNN_DIR}"
else
    echo "[1/3] NCNN already exists at ${NCNN_DIR}, skipping download."
fi

# ── 2. Setup Gradle wrapper ──────────────────────────────────────────────
if [ ! -f "gradlew" ]; then
    echo ""
    echo "[2/3] Setting up Gradle wrapper..."

    GRADLE_VERSION="8.4"
    GRADLE_DIST_URL="https://services.gradle.org/distributions/gradle-${GRADLE_VERSION}-bin.zip"

    mkdir -p gradle/wrapper

    cat > gradle/wrapper/gradle-wrapper.properties << EOF
distributionBase=GRADLE_USER_HOME
distributionPath=wrapper/dists
distributionUrl=https\://services.gradle.org/distributions/gradle-${GRADLE_VERSION}-bin.zip
networkTimeout=10000
validateDistributionUrl=true
zipStoreBase=GRADLE_USER_HOME
zipStorePath=wrapper/dists
EOF

    # Download gradle-wrapper.jar
    WRAPPER_JAR_URL="https://raw.githubusercontent.com/gradle/gradle/v${GRADLE_VERSION}/gradle/wrapper/gradle-wrapper.jar"
    echo "Downloading gradle-wrapper.jar..."
    if command -v curl &> /dev/null; then
        curl -L -o gradle/wrapper/gradle-wrapper.jar "$WRAPPER_JAR_URL" 2>/dev/null || true
    fi

    # Create gradlew script
    cat > gradlew << 'GRADLEW'
#!/bin/sh
# Gradle wrapper script
APP_NAME="Gradle"
APP_BASE_NAME=$(basename "$0")
DIRNAME=$(dirname "$0")
CLASSPATH=$DIRNAME/gradle/wrapper/gradle-wrapper.jar
JAVACMD="java"
if [ -n "$JAVA_HOME" ]; then
    JAVACMD="$JAVA_HOME/bin/java"
fi
exec "$JAVACMD" -classpath "$CLASSPATH" org.gradle.wrapper.GradleWrapperMain "$@"
GRADLEW
    chmod +x gradlew

    # Create gradlew.bat for Windows
    cat > gradlew.bat << 'GRADLEWBAT'
@rem Gradle wrapper script for Windows
@if "%DEBUG%"=="" @echo off
set DIRNAME=%~dp0
set CLASSPATH=%DIRNAME%\gradle\wrapper\gradle-wrapper.jar
set JAVACMD=java
if defined JAVA_HOME set JAVACMD=%JAVA_HOME%\bin\java
"%JAVACMD%" -classpath "%CLASSPATH%" org.gradle.wrapper.GradleWrapperMain %*
GRADLEWBAT

    echo "Gradle wrapper created."
else
    echo "[2/3] Gradle wrapper already exists, skipping."
fi

# ── 3. Verify setup ─────────────────────────────────────────────────────
echo ""
echo "[3/3] Verifying setup..."

ERRORS=0

if [ ! -d "$NCNN_DIR/arm64-v8a" ]; then
    echo "  ERROR: NCNN arm64-v8a not found at ${NCNN_DIR}/arm64-v8a"
    ERRORS=$((ERRORS + 1))
else
    echo "  OK: NCNN arm64-v8a found"
fi

if [ ! -d "$NCNN_DIR/armeabi-v7a" ]; then
    echo "  WARN: NCNN armeabi-v7a not found (optional)"
else
    echo "  OK: NCNN armeabi-v7a found"
fi

if [ ! -f "gradlew" ] && [ ! -f "gradlew.bat" ]; then
    echo "  ERROR: Gradle wrapper not found"
    ERRORS=$((ERRORS + 1))
else
    echo "  OK: Gradle wrapper found"
fi

if [ -z "$ANDROID_HOME" ] && [ -z "$ANDROID_SDK_ROOT" ]; then
    echo "  WARN: ANDROID_HOME not set. Set it before building."
    echo "        export ANDROID_HOME=/path/to/android-sdk"
else
    echo "  OK: ANDROID_HOME=${ANDROID_HOME:-$ANDROID_SDK_ROOT}"
fi

echo ""
if [ $ERRORS -eq 0 ]; then
    echo "================================================"
    echo "  Setup complete! Ready to build."
    echo "================================================"
    echo ""
    echo "To build the APK:"
    echo "  ./gradlew assembleDebug"
    echo ""
    echo "APK will be at:"
    echo "  app/build/outputs/apk/debug/app-debug.apk"
else
    echo "Setup completed with $ERRORS error(s). Please fix them before building."
fi
