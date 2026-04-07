@echo off
REM ═══════════════════════════════════════════════════════════════════════════
REM YOLOv8 NCNN Android App - Windows Setup Script
REM Downloads NCNN prebuilt libraries and sets up Gradle wrapper
REM ═══════════════════════════════════════════════════════════════════════════

echo ================================================
echo   YOLOv8 NCNN Android App - Setup (Windows)
echo ================================================
echo.

set NCNN_VERSION=20240410
set NCNN_URL=https://github.com/Tencent/ncnn/releases/download/%NCNN_VERSION%/ncnn-%NCNN_VERSION%-android-vulkan.zip
set NCNN_DIR=app\src\main\jni\ncnn-android-vulkan

REM ── 1. Download NCNN ──────────────────────────────────────────────────
if not exist "%NCNN_DIR%" (
    echo [1/3] Downloading NCNN %NCNN_VERSION% Android Vulkan...
    echo URL: %NCNN_URL%

    where curl >nul 2>&1
    if %errorlevel%==0 (
        curl -L -o ncnn-android-vulkan.zip "%NCNN_URL%"
    ) else (
        echo ERROR: curl not found. Please download manually from:
        echo %NCNN_URL%
        echo Extract to: %NCNN_DIR%
        goto :check_gradle
    )

    echo Extracting NCNN...
    powershell -Command "Expand-Archive -Path 'ncnn-android-vulkan.zip' -DestinationPath 'app\src\main\jni\' -Force"
    rename "app\src\main\jni\ncnn-%NCNN_VERSION%-android-vulkan" "ncnn-android-vulkan"
    move "app\src\main\jni\ncnn-android-vulkan" "%NCNN_DIR%" >nul 2>&1
    del ncnn-android-vulkan.zip
    echo NCNN extracted to: %NCNN_DIR%
) else (
    echo [1/3] NCNN already exists, skipping download.
)

:check_gradle
REM ── 2. Setup Gradle wrapper ────────────────────────────────────────────
if not exist "gradle\wrapper" mkdir "gradle\wrapper"

echo [2/3] Creating Gradle wrapper properties...

(
echo distributionBase=GRADLE_USER_HOME
echo distributionPath=wrapper/dists
echo distributionUrl=https\://services.gradle.org/distributions/gradle-8.4-bin.zip
echo networkTimeout=10000
echo validateDistributionUrl=true
echo zipStoreBase=GRADLE_USER_HOME
echo zipStorePath=wrapper/dists
) > gradle\wrapper\gradle-wrapper.properties

REM Create gradlew.bat
(
echo @rem Gradle wrapper script for Windows
echo @if "%%DEBUG%%"=="" @echo off
echo set DIRNAME=%%~dp0
echo set CLASSPATH=%%DIRNAME%%\gradle\wrapper\gradle-wrapper.jar
echo set JAVACMD=java
echo if defined JAVA_HOME set JAVACMD=%%JAVA_HOME%%\bin\java
echo "%%JAVACMD%%" -classpath "%%CLASSPATH%%" org.gradle.wrapper.GradleWrapperMain %%*
) > gradlew.bat

echo Gradle wrapper created.

REM ── 3. Check environment ──────────────────────────────────────────────
echo.
echo [3/3] Checking environment...

if defined ANDROID_HOME (
    echo   OK: ANDROID_HOME=%ANDROID_HOME%
) else if defined ANDROID_SDK_ROOT (
    echo   OK: ANDROID_SDK_ROOT=%ANDROID_SDK_ROOT%
) else (
    echo   WARN: ANDROID_HOME not set!
    echo         Please set: set ANDROID_HOME=C:\path\to\android-sdk
)

if exist "%NCNN_DIR%\arm64-v8a" (
    echo   OK: NCNN arm64-v8a found
) else (
    echo   ERROR: NCNN arm64-v8a not found at %NCNN_DIR%\arm64-v8a
)

echo.
echo ================================================
echo   Setup complete!
echo ================================================
echo.
echo To build: gradlew.bat assembleDebug
echo APK at:   app\build\outputs\apk\debug\app-debug.apk
echo.
echo IMPORTANT: You need gradle-wrapper.jar
echo Download from: https://github.com/nicoulaj/gradle-wrapper/releases
echo Place at: gradle\wrapper\gradle-wrapper.jar
echo.
pause
