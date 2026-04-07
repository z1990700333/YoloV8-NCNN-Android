package com.yolov8ncnn;

import android.content.Context;
import android.content.SharedPreferences;

/**
 * Manages app settings via SharedPreferences.
 */
public class SettingsManager {
    private static final String PREFS_NAME = "yolov8_settings";

    private static final String KEY_CONF_THRESH = "conf_thresh";
    private static final String KEY_NMS_THRESH = "nms_thresh";
    private static final String KEY_TARGET_SIZE = "target_size";
    private static final String KEY_USE_GPU = "use_gpu";
    private static final String KEY_NUM_THREADS = "num_threads";
    private static final String KEY_PARAM_PATH = "param_path";
    private static final String KEY_BIN_PATH = "bin_path";
    private static final String KEY_CLICK_X = "click_x";
    private static final String KEY_CLICK_Y = "click_y";
    private static final String KEY_AUTO_CLICK = "auto_click";
    private static final String KEY_CLICK_DELAY = "click_delay";
    private static final String KEY_TARGET_LABEL = "target_label";
    private static final String KEY_CAPTURE_SCALE = "capture_scale";

    private final SharedPreferences prefs;

    public SettingsManager(Context context) {
        prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE);
    }

    // ── Confidence threshold ────────────────────────────────────────────
    public float getConfThresh() {
        return prefs.getFloat(KEY_CONF_THRESH, 0.35f);
    }

    public void setConfThresh(float value) {
        prefs.edit().putFloat(KEY_CONF_THRESH, value).apply();
    }

    // ── NMS threshold ───────────────────────────────────────────────────
    public float getNmsThresh() {
        return prefs.getFloat(KEY_NMS_THRESH, 0.45f);
    }

    public void setNmsThresh(float value) {
        prefs.edit().putFloat(KEY_NMS_THRESH, value).apply();
    }

    // ── Target input size ───────────────────────────────────────────────
    public int getTargetSize() {
        return prefs.getInt(KEY_TARGET_SIZE, 320);
    }

    public void setTargetSize(int value) {
        prefs.edit().putInt(KEY_TARGET_SIZE, value).apply();
    }

    // ── GPU toggle ──────────────────────────────────────────────────────
    public boolean getUseGpu() {
        return prefs.getBoolean(KEY_USE_GPU, false);
    }

    public void setUseGpu(boolean value) {
        prefs.edit().putBoolean(KEY_USE_GPU, value).apply();
    }

    // ── Thread count ────────────────────────────────────────────────────
    public int getNumThreads() {
        return prefs.getInt(KEY_NUM_THREADS, 4);
    }

    public void setNumThreads(int value) {
        prefs.edit().putInt(KEY_NUM_THREADS, value).apply();
    }

    // ── Model paths ─────────────────────────────────────────────────────
    public String getParamPath() {
        return prefs.getString(KEY_PARAM_PATH, "");
    }

    public void setParamPath(String path) {
        prefs.edit().putString(KEY_PARAM_PATH, path).apply();
    }

    public String getBinPath() {
        return prefs.getString(KEY_BIN_PATH, "");
    }

    public void setBinPath(String path) {
        prefs.edit().putString(KEY_BIN_PATH, path).apply();
    }

    // ── Click coordinates ───────────────────────────────────────────────
    public int getClickX() {
        return prefs.getInt(KEY_CLICK_X, -1);
    }

    public void setClickX(int x) {
        prefs.edit().putInt(KEY_CLICK_X, x).apply();
    }

    public int getClickY() {
        return prefs.getInt(KEY_CLICK_Y, -1);
    }

    public void setClickY(int y) {
        prefs.edit().putInt(KEY_CLICK_Y, y).apply();
    }

    // ── Auto click ──────────────────────────────────────────────────────
    public boolean getAutoClick() {
        return prefs.getBoolean(KEY_AUTO_CLICK, false);
    }

    public void setAutoClick(boolean value) {
        prefs.edit().putBoolean(KEY_AUTO_CLICK, value).apply();
    }

    // ── Click delay (ms) ────────────────────────────────────────────────
    public int getClickDelay() {
        return prefs.getInt(KEY_CLICK_DELAY, 500);
    }

    public void setClickDelay(int ms) {
        prefs.edit().putInt(KEY_CLICK_DELAY, ms).apply();
    }

    // ── Target label filter (-1 = all) ──────────────────────────────────
    public int getTargetLabel() {
        return prefs.getInt(KEY_TARGET_LABEL, -1);
    }

    public void setTargetLabel(int label) {
        prefs.edit().putInt(KEY_TARGET_LABEL, label).apply();
    }

    // ── Capture scale (0.25 - 1.0) ─────────────────────────────────────
    public float getCaptureScale() {
        return prefs.getFloat(KEY_CAPTURE_SCALE, 0.5f);
    }

    public void setCaptureScale(float scale) {
        prefs.edit().putFloat(KEY_CAPTURE_SCALE, scale).apply();
    }
}
