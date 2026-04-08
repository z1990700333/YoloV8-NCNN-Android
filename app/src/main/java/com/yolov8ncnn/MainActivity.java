package com.yolov8ncnn;

import android.Manifest;
import android.app.Activity;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.pm.PackageManager;
import android.media.projection.MediaProjectionManager;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.Looper;
import android.provider.Settings;
import android.util.Log;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.EditText;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.ScrollView;
import android.widget.SeekBar;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.lang.ref.WeakReference;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "MainActivity";
    private static final int REQUEST_MEDIA_PROJECTION = 1001;
    private static final int REQUEST_OVERLAY = 1002;
    private static final int REQUEST_ACCESSIBILITY = 1003;
    private static final int REQUEST_PICK_PARAM = 1004;
    private static final int REQUEST_PICK_BIN = 1005;
    private static final int REQUEST_STORAGE = 1006;
    private static final int REQUEST_NOTIFICATIONS = 1007;
    private static final int REQUEST_MANAGE_STORAGE = 1008;

    // 静态引用，供 Service 写日志
    private static WeakReference<MainActivity> sInstance;
    private static final Handler sMainHandler = new Handler(Looper.getMainLooper());
    private static final SimpleDateFormat sTimeFmt = new SimpleDateFormat("HH:mm:ss.SSS", Locale.US);

    private SettingsManager settings;

    private SeekBar seekConfThresh, seekNmsThresh, seekCaptureScale;
    private TextView tvConfValue, tvNmsValue, tvCaptureScaleValue;
    private Spinner spinnerTargetSize, spinnerThreads;
    private CheckBox cbUseGpu;
    private Button btnSelectParam, btnSelectBin, btnLoadModel;
    private Button btnStartOverlay, btnStartCapture;
    private Button btnGrantOverlay, btnGrantAccessibility, btnGrantStorage, btnGrantNotification;
    private Button btnClearLog;
    private TextView tvParamPath, tvBinPath, tvModelStatus;
    private TextView tvOverlayStatus, tvAccessibilityStatus, tvStorageStatus, tvNotificationStatus;
    private TextView tvRootStatus;
    private TextView tvMainLog;
    private ScrollView svLog;
    private EditText etTargetLabel, etClickDelay;
    private RadioGroup rgCaptureMode;
    private RadioButton rbMediaProjection, rbRoot;

    private BroadcastReceiver startCaptureReceiver;

    /**
     * 静态方法：从任何地方写日志到主界面（线程安全）
     */
    public static void appendLog(String msg) {
        sMainHandler.post(() -> {
            MainActivity act = sInstance != null ? sInstance.get() : null;
            if (act != null && act.tvMainLog != null) {
                String time = sTimeFmt.format(new Date());
                act.tvMainLog.append("[" + time + "] " + msg + "\n");
                // 自动滚动到底部
                if (act.svLog != null) {
                    act.svLog.post(() -> act.svLog.fullScroll(View.FOCUS_DOWN));
                }
            }
        });
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        sInstance = new WeakReference<>(this);
        settings = new SettingsManager(this);
        YoloV8Ncnn.nativeInit();

        initViews();
        loadSettings();
        updatePermissionStatus();
        registerReceivers();

        appendLog("应用启动");
    }

    private void initViews() {
        // 置信度
        seekConfThresh = findViewById(R.id.seek_conf_thresh);
        tvConfValue = findViewById(R.id.tv_conf_value);
        seekConfThresh.setMax(100);
        seekConfThresh.setOnSeekBarChangeListener(new SimpleSeekBarListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                float val = progress / 100.0f;
                tvConfValue.setText(String.format("%.2f", val));
                if (fromUser) settings.setConfThresh(val);
            }
        });

        // NMS
        seekNmsThresh = findViewById(R.id.seek_nms_thresh);
        tvNmsValue = findViewById(R.id.tv_nms_value);
        seekNmsThresh.setMax(100);
        seekNmsThresh.setOnSeekBarChangeListener(new SimpleSeekBarListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                float val = progress / 100.0f;
                tvNmsValue.setText(String.format("%.2f", val));
                if (fromUser) settings.setNmsThresh(val);
            }
        });

        // 捕获缩放
        seekCaptureScale = findViewById(R.id.seek_capture_scale);
        tvCaptureScaleValue = findViewById(R.id.tv_capture_scale_value);
        seekCaptureScale.setMax(100);
        seekCaptureScale.setOnSeekBarChangeListener(new SimpleSeekBarListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                float val = Math.max(0.1f, progress / 100.0f);
                tvCaptureScaleValue.setText(String.format("%.0f%%", val * 100));
                if (fromUser) settings.setCaptureScale(val);
            }
        });

        // 输入尺寸
        spinnerTargetSize = findViewById(R.id.spinner_target_size);
        ArrayAdapter<String> sizeAdapter = new ArrayAdapter<>(this,
                android.R.layout.simple_spinner_dropdown_item,
                new String[]{"320", "416", "640"});
        spinnerTargetSize.setAdapter(sizeAdapter);
        spinnerTargetSize.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override public void onItemSelected(AdapterView<?> p, View v, int pos, long id) {
                int[] sizes = {320, 416, 640};
                settings.setTargetSize(sizes[pos]);
            }
            @Override public void onNothingSelected(AdapterView<?> p) {}
        });

        // 线程数
        spinnerThreads = findViewById(R.id.spinner_threads);
        ArrayAdapter<String> threadAdapter = new ArrayAdapter<>(this,
                android.R.layout.simple_spinner_dropdown_item,
                new String[]{"1", "2", "4", "6", "8"});
        spinnerThreads.setAdapter(threadAdapter);
        spinnerThreads.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override public void onItemSelected(AdapterView<?> p, View v, int pos, long id) {
                int[] threads = {1, 2, 4, 6, 8};
                settings.setNumThreads(threads[pos]);
            }
            @Override public void onNothingSelected(AdapterView<?> p) {}
        });

        // GPU
        cbUseGpu = findViewById(R.id.cb_use_gpu);
        cbUseGpu.setOnCheckedChangeListener((b, checked) -> settings.setUseGpu(checked));

        // 模型选择
        btnSelectParam = findViewById(R.id.btn_select_param);
        btnSelectBin = findViewById(R.id.btn_select_bin);
        btnLoadModel = findViewById(R.id.btn_load_model);
        tvParamPath = findViewById(R.id.tv_param_path);
        tvBinPath = findViewById(R.id.tv_bin_path);
        tvModelStatus = findViewById(R.id.tv_model_status);

        btnSelectParam.setOnClickListener(v -> pickFile(REQUEST_PICK_PARAM));
        btnSelectBin.setOnClickListener(v -> pickFile(REQUEST_PICK_BIN));
        btnLoadModel.setOnClickListener(v -> loadModel());

        // 目标标签 & 点击延迟
        etTargetLabel = findViewById(R.id.et_target_label);
        etClickDelay = findViewById(R.id.et_click_delay);

        // 截图模式
        rgCaptureMode = findViewById(R.id.rg_capture_mode);
        rbMediaProjection = findViewById(R.id.rb_media_projection);
        rbRoot = findViewById(R.id.rb_root);
        tvRootStatus = findViewById(R.id.tv_root_status);

        // 异步检测 Root
        new Thread(() -> {
            boolean hasRoot = ScreenCaptureService.checkRootAccess();
            runOnUiThread(() -> {
                if (hasRoot) {
                    tvRootStatus.setText("Root: 可用");
                    tvRootStatus.setTextColor(0xFF4CAF50);
                    // Root 自动授权无障碍服务
                    if (!AutoClickService.isRunning()) {
                        autoGrantAccessibilityViaRoot();
                    }
                } else {
                    tvRootStatus.setText("Root: 不可用");
                    tvRootStatus.setTextColor(0xFFF44336);
                    rbRoot.setEnabled(false);
                }
            });
        }).start();

        // Sync radio buttons with saved capture mode
        int savedMode = settings.getCaptureMode();
        if (savedMode == 1) rbRoot.setChecked(true);
        else rbMediaProjection.setChecked(true);

        rgCaptureMode.setOnCheckedChangeListener((group, checkedId) -> {
            if (checkedId == R.id.rb_root) settings.setCaptureMode(1);
            else settings.setCaptureMode(0);
        });

        // 权限按钮
        btnGrantOverlay = findViewById(R.id.btn_grant_overlay);
        btnGrantAccessibility = findViewById(R.id.btn_grant_accessibility);
        btnGrantStorage = findViewById(R.id.btn_grant_storage);
        btnGrantNotification = findViewById(R.id.btn_grant_notification);
        tvOverlayStatus = findViewById(R.id.tv_overlay_status);
        tvAccessibilityStatus = findViewById(R.id.tv_accessibility_status);
        tvStorageStatus = findViewById(R.id.tv_storage_status);
        tvNotificationStatus = findViewById(R.id.tv_notification_status);

        btnGrantOverlay.setOnClickListener(v -> requestOverlayPermission());
        btnGrantAccessibility.setOnClickListener(v -> requestAccessibilityPermission());
        btnGrantStorage.setOnClickListener(v -> requestStoragePermission());
        btnGrantNotification.setOnClickListener(v -> requestNotificationPermission());

        // 操作按钮
        btnStartOverlay = findViewById(R.id.btn_start_overlay);
        btnStartCapture = findViewById(R.id.btn_start_capture);

        btnStartOverlay.setOnClickListener(v -> startOverlayService());
        btnStartCapture.setOnClickListener(v -> startScreenCapture());

        // 日志面板
        tvMainLog = findViewById(R.id.tv_main_log);
        svLog = findViewById(R.id.sv_log);
        btnClearLog = findViewById(R.id.btn_clear_log);
        btnClearLog.setOnClickListener(v -> {
            tvMainLog.setText("");
            appendLog("日志已清除");
        });

        Button btnSaveSettings = findViewById(R.id.btn_save_settings);
        btnSaveSettings.setOnClickListener(v -> saveExtraSettings());
    }

    private void loadSettings() {
        seekConfThresh.setProgress((int)(settings.getConfThresh() * 100));
        tvConfValue.setText(String.format("%.2f", settings.getConfThresh()));

        seekNmsThresh.setProgress((int)(settings.getNmsThresh() * 100));
        tvNmsValue.setText(String.format("%.2f", settings.getNmsThresh()));

        seekCaptureScale.setProgress((int)(settings.getCaptureScale() * 100));
        tvCaptureScaleValue.setText(String.format("%.0f%%", settings.getCaptureScale() * 100));

        int targetSize = settings.getTargetSize();
        spinnerTargetSize.setSelection(targetSize == 320 ? 0 : targetSize == 416 ? 1 : 2);

        int threads = settings.getNumThreads();
        spinnerThreads.setSelection(threads <= 1 ? 0 : threads <= 2 ? 1 : threads <= 4 ? 2 : threads <= 6 ? 3 : 4);

        cbUseGpu.setChecked(settings.getUseGpu());

        String paramPath = settings.getParamPath();
        String binPath = settings.getBinPath();
        tvParamPath.setText(paramPath.isEmpty() ? "未选择" : new File(paramPath).getName());
        tvBinPath.setText(binPath.isEmpty() ? "未选择" : new File(binPath).getName());

        etTargetLabel.setText(settings.getTargetLabel() == -1 ? "-1" : String.valueOf(settings.getTargetLabel()));
        etClickDelay.setText(String.valueOf(settings.getClickDelay()));

        tvModelStatus.setText(YoloV8Ncnn.nativeIsLoaded() ? "模型: 已加载" : "模型: 未加载");
    }

    private void saveExtraSettings() {
        try {
            int label = Integer.parseInt(etTargetLabel.getText().toString().trim());
            settings.setTargetLabel(label);
        } catch (NumberFormatException e) { settings.setTargetLabel(-1); }

        try {
            int delay = Integer.parseInt(etClickDelay.getText().toString().trim());
            settings.setClickDelay(Math.max(100, delay));
        } catch (NumberFormatException e) { settings.setClickDelay(500); }

        Toast.makeText(this, "设置已保存", Toast.LENGTH_SHORT).show();
    }

    private void loadModel() {
        String paramPath = settings.getParamPath();
        String binPath = settings.getBinPath();

        if (paramPath.isEmpty() || binPath.isEmpty()) {
            Toast.makeText(this, "请先选择 .param 和 .bin 文件", Toast.LENGTH_SHORT).show();
            return;
        }

        int targetSize = settings.getTargetSize();
        boolean useGpu = settings.getUseGpu();
        int numThreads = settings.getNumThreads();

        tvModelStatus.setText("模型: 加载中...");
        Log.i(TAG, String.format("加载模型: size=%d gpu=%b threads=%d param=%s",
                targetSize, useGpu, numThreads, paramPath));

        new Thread(() -> {
            boolean ok = YoloV8Ncnn.nativeLoadModelPath(paramPath, binPath,
                    targetSize, useGpu, numThreads);
            runOnUiThread(() -> {
                if (ok) {
                    tvModelStatus.setText(String.format("模型: 已加载 (size=%d, thr=%d)", targetSize, numThreads));
                    Toast.makeText(this, "模型加载成功!", Toast.LENGTH_SHORT).show();
                } else {
                    tvModelStatus.setText("模型: 加载失败");
                    Toast.makeText(this, "模型加载失败，请检查文件", Toast.LENGTH_LONG).show();
                }
            });
        }).start();
    }

    private void pickFile(int requestCode) {
        Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
        intent.setType("*/*");
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        startActivityForResult(Intent.createChooser(intent, "选择文件"), requestCode);
    }

    // ═══ 权限 ═══════════════════════════════════════════════════════════
    private void requestOverlayPermission() {
        if (!Settings.canDrawOverlays(this)) {
            startActivityForResult(new Intent(Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
                    Uri.parse("package:" + getPackageName())), REQUEST_OVERLAY);
        }
    }

    private void requestAccessibilityPermission() {
        startActivityForResult(new Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS), REQUEST_ACCESSIBILITY);
    }

    private void requestStoragePermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            if (!Environment.isExternalStorageManager()) {
                startActivityForResult(new Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION,
                        Uri.parse("package:" + getPackageName())), REQUEST_MANAGE_STORAGE);
            }
        } else {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.READ_EXTERNAL_STORAGE, Manifest.permission.WRITE_EXTERNAL_STORAGE},
                    REQUEST_STORAGE);
        }
    }

    private void requestNotificationPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.POST_NOTIFICATIONS}, REQUEST_NOTIFICATIONS);
        }
    }

    private void updatePermissionStatus() {
        boolean hasOverlay = Settings.canDrawOverlays(this);
        tvOverlayStatus.setText(hasOverlay ? "✓ 已授权" : "✗ 需授权");
        tvOverlayStatus.setTextColor(hasOverlay ? 0xFF4CAF50 : 0xFFF44336);

        boolean hasAcc = AutoClickService.isRunning();
        tvAccessibilityStatus.setText(hasAcc ? "✓ 运行中" : "✗ 未启用");
        tvAccessibilityStatus.setTextColor(hasAcc ? 0xFF4CAF50 : 0xFFF44336);

        boolean hasStorage;
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            hasStorage = Environment.isExternalStorageManager();
        } else {
            hasStorage = ContextCompat.checkSelfPermission(this,
                    Manifest.permission.READ_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED;
        }
        tvStorageStatus.setText(hasStorage ? "✓ 已授权" : "✗ 需授权");
        tvStorageStatus.setTextColor(hasStorage ? 0xFF4CAF50 : 0xFFF44336);

        boolean hasNotif = true;
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            hasNotif = ContextCompat.checkSelfPermission(this,
                    Manifest.permission.POST_NOTIFICATIONS) == PackageManager.PERMISSION_GRANTED;
        }
        tvNotificationStatus.setText(hasNotif ? "✓ 已授权" : "✗ 需授权");
        tvNotificationStatus.setTextColor(hasNotif ? 0xFF4CAF50 : 0xFFF44336);
    }

    // ═══ 启动服务 ═══════════════════════════════════════════════════════
    private void startOverlayService() {
        if (!Settings.canDrawOverlays(this)) {
            Toast.makeText(this, "请先授权悬浮窗权限!", Toast.LENGTH_SHORT).show();
            requestOverlayPermission();
            return;
        }
        Intent intent = new Intent(this, OverlayService.class);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) startForegroundService(intent);
        else startService(intent);
        Toast.makeText(this, "悬浮窗已启动", Toast.LENGTH_SHORT).show();
    }

    private void startScreenCapture() {
        if (!YoloV8Ncnn.nativeIsLoaded()) {
            Toast.makeText(this, "请先加载模型!", Toast.LENGTH_SHORT).show();
            return;
        }

        // 确保悬浮窗已启动
        if (OverlayService.getInstance() == null) {
            startOverlayService();
            try { Thread.sleep(300); } catch (InterruptedException ignored) {}
        }

        boolean useRoot = rbRoot.isChecked();
        settings.setCaptureMode(useRoot ? 1 : 0);
        Log.i(TAG, "开始截图 useRoot=" + useRoot + " modelLoaded=" + YoloV8Ncnn.nativeIsLoaded());

        OverlayService ov = OverlayService.getInstance();
        if (ov != null) ov.appendLog("从主界面启动: " + (useRoot ? "Root" : "MediaProjection"));

        if (useRoot) {
            Intent serviceIntent = new Intent(this, ScreenCaptureService.class);
            serviceIntent.putExtra(ScreenCaptureService.EXTRA_USE_ROOT, true);
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) startForegroundService(serviceIntent);
            else startService(serviceIntent);
            setupCallback();
            Toast.makeText(this, "Root 截图模式已启动!", Toast.LENGTH_SHORT).show();
        } else {
            Log.i(TAG, "请求 MediaProjection 授权...");
            if (ov != null) ov.appendLog("请求屏幕录制授权...");
            MediaProjectionManager mpManager =
                    (MediaProjectionManager) getSystemService(MEDIA_PROJECTION_SERVICE);
            startActivityForResult(mpManager.createScreenCaptureIntent(), REQUEST_MEDIA_PROJECTION);
        }
    }

    private void setupCallback() {
        new Thread(() -> {
            // 等待服务启动，最多等5秒
            for (int i = 0; i < 50; i++) {
                try { Thread.sleep(100); } catch (InterruptedException ignored) {}
                ScreenCaptureService service = ScreenCaptureService.getInstance();
                if (service != null && ScreenCaptureService.isServiceRunning()) {
                    Log.i(TAG, "ScreenCaptureService ready after " + (i*100) + "ms");
                    OverlayService ov = OverlayService.getInstance();
                    if (ov != null) ov.appendLog("截图服务已就绪 (" + (i*100) + "ms)");
                    service.setCallback((boxes, inferTimeMs, fps, captureW, captureH) -> {
                        OverlayService overlay = OverlayService.getInstance();
                        if (overlay != null) {
                            overlay.updateDetections(boxes, inferTimeMs, fps, captureW, captureH);
                        }
                    });
                    OverlayService overlay = OverlayService.getInstance();
                    if (overlay != null) overlay.setCapturing(true);
                    return;
                }
            }
            Log.e(TAG, "ScreenCaptureService did not start in time");
            OverlayService ov = OverlayService.getInstance();
            if (ov != null) ov.appendLog("截图服务启动超时!");
        }).start();
    }

    // ═══ Activity Results ═══════════════════════════════════════════════
    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        switch (requestCode) {
            case REQUEST_MEDIA_PROJECTION:
                Log.i(TAG, "MP onActivityResult: resultCode=" + resultCode + " data=" + data);
                OverlayService ovMP = OverlayService.getInstance();
                if (resultCode == RESULT_OK && data != null) {
                    if (ovMP != null) ovMP.appendLog("MP授权成功, 启动服务...");
                    Intent serviceIntent = new Intent(this, ScreenCaptureService.class);
                    serviceIntent.putExtra(ScreenCaptureService.EXTRA_RESULT_CODE, resultCode);
                    serviceIntent.putExtra(ScreenCaptureService.EXTRA_DATA, data);
                    serviceIntent.putExtra(ScreenCaptureService.EXTRA_USE_ROOT, false);
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) startForegroundService(serviceIntent);
                    else startService(serviceIntent);
                    setupCallback();
                    Toast.makeText(this, "屏幕捕获已启动!", Toast.LENGTH_SHORT).show();
                } else {
                    Log.w(TAG, "MP授权被拒绝: resultCode=" + resultCode);
                    if (ovMP != null) ovMP.appendLog("MP授权被拒绝: code=" + resultCode);
                    Toast.makeText(this, "屏幕录制授权被拒绝", Toast.LENGTH_SHORT).show();
                }
                break;
            case REQUEST_PICK_PARAM:
                if (resultCode == RESULT_OK && data != null) {
                    String path = copyUriToLocal(data.getData(), "model.param");
                    if (path != null) { settings.setParamPath(path); tvParamPath.setText(new File(path).getName()); }
                }
                break;
            case REQUEST_PICK_BIN:
                if (resultCode == RESULT_OK && data != null) {
                    String path = copyUriToLocal(data.getData(), "model.bin");
                    if (path != null) { settings.setBinPath(path); tvBinPath.setText(new File(path).getName()); }
                }
                break;
            case REQUEST_OVERLAY:
            case REQUEST_ACCESSIBILITY:
            case REQUEST_MANAGE_STORAGE:
                updatePermissionStatus();
                break;
        }
    }

    private String copyUriToLocal(Uri uri, String fileName) {
        try {
            File outFile = new File(getFilesDir(), fileName);
            InputStream is = getContentResolver().openInputStream(uri);
            FileOutputStream fos = new FileOutputStream(outFile);
            byte[] buffer = new byte[8192];
            int len;
            while ((len = is.read(buffer)) != -1) fos.write(buffer, 0, len);
            fos.close(); is.close();
            return outFile.getAbsolutePath();
        } catch (Exception e) {
            Toast.makeText(this, "文件读取失败: " + e.getMessage(), Toast.LENGTH_LONG).show();
            return null;
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        updatePermissionStatus();
    }

    @Override protected void onResume() {
        super.onResume();
        sInstance = new WeakReference<>(this);
        updatePermissionStatus();
    }

    /**
     * 通过 Root 自动启用无障碍服务
     */
    private void autoGrantAccessibilityViaRoot() {
        new Thread(() -> {
            try {
                String pkg = getPackageName();
                String svc = pkg + "/.AutoClickService";
                Process su = Runtime.getRuntime().exec("su");
                java.io.OutputStream os = su.getOutputStream();
                // 先获取当前已启用的无障碍服务
                os.write(("settings get secure enabled_accessibility_services\n").getBytes());
                os.flush();
                // 设置无障碍服务
                os.write(("settings put secure enabled_accessibility_services " + svc + "\n").getBytes());
                os.write("settings put secure accessibility_enabled 1\n".getBytes());
                os.write("exit\n".getBytes());
                os.flush();
                int ret = su.waitFor();
                String msg = ret == 0 ? "Root自动授权无障碍成功" : "Root自动授权无障碍失败(code=" + ret + ")";
                appendLog(msg);
                runOnUiThread(() -> {
                    // 延迟刷新状态
                    new Handler(Looper.getMainLooper()).postDelayed(this::updatePermissionStatus, 1000);
                });
            } catch (Exception e) {
                appendLog("Root授权无障碍异常: " + e.getMessage());
            }
        }).start();
    }

    @Override
    protected void onDestroy() {
        if (startCaptureReceiver != null) unregisterReceiver(startCaptureReceiver);
        sInstance = null;
        super.onDestroy();
    }

    private void registerReceivers() {
        startCaptureReceiver = new BroadcastReceiver() {
            @Override public void onReceive(Context context, Intent intent) { startScreenCapture(); }
        };
        IntentFilter filter = new IntentFilter("com.yolov8ncnn.START_CAPTURE");
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            registerReceiver(startCaptureReceiver, filter, Context.RECEIVER_NOT_EXPORTED);
        } else {
            registerReceiver(startCaptureReceiver, filter);
        }
    }

    abstract static class SimpleSeekBarListener implements SeekBar.OnSeekBarChangeListener {
        @Override public void onStartTrackingTouch(SeekBar seekBar) {}
        @Override public void onStopTrackingTouch(SeekBar seekBar) {}
    }
}
