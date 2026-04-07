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
import android.provider.Settings;
import android.util.Log;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.EditText;
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

/**
 * Main activity with settings UI and permission management.
 */
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

    private SettingsManager settings;

    // UI elements
    private SeekBar seekConfThresh, seekNmsThresh;
    private TextView tvConfValue, tvNmsValue;
    private Spinner spinnerTargetSize, spinnerThreads;
    private CheckBox cbUseGpu;
    private Button btnSelectParam, btnSelectBin, btnLoadModel;
    private Button btnStartOverlay, btnStartCapture;
    private Button btnGrantOverlay, btnGrantAccessibility, btnGrantStorage, btnGrantNotification;
    private TextView tvParamPath, tvBinPath, tvModelStatus;
    private TextView tvOverlayStatus, tvAccessibilityStatus, tvStorageStatus, tvNotificationStatus;
    private EditText etTargetLabel, etClickDelay;
    private SeekBar seekCaptureScale;
    private TextView tvCaptureScaleValue;

    private BroadcastReceiver startCaptureReceiver;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        settings = new SettingsManager(this);

        // Initialize NCNN
        YoloV8Ncnn.nativeInit();

        initViews();
        loadSettings();
        updatePermissionStatus();
        registerReceivers();
    }

    private void initViews() {
        // ── Confidence threshold ────────────────────────────────────────
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

        // ── NMS threshold ───────────────────────────────────────────────
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

        // ── Capture scale ───────────────────────────────────────────────
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

        // ── Target size ─────────────────────────────────────────────────
        spinnerTargetSize = findViewById(R.id.spinner_target_size);
        ArrayAdapter<String> sizeAdapter = new ArrayAdapter<>(this,
                android.R.layout.simple_spinner_dropdown_item,
                new String[]{"320", "416", "640"});
        spinnerTargetSize.setAdapter(sizeAdapter);
        spinnerTargetSize.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                int[] sizes = {320, 416, 640};
                settings.setTargetSize(sizes[position]);
            }
            @Override
            public void onNothingSelected(AdapterView<?> parent) {}
        });

        // ── Thread count ────────────────────────────────────────────────
        spinnerThreads = findViewById(R.id.spinner_threads);
        ArrayAdapter<String> threadAdapter = new ArrayAdapter<>(this,
                android.R.layout.simple_spinner_dropdown_item,
                new String[]{"1", "2", "4", "6", "8"});
        spinnerThreads.setAdapter(threadAdapter);
        spinnerThreads.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                int[] threads = {1, 2, 4, 6, 8};
                settings.setNumThreads(threads[position]);
            }
            @Override
            public void onNothingSelected(AdapterView<?> parent) {}
        });

        // ── GPU toggle ──────────────────────────────────────────────────
        cbUseGpu = findViewById(R.id.cb_use_gpu);
        cbUseGpu.setOnCheckedChangeListener((buttonView, isChecked) ->
                settings.setUseGpu(isChecked));

        // ── Model selection ─────────────────────────────────────────────
        btnSelectParam = findViewById(R.id.btn_select_param);
        btnSelectBin = findViewById(R.id.btn_select_bin);
        btnLoadModel = findViewById(R.id.btn_load_model);
        tvParamPath = findViewById(R.id.tv_param_path);
        tvBinPath = findViewById(R.id.tv_bin_path);
        tvModelStatus = findViewById(R.id.tv_model_status);

        btnSelectParam.setOnClickListener(v -> pickFile(REQUEST_PICK_PARAM));
        btnSelectBin.setOnClickListener(v -> pickFile(REQUEST_PICK_BIN));
        btnLoadModel.setOnClickListener(v -> loadModel());

        // ── Target label & click delay ──────────────────────────────────
        etTargetLabel = findViewById(R.id.et_target_label);
        etClickDelay = findViewById(R.id.et_click_delay);

        // ── Permission buttons ──────────────────────────────────────────
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

        // ── Action buttons ──────────────────────────────────────────────
        btnStartOverlay = findViewById(R.id.btn_start_overlay);
        btnStartCapture = findViewById(R.id.btn_start_capture);

        btnStartOverlay.setOnClickListener(v -> startOverlayService());
        btnStartCapture.setOnClickListener(v -> startScreenCapture());

        // ── Save settings button ────────────────────────────────────────
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

        // Target size spinner
        int targetSize = settings.getTargetSize();
        int sizeIndex = targetSize == 320 ? 0 : targetSize == 416 ? 1 : 2;
        spinnerTargetSize.setSelection(sizeIndex);

        // Thread spinner
        int threads = settings.getNumThreads();
        int threadIndex = threads <= 1 ? 0 : threads <= 2 ? 1 : threads <= 4 ? 2 : threads <= 6 ? 3 : 4;
        spinnerThreads.setSelection(threadIndex);

        cbUseGpu.setChecked(settings.getUseGpu());

        // Model paths
        String paramPath = settings.getParamPath();
        String binPath = settings.getBinPath();
        tvParamPath.setText(paramPath.isEmpty() ? "Not selected" : new File(paramPath).getName());
        tvBinPath.setText(binPath.isEmpty() ? "Not selected" : new File(binPath).getName());

        // Extra settings
        int targetLabel = settings.getTargetLabel();
        etTargetLabel.setText(targetLabel == -1 ? "-1" : String.valueOf(targetLabel));
        etClickDelay.setText(String.valueOf(settings.getClickDelay()));

        // Model status
        tvModelStatus.setText(YoloV8Ncnn.nativeIsLoaded() ? "Model: Loaded" : "Model: Not loaded");
    }

    private void saveExtraSettings() {
        try {
            String labelStr = etTargetLabel.getText().toString().trim();
            int label = Integer.parseInt(labelStr);
            settings.setTargetLabel(label);
        } catch (NumberFormatException e) {
            settings.setTargetLabel(-1);
        }

        try {
            String delayStr = etClickDelay.getText().toString().trim();
            int delay = Integer.parseInt(delayStr);
            settings.setClickDelay(Math.max(100, delay));
        } catch (NumberFormatException e) {
            settings.setClickDelay(500);
        }

        Toast.makeText(this, "Settings saved", Toast.LENGTH_SHORT).show();
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Model loading
    // ═══════════════════════════════════════════════════════════════════════

    private void loadModel() {
        String paramPath = settings.getParamPath();
        String binPath = settings.getBinPath();

        if (paramPath.isEmpty() || binPath.isEmpty()) {
            Toast.makeText(this, "Please select both .param and .bin files", Toast.LENGTH_SHORT).show();
            return;
        }

        tvModelStatus.setText("Model: Loading...");

        new Thread(() -> {
            boolean ok = YoloV8Ncnn.nativeLoadModelPath(
                    paramPath, binPath,
                    settings.getTargetSize(),
                    settings.getUseGpu(),
                    settings.getNumThreads());

            runOnUiThread(() -> {
                if (ok) {
                    tvModelStatus.setText("Model: Loaded ✓");
                    Toast.makeText(this, "Model loaded successfully!", Toast.LENGTH_SHORT).show();
                } else {
                    tvModelStatus.setText("Model: Load failed ✗");
                    Toast.makeText(this, "Failed to load model. Check file paths.", Toast.LENGTH_LONG).show();
                }
            });
        }).start();
    }

    // ═══════════════════════════════════════════════════════════════════════
    // File picker
    // ═══════════════════════════════════════════════════════════════════════

    private void pickFile(int requestCode) {
        Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
        intent.setType("*/*");
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        startActivityForResult(Intent.createChooser(intent, "Select file"), requestCode);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Permission handling
    // ═══════════════════════════════════════════════════════════════════════

    private void requestOverlayPermission() {
        if (!Settings.canDrawOverlays(this)) {
            Intent intent = new Intent(Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
                    Uri.parse("package:" + getPackageName()));
            startActivityForResult(intent, REQUEST_OVERLAY);
        }
    }

    private void requestAccessibilityPermission() {
        Intent intent = new Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS);
        startActivityForResult(intent, REQUEST_ACCESSIBILITY);
    }

    private void requestStoragePermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            if (!Environment.isExternalStorageManager()) {
                Intent intent = new Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION,
                        Uri.parse("package:" + getPackageName()));
                startActivityForResult(intent, REQUEST_MANAGE_STORAGE);
            }
        } else {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.READ_EXTERNAL_STORAGE,
                            Manifest.permission.WRITE_EXTERNAL_STORAGE},
                    REQUEST_STORAGE);
        }
    }

    private void requestNotificationPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.POST_NOTIFICATIONS},
                    REQUEST_NOTIFICATIONS);
        }
    }

    private void updatePermissionStatus() {
        // Overlay
        boolean hasOverlay = Settings.canDrawOverlays(this);
        tvOverlayStatus.setText(hasOverlay ? "✓ Granted" : "✗ Required");
        tvOverlayStatus.setTextColor(hasOverlay ? 0xFF4CAF50 : 0xFFF44336);

        // Accessibility
        boolean hasAccessibility = AutoClickService.isRunning();
        tvAccessibilityStatus.setText(hasAccessibility ? "✓ Running" : "✗ Not enabled");
        tvAccessibilityStatus.setTextColor(hasAccessibility ? 0xFF4CAF50 : 0xFFF44336);

        // Storage
        boolean hasStorage;
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            hasStorage = Environment.isExternalStorageManager();
        } else {
            hasStorage = ContextCompat.checkSelfPermission(this,
                    Manifest.permission.READ_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED;
        }
        tvStorageStatus.setText(hasStorage ? "✓ Granted" : "✗ Required");
        tvStorageStatus.setTextColor(hasStorage ? 0xFF4CAF50 : 0xFFF44336);

        // Notification
        boolean hasNotification = true;
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            hasNotification = ContextCompat.checkSelfPermission(this,
                    Manifest.permission.POST_NOTIFICATIONS) == PackageManager.PERMISSION_GRANTED;
        }
        tvNotificationStatus.setText(hasNotification ? "✓ Granted" : "✗ Required");
        tvNotificationStatus.setTextColor(hasNotification ? 0xFF4CAF50 : 0xFFF44336);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Start services
    // ═══════════════════════════════════════════════════════════════════════

    private void startOverlayService() {
        if (!Settings.canDrawOverlays(this)) {
            Toast.makeText(this, "Overlay permission required!", Toast.LENGTH_SHORT).show();
            requestOverlayPermission();
            return;
        }

        Intent intent = new Intent(this, OverlayService.class);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            startForegroundService(intent);
        } else {
            startService(intent);
        }
        Toast.makeText(this, "Overlay started", Toast.LENGTH_SHORT).show();
    }

    private void startScreenCapture() {
        if (!YoloV8Ncnn.nativeIsLoaded()) {
            Toast.makeText(this, "Please load a model first!", Toast.LENGTH_SHORT).show();
            return;
        }

        MediaProjectionManager mpManager =
                (MediaProjectionManager) getSystemService(MEDIA_PROJECTION_SERVICE);
        startActivityForResult(mpManager.createScreenCaptureIntent(), REQUEST_MEDIA_PROJECTION);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Activity results
    // ═══════════════════════════════════════════════════════════════════════

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        switch (requestCode) {
            case REQUEST_MEDIA_PROJECTION:
                if (resultCode == RESULT_OK && data != null) {
                    Intent serviceIntent = new Intent(this, ScreenCaptureService.class);
                    serviceIntent.putExtra(ScreenCaptureService.EXTRA_RESULT_CODE, resultCode);
                    serviceIntent.putExtra(ScreenCaptureService.EXTRA_DATA, data);

                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                        startForegroundService(serviceIntent);
                    } else {
                        startService(serviceIntent);
                    }

                    // Set up detection callback
                    new Thread(() -> {
                        // Wait for service to start
                        try { Thread.sleep(500); } catch (InterruptedException ignored) {}
                        ScreenCaptureService service = ScreenCaptureService.getInstance();
                        if (service != null) {
                            service.setCallback((boxes, inferTimeMs, fps, captureW, captureH) -> {
                                OverlayService overlay = OverlayService.getInstance();
                                if (overlay != null) {
                                    overlay.updateDetections(boxes, inferTimeMs, fps, captureW, captureH);
                                }
                            });
                        }
                        OverlayService overlay = OverlayService.getInstance();
                        if (overlay != null) {
                            overlay.setCapturing(true);
                        }
                    }).start();

                    Toast.makeText(this, "Screen capture started!", Toast.LENGTH_SHORT).show();
                }
                break;

            case REQUEST_PICK_PARAM:
                if (resultCode == RESULT_OK && data != null) {
                    String path = copyUriToLocal(data.getData(), "model.param");
                    if (path != null) {
                        settings.setParamPath(path);
                        tvParamPath.setText(new File(path).getName());
                    }
                }
                break;

            case REQUEST_PICK_BIN:
                if (resultCode == RESULT_OK && data != null) {
                    String path = copyUriToLocal(data.getData(), "model.bin");
                    if (path != null) {
                        settings.setBinPath(path);
                        tvBinPath.setText(new File(path).getName());
                    }
                }
                break;

            case REQUEST_OVERLAY:
            case REQUEST_ACCESSIBILITY:
            case REQUEST_MANAGE_STORAGE:
                updatePermissionStatus();
                break;
        }
    }

    /**
     * Copy a content URI to local app storage for NCNN to read.
     * NCNN needs a real file path, not a content URI.
     */
    private String copyUriToLocal(Uri uri, String fileName) {
        try {
            File outFile = new File(getFilesDir(), fileName);
            InputStream is = getContentResolver().openInputStream(uri);
            FileOutputStream fos = new FileOutputStream(outFile);
            byte[] buffer = new byte[8192];
            int len;
            while ((len = is.read(buffer)) != -1) {
                fos.write(buffer, 0, len);
            }
            fos.close();
            is.close();
            Log.i(TAG, "Copied to: " + outFile.getAbsolutePath());
            return outFile.getAbsolutePath();
        } catch (Exception e) {
            Log.e(TAG, "Failed to copy file", e);
            Toast.makeText(this, "Failed to read file: " + e.getMessage(), Toast.LENGTH_LONG).show();
            return null;
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        updatePermissionStatus();
    }

    @Override
    protected void onResume() {
        super.onResume();
        updatePermissionStatus();
    }

    @Override
    protected void onDestroy() {
        if (startCaptureReceiver != null) {
            unregisterReceiver(startCaptureReceiver);
        }
        super.onDestroy();
    }

    private void registerReceivers() {
        startCaptureReceiver = new BroadcastReceiver() {
            @Override
            public void onReceive(Context context, Intent intent) {
                startScreenCapture();
            }
        };
        IntentFilter filter = new IntentFilter("com.yolov8ncnn.START_CAPTURE");
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            registerReceiver(startCaptureReceiver, filter, Context.RECEIVER_NOT_EXPORTED);
        } else {
            registerReceiver(startCaptureReceiver, filter);
        }
    }

    // Simple SeekBar listener helper
    abstract static class SimpleSeekBarListener implements SeekBar.OnSeekBarChangeListener {
        @Override
        public void onStartTrackingTouch(SeekBar seekBar) {}
        @Override
        public void onStopTrackingTouch(SeekBar seekBar) {}
    }
}
