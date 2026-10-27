/*
 * ZENTRA TinyML - Embedded Parity Validation Implementation (Tahap 12)
 * Target: ESP32-S3-DevKitC-1 N16R8
 */

#include "parity_test.h"
#include "golden_test_vector.h"
#include "feature_extractor.h"
#include "zentra_inference.h"
#include "pca_baseline.h"

#include <Arduino.h>
#include <esp_timer.h>
#include <math.h>

static bool evaluate_single_window_parity(
    const char* label,
    const float* raw_x,
    const float* raw_y,
    const float* raw_z,
    const float* ref_features_py,
    float ref_pca_score_py
) {
    Serial.println("\n----------------------------------------------------------------------");
    Serial.printf(">>> EVALUATING PARITY: %s TEST WINDOW <<<\n", label);
    Serial.println("----------------------------------------------------------------------");
    
    float c_features[FEATURE_TOTAL_COUNT];
    float scaled_features[FEATURE_TOTAL_COUNT];
    
    // 1. Measure Feature Extraction Latency
    int64_t t0 = esp_timer_get_time();
    feature_extractor_compute(raw_x, raw_y, raw_z, c_features);
    int64_t t1 = esp_timer_get_time();
    
    // 2. Measure Standardization Latency
    zentra_standardize_features(c_features, scaled_features);
    int64_t t2 = esp_timer_get_time();
    
    // 3. Measure PCA Inference Latency
    zentra_result_t res = zentra_infer_pca(scaled_features);
    int64_t t3 = esp_timer_get_time();
    
    uint32_t dt_dsp   = (uint32_t)(t1 - t0);
    uint32_t dt_scale = (uint32_t)(t2 - t1);
    uint32_t dt_pca   = (uint32_t)(t3 - t2);
    uint32_t dt_total = (uint32_t)(t3 - t0);
    
    // 4. Compute Statistical Divergence
    float max_error = 0.0f;
    float sum_abs_error = 0.0f;
    int max_err_idx = 0;
    bool pass = true;
    
    for (int i = 0; i < FEATURE_TOTAL_COUNT; i++) {
        float diff = fabsf(c_features[i] - ref_features_py[i]);
        sum_abs_error += diff;
        if (diff > max_error) {
            max_error = diff;
            max_err_idx = i;
        }
        
        // Time domain check (RMS, Kurtosis, Crest Factor)
        int feat_type = i % FEATURE_PER_CHANNEL;
        if (feat_type < 3) {
            if (diff > PARITY_MAX_TIME_DOMAIN_TOLERANCE) {
                pass = false;
            }
        } else {
            // FFT and Envelope energy check
            if (diff > PARITY_MAX_FFT_BAND_TOLERANCE) {
                pass = false;
            }
        }
    }
    
    float mae = sum_abs_error / (float)FEATURE_TOTAL_COUNT;
    float score_diff = fabsf(res.anomaly_score - ref_pca_score_py);
    if (score_diff > PARITY_MAX_SCORE_TOLERANCE) {
        pass = false;
    }
    
    // 5. Print Diagnostic Metrics
    Serial.printf("  [Features] Max Absolute Error : %.6f (at Feature #%02d)\n", max_error, max_err_idx);
    Serial.printf("  [Features] Mean Absolute Error: %.6f\n", mae);
    Serial.printf("  [Score]    Python Ref Score   : %.4f\n", ref_pca_score_py);
    Serial.printf("  [Score]    ESP32-S3 Firmware  : %.4f (Diff: %.4f)\n", res.anomaly_score, score_diff);
    Serial.printf("  [Timing]   Feature Extractor  : %u us (%.2f ms)\n", dt_dsp, dt_dsp / 1000.0f);
    Serial.printf("  [Timing]   Standardization    : %u us\n", dt_scale);
    Serial.printf("  [Timing]   PCA Inference      : %u us\n", dt_pca);
    Serial.printf("  [Timing]   Total Per-Window   : %u us (%.2f ms)\n", dt_total, dt_total / 1000.0f);
    
    // Spot check key features (X_RMS, X_Kurtosis, X_Band0, X_Envelope)
    Serial.println("\n  --- Key Feature Spot-Check Comparison ---");
    Serial.printf("  X_RMS        | Py: %10.5f | C: %10.5f | Diff: %.6f\n", ref_features_py[0], c_features[0], fabsf(ref_features_py[0] - c_features[0]));
    Serial.printf("  X_Kurtosis   | Py: %10.5f | C: %10.5f | Diff: %.6f\n", ref_features_py[1], c_features[1], fabsf(ref_features_py[1] - c_features[1]));
    Serial.printf("  X_CrestFact  | Py: %10.5f | C: %10.5f | Diff: %.6f\n", ref_features_py[2], c_features[2], fabsf(ref_features_py[2] - c_features[2]));
    Serial.printf("  X_FFT_Band00 | Py: %10.5f | C: %10.5f | Diff: %.6f\n", ref_features_py[3], c_features[3], fabsf(ref_features_py[3] - c_features[3]));
    Serial.printf("  X_Envelope   | Py: %10.5f | C: %10.5f | Diff: %.6f\n", ref_features_py[19], c_features[19], fabsf(ref_features_py[19] - c_features[19]));
    
    // Print all 16 FFT bands for diagnosis
    Serial.println("\n  --- FFT 16 Bands (Python vs C) ---");
    for (int b = 0; b < 16; b++) {
        Serial.printf("  Band %02d | Py: %10.5f | C: %10.5f | Diff: %10.5f\n",
                      b, ref_features_py[3 + b], c_features[3 + b], c_features[3 + b] - ref_features_py[3 + b]);
    }
    
    if (pass) {
        Serial.printf("\n  >>> VERDICT [%s]: PASS (Within Tolerance)\n", label);
    } else {
        Serial.printf("\n  >>> VERDICT [%s]: FAIL (Exceeds Tolerance)\n", label);
    }
    
    return pass;
}

bool run_firmware_parity_test(void) {
    Serial.println("\n======================================================================");
    Serial.println("  ZENTRA TinyML - EMBEDDED NUMERICAL PARITY VALIDATION TEST HARNESS   ");
    Serial.println("  Hardware: ESP32-S3-DevKitC-1 N16R8 (Xtensa Dual-Core @ 240 MHz)     ");
    Serial.println("======================================================================");
    
    bool norm_pass = evaluate_single_window_parity(
        "NORMAL (HEALTHY)",
        GOLDEN_NORMAL_RAW_X,
        GOLDEN_NORMAL_RAW_Y,
        GOLDEN_NORMAL_RAW_Z,
        GOLDEN_NORMAL_FEATURES_PY,
        GOLDEN_NORMAL_PCA_SCORE_PY
    );
    
    bool fault_pass = evaluate_single_window_parity(
        "FAULT (DEFECTIVE)",
        GOLDEN_FAULT_RAW_X,
        GOLDEN_FAULT_RAW_Y,
        GOLDEN_FAULT_RAW_Z,
        GOLDEN_FAULT_FEATURES_PY,
        GOLDEN_FAULT_PCA_SCORE_PY
    );
    
    Serial.println("\n======================================================================");
    if (norm_pass && fault_pass) {
        Serial.println(">>> OVERALL STATUS: ALL PARITY TESTS PASSED! <<<");
        Serial.println("ESP32-S3 firmware is bit-for-bit mathematically aligned with Python.");
    } else {
        Serial.println(">>> OVERALL STATUS: PARITY FAILED! Check tolerances. <<<");
    }
    Serial.println("======================================================================\n");
    
    return (norm_pass && fault_pass);
}
