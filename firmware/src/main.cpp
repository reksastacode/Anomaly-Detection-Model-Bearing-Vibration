/*
 * ZENTRA TinyML - Main Firmware Entrypoint (Tahap 10)
 * Target: ESP32-S3-DevKitC-1 N16R8
 * Demonstrates system boot, memory diagnostics, standardization parity,
 * and high-speed TinyML anomaly scoring on mock sensor vectors.
 */

#include <Arduino.h>
#include "zentra_inference.h"

// ==============================================================================
// TEST VECTORS FOR VERIFICATION
// ==============================================================================
// 1. Mock Healthy (Normal) Vibration Features (Values around 0 before scaling)
static const float MOCK_NORMAL_RAW[60] = {
    0.635f, 2.316f, 2.622f, -0.484f, -0.160f, -0.245f, -0.101f, 1.641f,
    0.542f, 1.190f, 1.299f, 3.058f, 1.677f, 1.161f, 1.779f, 1.317f,
    0.915f, 1.072f, 1.285f, -0.979f,
    // Y-axis mock normal
    0.301f, 2.867f, 3.084f, -1.118f, -0.842f, -0.827f, -0.734f, 0.893f,
    0.031f, 0.458f, 0.491f, 1.631f, 1.652f, 0.434f, 0.830f, 1.581f,
    1.251f, 0.833f, 0.872f, -1.427f,
    // Z-axis mock normal
    0.274f, 2.845f, 3.025f, -1.166f, -0.926f, -0.926f, -0.864f, 0.541f,
    -0.192f, 0.161f, 0.212f, 0.656f, 1.852f, 0.822f, 1.502f, 1.318f,
    0.907f, 0.438f, 0.307f, -1.515f
};

// 2. Mock Defective (Fault) Vibration Features (High RMS & Envelope Energy)
static const float MOCK_FAULT_RAW[60] = {
    1.290f, 1.880f, 1.550f, 1.200f, 1.450f, 1.600f, 1.800f, 3.500f,
    2.800f, 3.200f, 3.400f, 5.100f, 3.800f, 3.200f, 3.500f, 3.100f,
    2.900f, 2.800f, 3.000f, 0.850f,
    // Y-axis mock fault
    0.850f, 1.950f, 1.620f, 0.500f, 0.800f, 0.900f, 1.100f, 2.200f,
    1.800f, 2.100f, 2.300f, 3.400f, 3.100f, 2.200f, 2.400f, 2.800f,
    2.600f, 2.200f, 2.100f, 0.250f,
    // Z-axis mock fault
    0.650f, 2.100f, 1.750f, 0.200f, 0.400f, 0.500f, 0.700f, 1.900f,
    1.500f, 1.700f, 1.900f, 2.800f, 2.600f, 1.900f, 2.100f, 2.500f,
    2.200f, 1.800f, 1.700f, 0.100f
};

void run_smoke_test(void) {
    Serial.println("\n" + String('=', 70));
    Serial.println(">>> RUNNING EMBEDDED INFERENCE SMOKE TEST <<<");
    Serial.println(String('=', 70));
    
    float scaled_features[60];
    
    // --- TEST 1: HEALTHY MOCK SAMPLE ---
    Serial.println("\n[TEST 1] Ingesting Mock Healthy Window...");
    zentra_standardize_features(MOCK_NORMAL_RAW, scaled_features);
    
    zentra_result_t res_norm = zentra_infer_pca(scaled_features);
    Serial.printf("  --> Raw X_RMS               : %.3f g\n", MOCK_NORMAL_RAW[0]);
    Serial.printf("  --> Scaled X_RMS            : %.2f sigma\n", scaled_features[0]);
    Serial.printf("  --> Anomaly Score           : %.2f\n", res_norm.anomaly_score);
    Serial.printf("  --> Machine Health Status   : %s\n", 
                  res_norm.status == ZENTRA_STATUS_NORMAL ? "NORMAL (Healthy)" : 
                  (res_norm.status == ZENTRA_STATUS_ALERT ? "ALERT (Warning)" : "DANGER (Fault)"));
    Serial.printf("  --> Inference Execution Time: %u microseconds (0.00%u ms)\n", 
                  res_norm.inference_time_us, res_norm.inference_time_us / 1000);

    // --- TEST 2: DEFECTIVE MOCK SAMPLE ---
    Serial.println("\n[TEST 2] Ingesting Mock Fault Window...");
    zentra_standardize_features(MOCK_FAULT_RAW, scaled_features);
    
    zentra_result_t res_fault = zentra_infer_pca(scaled_features);
    Serial.printf("  --> Raw X_RMS               : %.3f g\n", MOCK_FAULT_RAW[0]);
    Serial.printf("  --> Scaled X_RMS            : %.2f sigma\n", scaled_features[0]);
    Serial.printf("  --> Anomaly Score           : %.2f\n", res_fault.anomaly_score);
    Serial.printf("  --> Machine Health Status   : %s\n", 
                  res_fault.status == ZENTRA_STATUS_NORMAL ? "NORMAL (Healthy)" : 
                  (res_fault.status == ZENTRA_STATUS_ALERT ? "ALERT (Warning)" : "DANGER (Fault)"));
    Serial.printf("  --> Inference Execution Time: %u microseconds (0.00%u ms)\n", 
                  res_fault.inference_time_us, res_fault.inference_time_us / 1000);

    Serial.println("\n" + String('=', 70));
    if (res_norm.status == ZENTRA_STATUS_NORMAL && res_fault.status == ZENTRA_STATUS_DANGER) {
        Serial.println("[STATUS: PASS] Firmware logic successfully discriminates Normal vs Fault!");
    } else {
        Serial.println("[STATUS: CHECK] Status threshold output requires fine-tuning.");
    }
    Serial.println(String('=', 70));
}

void setup() {
    Serial.begin(115200);
    delay(2000); // Allow USB CDC enumeration on ESP32-S3
    
    pinMode(PIN_STATUS_LED, OUTPUT);
    digitalWrite(PIN_STATUS_LED, LOW);
    
    Serial.println("\n========================================================");
    Serial.println("  ZENTRA CNC SPINDLE BEARING ANOMALY MONITORING SYSTEM  ");
    Serial.println("  Core: ESP32-S3-DevKitC-1 N16R8 (Dual Core @ 240 MHz)  ");
    Serial.println("========================================================");
    
    zentra_inference_init();
    run_smoke_test();
}

void loop() {
    // Heartbeat blink every 1 second
    static uint32_t last_blink = 0;
    static bool led_state = false;
    
    if (millis() - last_blink >= 1000) {
        last_blink = millis();
        led_state = !led_state;
        digitalWrite(PIN_STATUS_LED, led_state ? HIGH : LOW);
    }
    
    delay(10);
}
