/*
 * ZENTRA TinyML - Main Firmware Entrypoint (Tahap 11)
 * Target: ESP32-S3-DevKitC-1 N16R8 (Dual Core @ 240 MHz)
 * Architecture:
 *   - Core 0: Real-Time Sensor Acquisition Task (SPI ADXL345 @ 3200 Hz & FIFO Windowing)
 *   - Core 1: TinyML Inference Task (esp-dsp FFT Feature Extraction, PCA Scoring, & Hysteresis)
 */

#include <Arduino.h>
#include "zentra_inference.h"
#include "adxl345_spi.h"
#include "feature_extractor.h"
#include "hysteresis.h"
#include "pca_baseline.h"
#include "parity_test.h"
#include "golden_test_vector.h"

// ==============================================================================
// 1. DATA STRUCTURES & INTER-CORE QUEUES
// ==============================================================================
typedef struct {
    float x[FEATURE_WINDOW_SIZE];
    float y[FEATURE_WINDOW_SIZE];
    float z[FEATURE_WINDOW_SIZE];
    uint32_t window_id;
    bool is_simulated;
} window_data_t;

// Double buffer in fast Internal SRAM (2 x 6.1 KB = ~12.2 KB)
static window_data_t s_window_buffers[2];
static QueueHandle_t s_window_queue = NULL;
static zentra_hysteresis_t s_bearing1_hysteresis;
static bool s_sensor_connected = false;

// LED Blinking parameters for Status Indication
static volatile zentra_status_t s_current_machine_status = ZENTRA_STATUS_NORMAL;

// ==============================================================================
// 2. CORE 0: SENSOR ACQUISITION TASK (SPI & WINDOWING)
// ==============================================================================
void task_sensor_acquisition(void *pvParameters) {
    Serial.printf("[CORE 0] Sensor Acquisition Task running on Core %d (Priority %d)\n", 
                    xPortGetCoreID(), uxTaskPriorityGet(NULL));

    // Check if sensor is responding
    s_sensor_connected = adxl345_init(PIN_SPI_CS_BRG1, ADXL345_ODR_3200HZ);
    if (!s_sensor_connected) {
        Serial.println("[CORE 0] ADXL345 not detected on hardware pins. Switching to Bench-Test Simulation Mode.");
    }
    
    static float x_ring[FEATURE_WINDOW_SIZE];
    static float y_ring[FEATURE_WINDOW_SIZE];
    static float z_ring[FEATURE_WINDOW_SIZE];
    
    uint16_t ring_idx = 0;
    uint16_t new_samples_count = 0;
    uint32_t window_seq = 0;
    uint8_t buf_toggle = 0;
    
    // Sample period at 3200 Hz is 312.5 microseconds. 
    // In polling mode for bench test or FIFO reading:
    TickType_t last_wake_time = xTaskGetTickCount();
    
    for (;;) {
        float x_g = 0.0f, y_g = 0.0f, z_g = 0.0f;
        
        if (s_sensor_connected) {
            // Read hardware sample from ADXL345 via SPI
            adxl345_read_xyz(PIN_SPI_CS_BRG1, &x_g, &y_g, &z_g);
        } else {
            // Replay real vibration samples from dataset: Normal (windows 0-24) -> Fault (windows 25-29)
            bool inject_fault = ((window_seq % 30) >= 25);
            if (inject_fault) {
                x_g = GOLDEN_FAULT_RAW_X[ring_idx];
                y_g = GOLDEN_FAULT_RAW_Y[ring_idx];
                z_g = GOLDEN_FAULT_RAW_Z[ring_idx];
            } else {
                x_g = GOLDEN_NORMAL_RAW_X[ring_idx];
                y_g = GOLDEN_NORMAL_RAW_Y[ring_idx];
                z_g = GOLDEN_NORMAL_RAW_Z[ring_idx];
            }
        }
        
        // Store in ring buffer
        x_ring[ring_idx] = x_g;
        y_ring[ring_idx] = y_g;
        z_ring[ring_idx] = z_g;
        
        ring_idx = (ring_idx + 1) % FEATURE_WINDOW_SIZE;
        new_samples_count++;
        
        // 50% overlap: trigger new window every 256 new samples
        if (new_samples_count >= 256) {
            new_samples_count = 0;
            window_seq++;
            
            // Unroll ring buffer into contiguous double-buffer slot
            window_data_t* p_dest = &s_window_buffers[buf_toggle];
            buf_toggle ^= 1; // Toggle between buffer 0 and 1
            
            for (int i = 0; i < FEATURE_WINDOW_SIZE; i++) {
                int src_idx = (ring_idx + i) % FEATURE_WINDOW_SIZE;
                p_dest->x[i] = x_ring[src_idx];
                p_dest->y[i] = y_ring[src_idx];
                p_dest->z[i] = z_ring[src_idx];
            }
            p_dest->window_id = window_seq;
            p_dest->is_simulated = !s_sensor_connected;
            
            // Dispatch window pointer to Core 1 without blocking Core 0
            if (s_window_queue != NULL) {
                xQueueSend(s_window_queue, &p_dest, 0);
            }
        }
        
        // At 3200 Hz in real sensor mode, timer or interrupt paces acquisition.
        // For simulation bench test, yield briefly:
        vTaskDelay(pdMS_TO_TICKS(1));
    }
}

// ==============================================================================
// 3. CORE 1: TINYML INFERENCE & FEATURE EXTRACTION TASK
// ==============================================================================
void task_tinyml_inference(void *pvParameters) {
    Serial.printf("[CORE 1] TinyML Inference Task running on Core %d (Priority %d)\n", 
                  xPortGetCoreID(), uxTaskPriorityGet(NULL));
                  
    feature_extractor_init();
    hysteresis_init(&s_bearing1_hysteresis, PCA_THRESHOLD_ALERT, PCA_THRESHOLD_DANGER);
    
    static float raw_features[FEATURE_TOTAL_COUNT];
    static float scaled_features[FEATURE_TOTAL_COUNT];
    window_data_t* p_win = NULL;
    
    for (;;) {
        // Block until Core 0 dispatches a new 512-sample window
        if (xQueueReceive(s_window_queue, &p_win, portMAX_DELAY) == pdTRUE) {
            int64_t t0 = esp_timer_get_time();
            
            // 1. Ekstraksi 60 Fitur menggunakan esp-dsp FFT
            feature_extractor_compute(p_win->x, p_win->y, p_win->z, raw_features);
            int64_t t1 = esp_timer_get_time();
            
            // 2. Standardisasi Fitur 1-cycle multiply
            zentra_standardize_features(raw_features, scaled_features);
            int64_t t2 = esp_timer_get_time();
            
            // 3. Inferensi PCA Rekonstruksi (Engine 1)
            zentra_result_t res = zentra_infer_pca(scaled_features);
            int64_t t3 = esp_timer_get_time();
            
            // 4. Update Mesin Status dengan Histeresis
            zentra_status_t debounced_status = hysteresis_update(&s_bearing1_hysteresis, res.anomaly_score);
            s_current_machine_status = debounced_status;
            
            // Print telemetry setiap 5 jendela (~1.2 detik)
            if (p_win->window_id % 5 == 0) {
                uint32_t t_dsp_us   = (uint32_t)(t1 - t0);
                uint32_t t_scale_us = (uint32_t)(t2 - t1);
                uint32_t t_inf_us   = (uint32_t)(t3 - t2);
                
                const char* status_str = (debounced_status == ZENTRA_STATUS_NORMAL) ? "NORMAL" :
                                         ((debounced_status == ZENTRA_STATUS_ALERT)  ? "ALERT"  : "DANGER");
                                         
                Serial.printf("\n[ZENTRA WIN #%u] %s | Raw Score: %.2f | Status: %s\n",
                              p_win->window_id, 
                              p_win->is_simulated ? "(SIM)" : "(SPI)",
                              res.anomaly_score, 
                              status_str);
                Serial.printf("  -> Extracted RMS [X,Y,Z] : [%.3f, %.3f, %.3f] g\n", 
                              raw_features[0], raw_features[20], raw_features[40]);
                Serial.printf("  -> Kurtosis [X,Y,Z]      : [%.2f, %.2f, %.2f]\n", 
                              raw_features[1], raw_features[21], raw_features[41]);
                Serial.printf("  -> Timing (us)           : DSP=%u | Scale=%u | PCA=%u (Total=%u us)\n", 
                              t_dsp_us, t_scale_us, t_inf_us, (uint32_t)(t3 - t0));
            }
        }
    }
}

// ==============================================================================
// 4. SETUP & ARDUINO MAIN LOOP
// ==============================================================================
void setup() {
    Serial.begin(115200);
    delay(2000);
    
    pinMode(PIN_STATUS_LED, OUTPUT);
    digitalWrite(PIN_STATUS_LED, LOW);
    
    Serial.println("\n========================================================");
    Serial.println("  ZENTRA CNC SPINDLE BEARING ANOMALY MONITORING SYSTEM  ");
    Serial.println("  Tahap 11: FreeRTOS Dual-Core SPI & DSP Feature Engine ");
    Serial.println("========================================================");
    
    // Inisialisasi inference subsystem
    zentra_inference_init();
    
    // Jalankan Uji Paritas Numerik Python-C (Tahap 12)
    run_firmware_parity_test();
    
    // Buat FreeRTOS Queue untuk komunikasi antar core (kapasitas 2 pointer)
    s_window_queue = xQueueCreate(2, sizeof(window_data_t*));
    if (s_window_queue == NULL) {
        Serial.println("[CRITICAL ERROR] Failed to create FreeRTOS Window Queue!");
        return;
    }
    
    // Core 0: Task Akuisisi Sensor SPI (Prioritas Tinggi: 10, Stack: 4096 byte)
    xTaskCreatePinnedToCore(
        task_sensor_acquisition,
        "Task_Sensor_SPI",
        4096,
        NULL,
        10,
        NULL,
        0
    );
    
    // Core 1: Task Ekstraksi Fitur & Inferensi TinyML (Prioritas: 5, Stack: 8192 byte)
    xTaskCreatePinnedToCore(
        task_tinyml_inference,
        "Task_TinyML_Infer",
        8192,
        NULL,
        5,
        NULL,
        1
    );
    
    Serial.println("[ZENTRA] Dual-Core FreeRTOS Subsystem successfully running!\n");
}

void loop() {
    // LED Heartbeat & Visual Status Indicator
    static uint32_t last_toggle = 0;
    static bool led_state = false;
    uint32_t interval_ms = 1000; // Normal: Kedip lambat (1 Hz)
    
    if (s_current_machine_status == ZENTRA_STATUS_ALERT) {
        interval_ms = 200; // Alert: Kedip cepat (5 Hz)
    } else if (s_current_machine_status == ZENTRA_STATUS_DANGER) {
        interval_ms = 50;  // Danger: Kedip strobo sangat cepat (20 Hz)
    }
    
    if (millis() - last_toggle >= interval_ms) {
        last_toggle = millis();
        led_state = !led_state;
        digitalWrite(PIN_STATUS_LED, led_state ? HIGH : LOW);
    }
    
    delay(20);
}
