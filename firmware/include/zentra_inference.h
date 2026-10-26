/*
 * ZENTRA TinyML - Embedded Inference Header (Tahap 10)
 * Target: ESP32-S3-DevKitC-1 N16R8
 * Defines system status enums, inference structs, pin definitions, and prototypes.
 */

#ifndef ZENTRA_INFERENCE_H_
#define ZENTRA_INFERENCE_H_

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ==============================================================================
// CRITICAL HARDWARE PIN DEFINITIONS (ESP32-S3 N16R8)
// ==============================================================================
// WARNING: NEVER USE THE FOLLOWING PINS:
//   - GPIO 33, 34, 35, 36, 37 : Connected to 8MB Octal PSRAM (OPI)!
//   - GPIO 26, 27, 28, 29, 30, 31, 32 : Connected to SPI Flash!
//   - GPIO 0, 3, 45, 46 : Strapping & Boot Configuration pins!

// SAFE SPI PIN MAPPING FOR ADXL345:
#define PIN_SPI_SCK       12  // Safe GPIO for SPI Clock
#define PIN_SPI_MISO      13  // Safe GPIO for SPI Master-In-Slave-Out
#define PIN_SPI_MOSI      11  // Safe GPIO for SPI Master-Out-Slave-In
#define PIN_SPI_CS_BRG1   10  // Chip Select for Bearing 1 Sensor
#define PIN_SPI_CS_BRG2    9  // Chip Select for Bearing 2 Sensor
#define PIN_ADXL_INT1     14  // Data-Ready Interrupt Input

// STATUS INDICATOR LED (DevKitC-1 Built-in RGB or external LED)
#define PIN_STATUS_LED     2  // Safe GPIO for status indication

// ==============================================================================
// ANOMALY STATUS & RESULT STRUCTURES
// ==============================================================================
typedef enum {
    ZENTRA_STATUS_NORMAL = 0,
    ZENTRA_STATUS_ALERT  = 1,
    ZENTRA_STATUS_DANGER = 2
} zentra_status_t;

typedef struct {
    float anomaly_score;
    zentra_status_t status;
    uint32_t inference_time_us;
} zentra_result_t;

// ==============================================================================
// PUBLIC INFERENCE & NORMALIZATION API
// ==============================================================================

/**
 * Initializes inference engines and prints hardware memory status.
 * Returns true if initialization succeeded.
 */
bool zentra_inference_init(void);

/**
 * Standardizes 60 raw features in 1 cycle/feature using precomputed scaler constants.
 * z[i] = (raw[i] - SCALER_MEAN[i]) * SCALER_SCALE[i];
 */
void zentra_standardize_features(const float* raw_features, float* scaled_features_out);

/**
 * Executes ultra-lightweight PCA Subspace Reconstruction Error inference.
 * Latency: < 5 microseconds | RAM: 260 bytes on stack.
 */
zentra_result_t zentra_infer_pca(const float* scaled_features);

/**
 * Executes TensorFlow Lite Micro (TFLM) Autoencoder inference.
 * Latency: ~350 - 500 microseconds | RAM: ~10 KB Tensor Arena in SRAM.
 */
zentra_result_t zentra_infer_tflm(const float* scaled_features);

#ifdef __cplusplus
}
#endif

#endif // ZENTRA_INFERENCE_H_
