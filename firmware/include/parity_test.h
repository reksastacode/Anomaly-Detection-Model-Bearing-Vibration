/*
 * ZENTRA TinyML - Embedded Parity Validation Header (Tahap 12)
 * Target: ESP32-S3-DevKitC-1 N16R8
 */

#ifndef PARITY_TEST_H_
#define PARITY_TEST_H_

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Parity Tolerance Thresholds (IEEE 754 float32 precision)
#define PARITY_MAX_TIME_DOMAIN_TOLERANCE   1e-4f   // Max allowable diff for RMS, Kurtosis, Crest
#define PARITY_MAX_FFT_BAND_TOLERANCE      0.05f   // Max allowable diff for log-energy FFT bands (dB)
#define PARITY_MAX_SCORE_TOLERANCE         0.10f   // Max allowable anomaly score deviation

/**
 * Runs bit-exact numerical parity test comparing ESP32-S3 firmware outputs 
 * against Python reference outputs for both Normal and Defect test windows.
 * Prints detailed comparative report and timing to Serial.
 * Returns true if all features and scores are within allowable tolerances.
 */
bool run_firmware_parity_test(void);

#ifdef __cplusplus
}
#endif

#endif // PARITY_TEST_H_
