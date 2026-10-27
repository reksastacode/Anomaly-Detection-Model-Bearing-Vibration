/*
 * ZENTRA TinyML - Embedded Feature Extractor Header (Tahap 11)
 * Target: ESP32-S3-DevKitC-1 N16R8
 * Computes 60 features per window (20 per axis: X, Y, Z) with bit-exact parity to Python.
 */

#ifndef FEATURE_EXTRACTOR_H_
#define FEATURE_EXTRACTOR_H_

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FEATURE_WINDOW_SIZE        512
#define FEATURE_NUM_CHANNELS       3
#define FEATURE_BANDS_COUNT        16
#define FEATURE_PER_CHANNEL        20
#define FEATURE_TOTAL_COUNT        60

/**
 * Initializes DSP tables for FFT and precomputes Hann window.
 * Must be called once before feature extraction.
 */
bool feature_extractor_init(void);

/**
 * Extracts 60 vibration features from a 512x3 window buffer.
 * window_x, window_y, window_z: Arrays of 512 float samples each (units: 'g').
 * features_out: Output buffer of size 60 float elements.
 */
void feature_extractor_compute(const float* window_x, 
                               const float* window_y, 
                               const float* window_z, 
                               float* features_out);

#ifdef __cplusplus
}
#endif

#endif // FEATURE_EXTRACTOR_H_
