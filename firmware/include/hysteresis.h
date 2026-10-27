/*
 * ZENTRA TinyML - Anomaly State Machine with Hysteresis (Tahap 11)
 * Target: ESP32-S3-DevKitC-1 N16R8
 * Debounces transient vibrations and prevents alarm chattering.
 */

#ifndef HYSTERESIS_H_
#define HYSTERESIS_H_

#include "zentra_inference.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Hysteresis consecutive window counts
#define HYST_ALERT_TRIGGER_COUNT    3  // 3 consecutive windows above Alert threshold to enter ALERT
#define HYST_DANGER_TRIGGER_COUNT   2  // 2 consecutive windows above Danger threshold to enter DANGER
#define HYST_RECOVERY_COUNT         5  // 5 consecutive healthy windows to de-escalate

typedef struct {
    zentra_status_t current_status;
    float alert_threshold;
    float danger_threshold;
    uint8_t alert_streak;
    uint8_t danger_streak;
    uint8_t healthy_streak;
    uint32_t total_windows_processed;
    uint32_t alert_events_count;
    uint32_t danger_events_count;
} zentra_hysteresis_t;

/**
 * Initializes hysteresis state machine with calibrated thresholds.
 */
void hysteresis_init(zentra_hysteresis_t* hyst, float alert_th, float danger_th);

/**
 * Updates hysteresis state machine with a new window anomaly score.
 * Returns the debounced machine health status.
 */
zentra_status_t hysteresis_update(zentra_hysteresis_t* hyst, float anomaly_score);

#ifdef __cplusplus
}
#endif

#endif // HYSTERESIS_H_
