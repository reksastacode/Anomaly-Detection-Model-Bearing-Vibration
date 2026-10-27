/*
 * ZENTRA TinyML - Anomaly State Machine with Hysteresis Implementation (Tahap 11)
 * Target: ESP32-S3-DevKitC-1 N16R8
 */

#include "hysteresis.h"
#include <Arduino.h>

void hysteresis_init(zentra_hysteresis_t* hyst, float alert_th, float danger_th) {
    hyst->current_status = ZENTRA_STATUS_NORMAL;
    hyst->alert_threshold = alert_th;
    hyst->danger_threshold = danger_th;
    hyst->alert_streak = 0;
    hyst->danger_streak = 0;
    hyst->healthy_streak = 0;
    hyst->total_windows_processed = 0;
    hyst->alert_events_count = 0;
    hyst->danger_events_count = 0;
}

zentra_status_t hysteresis_update(zentra_hysteresis_t* hyst, float anomaly_score) {
    hyst->total_windows_processed++;
    
    // Evaluate instantaneous status candidate
    bool is_danger_level = (anomaly_score >= hyst->danger_threshold);
    bool is_alert_level  = (anomaly_score >= hyst->alert_threshold);
    
    if (is_danger_level) {
        hyst->danger_streak++;
        hyst->alert_streak++;
        hyst->healthy_streak = 0;
    } else if (is_alert_level) {
        hyst->danger_streak = 0;
        hyst->alert_streak++;
        hyst->healthy_streak = 0;
    } else {
        hyst->danger_streak = 0;
        hyst->alert_streak = 0;
        hyst->healthy_streak++;
    }
    
    // State transitions based on hysteresis streaks
    switch (hyst->current_status) {
        case ZENTRA_STATUS_NORMAL:
            if (hyst->danger_streak >= HYST_DANGER_TRIGGER_COUNT) {
                hyst->current_status = ZENTRA_STATUS_DANGER;
                hyst->danger_events_count++;
                Serial.printf("\n[HYSTERESIS] State Escalation: NORMAL -> DANGER! (Score: %.2f)\n", anomaly_score);
            } else if (hyst->alert_streak >= HYST_ALERT_TRIGGER_COUNT) {
                hyst->current_status = ZENTRA_STATUS_ALERT;
                hyst->alert_events_count++;
                Serial.printf("\n[HYSTERESIS] State Escalation: NORMAL -> ALERT! (Score: %.2f)\n", anomaly_score);
            }
            break;
            
        case ZENTRA_STATUS_ALERT:
            if (hyst->danger_streak >= HYST_DANGER_TRIGGER_COUNT) {
                hyst->current_status = ZENTRA_STATUS_DANGER;
                hyst->danger_events_count++;
                Serial.printf("\n[HYSTERESIS] State Escalation: ALERT -> DANGER! (Score: %.2f)\n", anomaly_score);
            } else if (hyst->healthy_streak >= HYST_RECOVERY_COUNT) {
                hyst->current_status = ZENTRA_STATUS_NORMAL;
                Serial.printf("\n[HYSTERESIS] State Recovery: ALERT -> NORMAL. (Consecutive healthy: %d)\n", 
                              hyst->healthy_streak);
            }
            break;
            
        case ZENTRA_STATUS_DANGER:
            if (hyst->healthy_streak >= HYST_RECOVERY_COUNT * 2) {
                hyst->current_status = ZENTRA_STATUS_NORMAL;
                Serial.printf("\n[HYSTERESIS] Full State Recovery: DANGER -> NORMAL.\n");
            } else if (hyst->healthy_streak >= HYST_RECOVERY_COUNT) {
                hyst->current_status = ZENTRA_STATUS_ALERT;
                Serial.printf("\n[HYSTERESIS] Partial State De-escalation: DANGER -> ALERT.\n");
            }
            break;
    }
    
    return hyst->current_status;
}
