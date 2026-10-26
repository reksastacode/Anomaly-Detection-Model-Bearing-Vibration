"""
ZENTRA TinyML Configuration Module
Defines global paths, acquisition parameters, feature extraction bounds, 
and MCU target constraints.
"""

import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "CSV Files"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATA_SCALER_DIR = PROJECT_ROOT / "data" / "scaler"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_BASELINE_DIR = MODELS_DIR / "baseline"
MODELS_SAVED_DIR = MODELS_DIR / "saved_models"
MODELS_TFLITE_DIR = MODELS_DIR / "tflite"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Signal Acquisition Parameters
FS_SUBF = 1000.0           # Hz (SUBF v1.0 dataset sampling frequency)
FS_TARGET = 3200.0         # Hz (ADXL345 SPI maximum output data rate)
SHAFT_RPM_SUBF = 1440.0    # RPM (Motor test bench speed)
SHAFT_FREQ_SUBF = SHAFT_RPM_SUBF / 60.0  # 24.0 Hz

# Signal Segmentation (Power-of-2 for radix-2 FFT on esp-dsp)
WINDOW_SIZE = 512          # 512 samples (~0.512s at 1000 Hz, ~12.3 shaft cycles)
WINDOW_STRIDE = 256        # 50% overlap (256 samples stride)

# Classes
CLASSES = ["Normal", "Inner Race Fault", "Outer Race Fault"]

# Target Embedded Constraints (ESP32-S3 DevKitC-1 N16R8)
MAX_TENSOR_ARENA_BYTES = 100 * 1024   # 100 KB target arena limit in SRAM
MAX_MODEL_PARAMS = 50_000             # Parameter upper bound
MAX_INFERENCE_LATENCY_MS = 50.0       # Latency target per window on 240 MHz core
INT8_QUANTIZATION = True              # Strict int8 quantization for TFLM

# Reproducibility
RANDOM_SEED = 42
