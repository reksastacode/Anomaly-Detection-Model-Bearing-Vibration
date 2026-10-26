"""
ZENTRA TinyML - Feature Extraction Module (Tahap 3)
Extracts 60 vibration features per window (20 per axis: X, Y, Z) with exact 
numerical parity designed for 1:1 reproduction in embedded C on ESP32-S3.

Feature Vector Layout (per axis):
  - [0]      : RMS (Root Mean Square)
  - [1]      : Pearson Kurtosis
  - [2]      : Crest Factor
  - [3..18]  : 16 Log-Scaled FFT Band Energies (5.8 Hz to 500.0 Hz)
  - [19]     : Envelope AC Energy (Log-scaled)
"""

import json
from pathlib import Path
import numpy as np
import scipy.signal as signal

from src.config import (
    DATA_SCALER_DIR,
    WINDOW_SIZE,
    FS_SUBF
)

# 17 bin edges forming 16 contiguous, non-overlapping log-spaced frequency bands
# Bin resolution df = 1000 Hz / 512 = 1.953125 Hz
# Bin 3 = ~5.86 Hz, Bin 256 = 500.0 Hz (Nyquist)
BAND_EDGES_BINS = np.array(
    [3, 4, 6, 8, 11, 14, 19, 26, 34, 46, 61, 81, 108, 144, 192, 224, 257],
    dtype=np.int32
)
NUM_BANDS = 16
NUM_FEATURES_PER_AXIS = 20
TOTAL_FEATURES = NUM_FEATURES_PER_AXIS * 3  # 60 features

def get_feature_names():
    """Generates human-readable names for all 60 features."""
    names = []
    for axis in ["X", "Y", "Z"]:
        names.append(f"{axis}_RMS")
        names.append(f"{axis}_Kurtosis")
        names.append(f"{axis}_CrestFactor")
        for b in range(NUM_BANDS):
            f_low = BAND_EDGES_BINS[b] * (FS_SUBF / WINDOW_SIZE)
            f_high = (BAND_EDGES_BINS[b + 1] - 1) * (FS_SUBF / WINDOW_SIZE)
            names.append(f"{axis}_FFT_Band_{b:02d}_{f_low:.1f}-{f_high:.1f}Hz")
        names.append(f"{axis}_EnvelopeEnergy")
    return names

def extract_features_single_window(window_512x3):
    """
    Reference single-window implementation.
    Mirrors the exact sequential algorithm executed in firmware C on ESP32-S3.
    
    Args:
        window_512x3: np.ndarray of shape (512, 3), dtype float32.
    Returns:
        np.ndarray of shape (60,), dtype float32.
    """
    N, C = window_512x3.shape
    features = []
    
    # Precomputed Hann window: w[n] = 0.5 * (1.0 - cos(2*pi*n / N))
    hann = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(N, dtype=np.float32) / N))
    
    for ch in range(C):
        x = window_512x3[:, ch]
        
        # 1. Time-Domain Features
        m2 = np.mean(x ** 2)
        m4 = np.mean(x ** 4)
        rms = np.sqrt(m2)
        kurtosis = m4 / (m2 ** 2 + 1e-9)
        crest_factor = np.max(np.abs(x)) / (rms + 1e-9)
        features.extend([rms, kurtosis, crest_factor])
        
        # 2. Log-Scaled FFT Band Energy
        # Apply Hann window, then real FFT
        x_win = x * hann
        fft_c = np.fft.rfft(x_win)
        mag_sq = np.abs(fft_c) ** 2
        
        for b in range(NUM_BANDS):
            b_start = BAND_EDGES_BINS[b]
            b_end = BAND_EDGES_BINS[b + 1]
            # Average power in band b
            band_energy = np.mean(mag_sq[b_start:b_end])
            log_energy = np.log10(band_energy + 1e-7)
            features.append(log_energy)
            
        # 3. Envelope AC Energy
        # Analytic signal envelope via Hilbert transform
        env = np.abs(signal.hilbert(x))
        env_ac = env - np.mean(env)
        env_energy = np.log10(np.mean(env_ac ** 2) + 1e-7)
        features.append(env_energy)
        
    return np.array(features, dtype=np.float32)

def extract_features_batch(windows_tensor):
    """
    Fast vectorized batch extraction for dataset processing.
    
    Args:
        windows_tensor: np.ndarray of shape (N_windows, 512, 3), dtype float32.
    Returns:
        np.ndarray of shape (N_windows, 60), dtype float32.
    """
    n_windows, n_samples, n_channels = windows_tensor.shape
    
    # 1. Time-Domain Vectorized
    m2 = np.mean(windows_tensor ** 2, axis=1) # (N, 3)
    m4 = np.mean(windows_tensor ** 4, axis=1) # (N, 3)
    rms = np.sqrt(m2)                          # (N, 3)
    kurt = m4 / (m2 ** 2 + 1e-9)               # (N, 3)
    crest = np.max(np.abs(windows_tensor), axis=1) / (rms + 1e-9) # (N, 3)
    
    # 2. Frequency-Domain Vectorized
    hann = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n_samples) / n_samples))
    hann = hann.reshape(1, n_samples, 1).astype(np.float32)
    
    fft_c = np.fft.rfft(windows_tensor * hann, axis=1) # (N, 257, 3)
    mag_sq = np.abs(fft_c) ** 2
    
    band_energies = []
    for b in range(NUM_BANDS):
        b_s = BAND_EDGES_BINS[b]
        b_e = BAND_EDGES_BINS[b + 1]
        be = np.mean(mag_sq[:, b_s:b_e, :], axis=1) # (N, 3)
        band_energies.append(np.log10(be + 1e-7))
    band_energies = np.stack(band_energies, axis=-1) # (N, 3, 16)
    
    # 3. Envelope Vectorized
    env = np.abs(signal.hilbert(windows_tensor, axis=1)) # (N, 512, 3)
    env_ac = env - np.mean(env, axis=1, keepdims=True)
    env_energy = np.log10(np.mean(env_ac ** 2, axis=1) + 1e-7) # (N, 3)
    
    # Assemble into (N, 60) per-channel layout
    channel_blocks = []
    for c in range(n_channels):
        ch_features = np.column_stack([
            rms[:, c],
            kurt[:, c],
            crest[:, c],
            band_energies[:, c, :],
            env_energy[:, c]
        ])
        channel_blocks.append(ch_features)
        
    return np.concatenate(channel_blocks, axis=1).astype(np.float32)

def fit_scaler(train_features):
    """
    Fits Z-score standardization parameters ONLY on healthy baseline training data.
    Computes mean vector and inverse standard deviation vector for 1-cycle multiply in C.
    
    Returns:
        dict: containing 'mean', 'std', and 'scale' vectors.
    """
    mean = np.mean(train_features, axis=0)
    std = np.std(train_features, axis=0)
    # Prevent division by zero
    scale = 1.0 / (std + 1e-7)
    
    return {
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "scale": scale.astype(np.float32),
        "num_features": int(train_features.shape[1])
    }

def apply_scaler(features, scaler_dict):
    """Standardizes features: z = (x - mean) * scale."""
    return (features - scaler_dict["mean"]) * scaler_dict["scale"]

def export_scaler(scaler_dict, json_path=None, c_header_path=None):
    """
    Exports normalization parameters to:
    1. JSON for Python deployment.
    2. C Header (.h) for direct compilation into ESP32-S3 firmware.
    """
    DATA_SCALER_DIR.mkdir(parents=True, exist_ok=True)
    
    if json_path is None:
        json_path = DATA_SCALER_DIR / "scaler_params.json"
    if c_header_path is None:
        c_header_path = DATA_SCALER_DIR / "scaler_params.h"
        
    # 1. Save JSON
    serializable = {
        "num_features": scaler_dict["num_features"],
        "mean": scaler_dict["mean"].tolist(),
        "std": scaler_dict["std"].tolist(),
        "scale": scaler_dict["scale"].tolist()
    }
    with open(json_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"[OK] Scaler parameters saved to JSON: {json_path}")
    
    # 2. Save C Header
    mean_str = ", ".join([f"{v:.7e}f" for v in scaler_dict["mean"]])
    scale_str = ", ".join([f"{v:.7e}f" for v in scaler_dict["scale"]])
    
    header_content = f"""/*
 * ZENTRA TinyML - Normalization Parameters Header (Auto-Generated)
 * Embedded C parity header for ESP32-S3 firmware.
 * Fast standardization: z[i] = (x[i] - SCALER_MEAN[i]) * SCALER_SCALE[i];
 */

#ifndef ZENTRA_SCALER_PARAMS_H_
#define ZENTRA_SCALER_PARAMS_H_

#define ZENTRA_NUM_FEATURES {scaler_dict['num_features']}

static const float SCALER_MEAN[ZENTRA_NUM_FEATURES] = {{
    {mean_str}
}};

static const float SCALER_SCALE[ZENTRA_NUM_FEATURES] = {{
    {scale_str}
}};

#endif // ZENTRA_SCALER_PARAMS_H_
"""
    with open(c_header_path, "w") as f:
        f.write(header_content)
    print(f"[OK] Scaler parameters saved to C Header: {c_header_path}")
