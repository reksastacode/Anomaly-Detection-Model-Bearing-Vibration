"""
ZENTRA TinyML - Preprocessing & Windowing Module (Tahap 2)
Handles DC removal, anti-aliasing / bandpass filtering, window segmentation,
and batch conversion of raw vibration CSV files into fast NumPy binary tensors.
"""

import os
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.signal as signal

from src.config import (
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    CLASSES,
    FS_SUBF,
    WINDOW_SIZE,
    WINDOW_STRIDE
)

def build_highpass_filter(cutoff_hz=5.0, fs=FS_SUBF, order=4):
    """
    Constructs a 4th-order Butterworth High-Pass Filter in Second-Order Sections (SOS).
    SOS format guarantees numerical stability in 32-bit floating point.
    Eliminates static gravity (+1g on Z axis) and low-frequency mounting drift.
    """
    sos = signal.butter(order, cutoff_hz, btype='highpass', fs=fs, output='sos')
    return sos

def build_bandpass_filter(lowcut=5.0, highcut=490.0, fs=FS_SUBF, order=4):
    """
    Constructs a 4th-order Butterworth Band-Pass Filter (SOS format).
    Removes low-frequency baseline drift and attenuates near-Nyquist noise.
    """
    sos = signal.butter(order, [lowcut, highcut], btype='bandpass', fs=fs, output='sos')
    return sos

def preprocess_signal(raw_signal, sos_filter=None):
    """
    Applies DC detrending and zero-phase SOS digital filtering across 3 axes.
    Input shape: (N_samples, 3)
    Output shape: (N_samples, 3)
    """
    # 1. Zero-mean detrending per axis
    detrended = raw_signal - np.mean(raw_signal, axis=0, keepdims=True)
    
    # 2. Apply digital filter if provided
    if sos_filter is not None:
        filtered = signal.sosfiltfilt(sos_filter, detrended, axis=0)
        return filtered.astype(np.float32)
    
    return detrended.astype(np.float32)

def segment_windows(continuous_signal, window_size=WINDOW_SIZE, stride=WINDOW_STRIDE):
    """
    Segments continuous multi-channel signal into overlapping rectangular windows.
    Input shape: (N_samples, Channels)
    Output shape: (N_windows, window_size, Channels)
    """
    n_samples, n_channels = continuous_signal.shape
    if n_samples < window_size:
        return np.empty((0, window_size, n_channels), dtype=np.float32)
        
    num_windows = (n_samples - window_size) // stride + 1
    
    # Efficient strided view using numpy
    shape = (num_windows, window_size, n_channels)
    strides = (
        continuous_signal.strides[0] * stride,
        continuous_signal.strides[0],
        continuous_signal.strides[1]
    )
    windows = np.lib.stride_tricks.as_strided(continuous_signal, shape=shape, strides=strides)
    return windows.copy()

def process_and_cache_dataset(max_files_per_class=100, save_output=True):
    """
    Batch processes raw CSVs, filters them, segments into windows,
    and caches them as compressed .npz archives for fast model training.
    """
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    sos_hpf = build_highpass_filter(cutoff_hz=5.0, fs=FS_SUBF, order=4)
    
    dataset_summary = {}
    
    for cls in CLASSES:
        cls_dir = DATA_RAW_DIR / cls
        file_list = sorted(glob.glob(str(cls_dir / "*.csv")))
        if max_files_per_class is not None:
            file_list = file_list[:max_files_per_class]
            
        print(f"[*] Processing class '{cls}' ({len(file_list)} files)...")
        all_windows = []
        
        for fpath in file_list:
            df = pd.read_csv(fpath, header=None)
            raw_data = df.values.astype(np.float32)
            
            # Apply DC & gravity removal
            clean_data = preprocess_signal(raw_data, sos_filter=sos_hpf)
            
            # Segment into windows
            windows = segment_windows(clean_data, window_size=WINDOW_SIZE, stride=WINDOW_STRIDE)
            all_windows.append(windows)
            
        if all_windows:
            class_tensor = np.concatenate(all_windows, axis=0) # (Total_Windows, 512, 3)
            dataset_summary[cls] = class_tensor.shape
            print(f"    --> Generated tensor shape: {class_tensor.shape}")
            
            if save_output:
                out_file = DATA_PROCESSED_DIR / f"{cls.lower().replace(' ', '_')}_windows.npz"
                np.savez_compressed(out_file, data=class_tensor, fs=FS_SUBF, window_size=WINDOW_SIZE)
                print(f"    --> Saved to: {out_file.name}")
                
    return dataset_summary

if __name__ == "__main__":
    process_and_cache_dataset(max_files_per_class=50)
