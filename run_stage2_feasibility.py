"""
ZENTRA TinyML - Tahap 2: ADXL345 Bandwidth & Sensor Feasibility Experiment
Evaluates the impact of bandwidth limitations on bearing fault discriminability.
Compares full-bandwidth (5-500 Hz) vs restricted-bandwidth (5-200 Hz) signals
and determines whether MEMS accelerometers (like ADXL345) are sufficient for CNC spindles.
"""

import os
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.signal as signal
import matplotlib.pyplot as plt

from src.config import (
    DATA_RAW_DIR,
    REPORTS_DIR,
    CLASSES,
    FS_SUBF,
    SHAFT_FREQ_SUBF
)
from src.preprocessing import (
    build_highpass_filter,
    preprocess_signal,
    segment_windows
)

def compute_fisher_discriminant_ratio(feat_norm, feat_fault):
    """
    Computes Fisher's Discriminant Ratio (FDR) between Normal and Fault feature distributions.
    Higher FDR indicates superior class separability without overlap.
    FDR = (mean_fault - mean_norm)^2 / (var_fault + var_norm)
    """
    mu_norm, var_norm = np.mean(feat_norm), np.var(feat_norm)
    mu_fault, var_fault = np.mean(feat_fault), np.var(feat_fault)
    
    denominator = var_norm + var_fault
    if denominator < 1e-9:
        return 0.0
    return float((mu_fault - mu_norm) ** 2 / denominator)

def run_feasibility_experiment(num_files_per_class=40):
    print("=" * 80)
    print("ZENTRA TinyML - TAHAP 2: ADXL345 SENSOR & BANDWIDTH FEASIBILITY EXPERIMENT")
    print("=" * 80)
    
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Digital Filters
    # Filter A: High-Pass (5 Hz) -> Full 5-500 Hz bandwidth (covers ADXL345 3200 Hz ODR)
    sos_full = build_highpass_filter(cutoff_hz=5.0, fs=FS_SUBF, order=4)
    # Filter B: Low-Pass (200 Hz) on top of HPF -> Restricted 5-200 Hz bandwidth (simulating low ODR/bandwidth)
    sos_restricted = signal.butter(4, 200.0, btype='lowpass', fs=FS_SUBF, output='sos')
    
    features_full = {cls: [] for cls in CLASSES}
    features_restricted = {cls: [] for cls in CLASSES}
    ratio_high_low = {cls: [] for cls in CLASSES}
    
    sample_windows_for_plot = {}
    
    for cls in CLASSES:
        cls_dir = DATA_RAW_DIR / cls
        files = sorted(glob.glob(str(cls_dir / "*.csv")))[:num_files_per_class]
        
        for fpath in files:
            raw_data = pd.read_csv(fpath, header=None).values.astype(np.float32)
            
            # Full bandwidth signal (DC removed)
            sig_full = preprocess_signal(raw_data, sos_filter=sos_full)
            # Restricted bandwidth signal (5 - 200 Hz)
            sig_rest = signal.sosfiltfilt(sos_restricted, sig_full, axis=0)
            
            # Segment into 512-point windows
            win_full = segment_windows(sig_full, window_size=512, stride=256)
            win_rest = segment_windows(sig_rest, window_size=512, stride=256)
            
            if cls not in sample_windows_for_plot:
                sample_windows_for_plot[cls] = (win_full[0, :, 0], win_rest[0, :, 0])
                
            # Compute Radial (Axis X) RMS per window
            rms_f = np.sqrt(np.mean(win_full[:, :, 0] ** 2, axis=1))
            rms_r = np.sqrt(np.mean(win_rest[:, :, 0] ** 2, axis=1))
            
            features_full[cls].extend(rms_f)
            features_restricted[cls].extend(rms_r)
            
            # Compute Spectral Energy Ratio (High-band 200-500 Hz / Low-band 5-200 Hz)
            fft_mag = np.abs(np.fft.rfft(win_full[:, :, 0], axis=1))
            freqs = np.fft.rfftfreq(512, 1.0 / FS_SUBF)
            
            low_mask = (freqs >= 5) & (freqs < 200)
            high_mask = (freqs >= 200) & (freqs < 500)
            
            e_low = np.sum(fft_mag[:, low_mask] ** 2, axis=1) + 1e-6
            e_high = np.sum(fft_mag[:, high_mask] ** 2, axis=1)
            ratio_high_low[cls].extend(e_high / e_low)

    for cls in CLASSES:
        features_full[cls] = np.array(features_full[cls])
        features_restricted[cls] = np.array(features_restricted[cls])
        ratio_high_low[cls] = np.array(ratio_high_low[cls])

    # 2. Print Comparative Metrics
    print("\n[1] METRIK PEMISAHAN RMS KELAS (Mean +/- Std):")
    print("-" * 80)
    print(f"{'Class':<20} | {'Full Band (5-500 Hz) RMS':<25} | {'Restricted (5-200 Hz) RMS':<25}")
    print("-" * 80)
    for cls in CLASSES:
        mf, sf = np.mean(features_full[cls]), np.std(features_full[cls])
        mr, sr = np.mean(features_restricted[cls]), np.std(features_restricted[cls])
        print(f"{cls:<20} | {mf:.4f} +/- {sf:.4f}{'':<11} | {mr:.4f} +/- {sr:.4f}")
    print("-" * 80)

    # 3. Fisher's Discriminant Ratio (FDR) Evaluation
    fdr_full_ir = compute_fisher_discriminant_ratio(features_full["Normal"], features_full["Inner Race Fault"])
    fdr_rest_ir = compute_fisher_discriminant_ratio(features_restricted["Normal"], features_restricted["Inner Race Fault"])
    
    fdr_full_or = compute_fisher_discriminant_ratio(features_full["Normal"], features_full["Outer Race Fault"])
    fdr_rest_or = compute_fisher_discriminant_ratio(features_restricted["Normal"], features_restricted["Outer Race Fault"])
    
    print("\n[2] FISHER'S DISCRIMINANT RATIO (FDR - Semakin tinggi, semakin mudah dipisahkan):")
    print(f"  Inner Race vs Normal -> Full Band FDR: {fdr_full_ir:.3f} | Restricted (<=200Hz) FDR: {fdr_rest_ir:.4f}")
    print(f"  Outer Race vs Normal -> Full Band FDR: {fdr_full_or:.3f} | Restricted (<=200Hz) FDR: {fdr_rest_or:.4f}")
    
    # 4. Generate Diagnostic Visualization
    fig, axes = plt.subplots(3, 2, figsize=(15, 11))
    fig.suptitle("Tahap 2: ADXL345 Sensor Feasibility & Bandwidth Evaluation", fontsize=14, fontweight="bold")
    
    colors = {"Normal": "tab:blue", "Inner Race Fault": "tab:orange", "Outer Race Fault": "tab:red"}
    t = np.arange(512) / FS_SUBF
    
    for i, cls in enumerate(CLASSES):
        sig_f, sig_r = sample_windows_for_plot[cls]
        
        # Left column: Waveforms Full vs Restricted
        axes[i, 0].plot(t, sig_f, label="Full Band (5-500 Hz)", color=colors[cls], lw=1.2)
        axes[i, 0].plot(t, sig_r, label="Restricted (5-200 Hz)", color="black", linestyle="--", alpha=0.7, lw=1.0)
        axes[i, 0].set_title(f"{cls} - Window Waveform Comparison", fontsize=11, fontweight="bold")
        axes[i, 0].set_ylabel("Accel (g)")
        axes[i, 0].grid(True, linestyle=":", alpha=0.6)
        axes[i, 0].legend(loc="upper right", fontsize=8)
        
        # Right column: RMS Distribution Boxplot Comparison
        data_to_plot = [features_full[cls], features_restricted[cls]]
        bp = axes[i, 1].boxplot(data_to_plot, patch_artist=True, tick_labels=["Full (5-500 Hz)", "Restricted (5-200 Hz)"])
        bp['boxes'][0].set_facecolor(colors[cls])
        bp['boxes'][1].set_facecolor("gray")
        axes[i, 1].set_title(f"{cls} - RMS Distribution Shift", fontsize=11, fontweight="bold")
        axes[i, 1].set_ylabel("RMS (g)")
        axes[i, 1].grid(True, linestyle=":", alpha=0.6)

    axes[2, 0].set_xlabel("Time (seconds)")
    
    plt.tight_layout()
    out_plot = REPORTS_DIR / "stage2_adxl345_feasibility.png"
    plt.savefig(out_plot, dpi=200)
    print(f"\n[INFO] Diagnostic plot saved to: {out_plot.resolve()}")
    
    return {
        "fdr_full_ir": fdr_full_ir,
        "fdr_rest_ir": fdr_rest_ir,
        "fdr_full_or": fdr_full_or,
        "fdr_rest_or": fdr_rest_or
    }

if __name__ == "__main__":
    run_feasibility_experiment()
