import os
import glob
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

# ==============================================================================
# SCRIPT: inspect_subf.py
# PURPOSE: Tahap 0 - Dataset Integrity, Statistics, and Spectral Inspection
# DATASET: SUBF v1.0 (Bearing Fault Vibration Data)
# ==============================================================================

DATA_DIR = r"C:\Users\meiza\Downloads\New ML\CSV Files"
SAMPLING_RATE = 1000.0  # Hz (From dataset technical specifications)
SHAFT_SPEED_RPM = 1440.0 # Motor rotational speed (1440 RPM)
SHAFT_FREQ_HZ = SHAFT_SPEED_RPM / 60.0 # 24.0 Hz
CLASSES = ["Normal", "Inner Race Fault", "Outer Race Fault"]

def inspect_dataset(base_dir):
    print("=" * 80)
    print("ZENTRA TinyML - TAHAP 0: DATASET INSPECTION & INTEGRITY CHECK")
    print("=" * 80)
    
    summary_data = []
    sample_signals = {}
    
    for cls in CLASSES:
        cls_path = os.path.join(base_dir, cls)
        if not os.path.exists(cls_path):
            print(f"[ERROR] Directory not found: {cls_path}")
            continue
            
        file_list = sorted(glob.glob(os.path.join(cls_path, "*.csv")))
        num_files = len(file_list)
        
        if num_files == 0:
            print(f"[WARNING] No CSV files found in {cls_path}")
            continue
            
        # Read the first file to determine sample length and column count
        df_first = pd.read_csv(file_list[0], header=None)
        samples_per_file, num_channels = df_first.shape
        duration_per_file = samples_per_file / SAMPLING_RATE
        total_duration_hours = (num_files * duration_per_file) / 3600.0
        
        # Analyze a representative batch (e.g., first 20 files) for robust statistics
        batch_size = min(20, num_files)
        mins, maxs, means, stds, rms_list, kurt_list = [], [], [], [], [], []
        clipped_samples_x = 0
        total_inspected_samples = 0
        
        for f in file_list[:batch_size]:
            data = pd.read_csv(f, header=None).values
            mins.append(data.min(axis=0))
            maxs.append(data.max(axis=0))
            means.append(data.mean(axis=0))
            stds.append(data.std(axis=0))
            rms_list.append(np.sqrt(np.mean(data**2, axis=0)))
            kurt_list.append(stats.kurtosis(data, axis=0, fisher=False)) # Pearson kurtosis (normal=3)
            
            # BeanDevice AX-3D +/- 2g range saturation check (~2.0g)
            clipped_samples_x += np.sum((data[:, 0] <= -2.01) | (data[:, 0] >= 1.97))
            total_inspected_samples += samples_per_file
            
        sample_signals[cls] = df_first.values
        
        avg_min = np.mean(mins, axis=0)
        avg_max = np.mean(maxs, axis=0)
        avg_mean = np.mean(means, axis=0)
        avg_std = np.mean(stds, axis=0)
        avg_rms = np.mean(rms_list, axis=0)
        avg_kurt = np.mean(kurt_list, axis=0)
        clip_pct_x = (clipped_samples_x / total_inspected_samples) * 100.0
        
        summary_data.append({
            "Class": cls,
            "Files": num_files,
            "Samples/File": samples_per_file,
            "Duration/File (s)": duration_per_file,
            "Total Hours": total_duration_hours,
            "X Mean (g)": avg_mean[0],
            "X Std (g)": avg_std[0],
            "X RMS (g)": avg_rms[0],
            "X Kurtosis": avg_kurt[0],
            "X Min (g)": avg_min[0],
            "X Max (g)": avg_max[0],
            "X Clip (%)": clip_pct_x
        })
        
    df_summary = pd.DataFrame(summary_data)
    print("\n--- Ringkasan Dataset per Kelas ---")
    print(df_summary[["Class", "Files", "Duration/File (s)", "Total Hours", "X RMS (g)", "X Kurtosis", "X Clip (%)"]].to_string(index=False))
    
    print("\n--- Detail Statistik Sumbu X, Y, Z (File Pertama) ---")
    for cls in CLASSES:
        if cls in sample_signals:
            sig = sample_signals[cls]
            print(f"\n[{cls}]")
            print(f"  Axis X (g): Mean={sig[:,0].mean():.3f}, Std={sig[:,0].std():.3f}, Min={sig[:,0].min():.3f}, Max={sig[:,0].max():.3f}")
            print(f"  Axis Y (g): Mean={sig[:,1].mean():.3f}, Std={sig[:,1].std():.3f}, Min={sig[:,1].min():.3f}, Max={sig[:,1].max():.3f}")
            print(f"  Axis Z (g): Mean={sig[:,2].mean():.3f}, Std={sig[:,2].std():.3f}, Min={sig[:,2].min():.3f}, Max={sig[:,2].max():.3f} (Static Gravity ~1g)")

    return sample_signals

def plot_signals_and_spectra(sample_signals, output_image="dataset_inspection_tahap0.png"):
    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    fig.suptitle(f"SUBF v1.0 Dataset Inspection (fs = {int(SAMPLING_RATE)} Hz, Motor = {int(SHAFT_SPEED_RPM)} RPM)", fontsize=14, fontweight="bold")
    
    # 2048-point FFT for clear spectral resolution (~0.488 Hz bin width)
    n_fft = 2048
    freqs = np.fft.rfftfreq(n_fft, 1.0 / SAMPLING_RATE)
    
    for idx, cls in enumerate(CLASSES):
        if cls not in sample_signals:
            continue
        sig_x = sample_signals[cls][:, 0] # Radial Axis X
        
        # 1. Time-Domain Plot (1 second window = 1000 samples)
        t = np.arange(1000) / SAMPLING_RATE
        axes[idx, 0].plot(t, sig_x[:1000], color='tab:blue' if cls=='Normal' else ('tab:orange' if 'Inner' in cls else 'tab:red'), lw=1.0)
        axes[idx, 0].axhline(2.0, color='gray', linestyle='--', alpha=0.7, label='+/-2g Clip Boundary')
        axes[idx, 0].axhline(-2.0, color='gray', linestyle='--', alpha=0.7)
        axes[idx, 0].set_title(f"{cls} - Time Domain (Axis X, 1 Second)", fontsize=11, fontweight="bold")
        axes[idx, 0].set_ylabel("Accel (g)")
        axes[idx, 0].set_ylim(-2.5, 2.5)
        axes[idx, 0].grid(True, linestyle=":", alpha=0.6)
        if idx == 0:
            axes[idx, 0].legend(loc="upper right", fontsize=8)
        
        # 2. Frequency-Domain Plot (FFT Magnitude Spectrum)
        window = np.hanning(n_fft)
        fft_raw = np.fft.rfft((sig_x[:n_fft] - np.mean(sig_x[:n_fft])) * window)
        fft_mag = (2.0 / n_fft) * np.abs(fft_raw)
        
        axes[idx, 1].plot(freqs, fft_mag, color='tab:purple', lw=1.0)
        axes[idx, 1].axvline(SHAFT_FREQ_HZ, color='green', linestyle=':', label=f'Shaft 1X ({SHAFT_FREQ_HZ:.0f} Hz)')
        axes[idx, 1].set_title(f"{cls} - FFT Amplitude Spectrum (0 - 500 Hz)", fontsize=11, fontweight="bold")
        axes[idx, 1].set_ylabel("Magnitude (g)")
        axes[idx, 1].set_xlim(0, 500)
        axes[idx, 1].grid(True, linestyle=":", alpha=0.6)
        axes[idx, 1].legend(loc="upper right", fontsize=8)

    axes[2, 0].set_xlabel("Time (seconds)")
    axes[2, 1].set_xlabel("Frequency (Hz)")
    
    plt.tight_layout()
    plt.savefig(output_image, dpi=200)
    print(f"\n[INFO] Plot berhasil disimpan ke: {os.path.abspath(output_image)}")

if __name__ == "__main__":
    signals = inspect_dataset(DATA_DIR)
    plot_signals_and_spectra(signals)
