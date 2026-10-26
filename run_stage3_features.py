"""
ZENTRA TinyML - Tahap 3: Feature Extraction & Parity Verification Script
Extracts 60 vibration features across classes, verifies single vs batch parity,
fits normalization scaler exclusively on healthy baseline data, and exports
scaler parameters to JSON and C Header format.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED_DIR, DATA_SCALER_DIR
from src.features import (
    extract_features_single_window,
    extract_features_batch,
    get_feature_names,
    fit_scaler,
    apply_scaler,
    export_scaler,
    TOTAL_FEATURES
)

def run_feature_pipeline():
    print("=" * 80)
    print("ZENTRA TinyML - TAHAP 3: FEATURE EXTRACTION & EMBEDDED C PARITY CHECK")
    print("=" * 80)
    
    # 1. Load Preprocessed Windows
    classes = ["normal", "inner_race_fault", "outer_race_fault"]
    tensors = {}
    
    for cls in classes:
        npz_file = DATA_PROCESSED_DIR / f"{cls}_windows.npz"
        if not npz_file.exists():
            raise FileNotFoundError(f"Missing preprocessed tensor: {npz_file}. Run 'python -m src.preprocessing' first.")
        data = np.load(npz_file)["data"]
        tensors[cls] = data
        print(f"[*] Loaded {cls:<20}: {data.shape[0]} windows, shape = {data.shape}")

    # 2. Verify Single-Window vs Batch Parity
    print("\n[1] VERIFIKASI PARITAS NUMERIK (Single-Window Reference vs Vectorized Batch):")
    sample_batch = tensors["normal"][:10]  # First 10 windows
    
    # Reference sequential computation (matches C firmware implementation)
    feat_single_list = [extract_features_single_window(sample_batch[i]) for i in range(10)]
    feat_single = np.array(feat_single_list)
    
    # Batch vectorized computation
    feat_batch = extract_features_batch(sample_batch)
    
    max_discrepancy = np.max(np.abs(feat_single - feat_batch))
    print(f"    --> Total Features per Window    : {TOTAL_FEATURES}")
    print(f"    --> Max Absolute Discrepancy     : {max_discrepancy:.3e}")
    if max_discrepancy < 1e-4:
        print("    --> [STATUS: PASS] Paritas numerik identik (dalam batas toleransi FP32).")
    else:
        print("    --> [STATUS: FAIL] Terdeteksi perbedaan komputasi yang melebihi batas FP32.")

    # 3. Extract Features Across All Classes
    print("\n[2] EKSTRAKSI FITUR KESELURUHAN KELAS:")
    features_dict = {}
    for cls in classes:
        feats = extract_features_batch(tensors[cls])
        features_dict[cls] = feats
        print(f"    --> {cls:<20} : {feats.shape[0]} samples x {feats.shape[1]} features")

    # 4. Fit Scaler ONLY on Normal (Healthy) Data
    # Rule 4: Normalization statistics must never be contaminated by anomaly/defect data!
    print("\n[3] PENYETELAN SKALER NORMALISASI (HANYA PADA DATA SEHAT):")
    scaler = fit_scaler(features_dict["normal"])
    export_scaler(scaler)

    # 5. Apply Normalization and Inspect Separation
    norm_scaled = apply_scaler(features_dict["normal"], scaler)
    ir_scaled = apply_scaler(features_dict["inner_race_fault"], scaler)
    or_scaled = apply_scaler(features_dict["outer_race_fault"], scaler)
    
    feature_names = get_feature_names()
    
    # Print sample comparison for key features
    print("\n[4] PERBANDINGAN NILAI FITUR SEBELUM & SETELAH NORMALISASI (Mean +/- Std):")
    print("-" * 85)
    print(f"{'Feature Name':<30} | {'Normal (Raw)':<16} | {'Normal (Scaled)':<16} | {'Inner Race (Scaled)':<16}")
    print("-" * 85)
    
    sample_indices = [0, 1, 2, 6, 12, 19] # RMS_X, Kurt_X, Crest_X, Band3_X, Band9_X, Env_X
    for idx in sample_indices:
        fname = feature_names[idx]
        raw_norm_m = np.mean(features_dict["normal"][:, idx])
        raw_norm_s = np.std(features_dict["normal"][:, idx])
        sc_norm_m = np.mean(norm_scaled[:, idx])
        sc_norm_s = np.std(norm_scaled[:, idx])
        sc_ir_m = np.mean(ir_scaled[:, idx])
        sc_ir_s = np.std(ir_scaled[:, idx])
        
        print(f"{fname:<30} | {raw_norm_m:.3f} +/- {raw_norm_s:.3f} | {sc_norm_m:.2f} +/- {sc_norm_s:.2f}  | {sc_ir_m:.2f} +/- {sc_ir_s:.2f}")
    print("-" * 85)
    
    # 6. Save Extracted Features for Stage 4-8
    out_features_file = DATA_PROCESSED_DIR / "features_extracted.npz"
    np.savez_compressed(
        out_features_file,
        X_normal=features_dict["normal"],
        X_inner_race=features_dict["inner_race_fault"],
        X_outer_race=features_dict["outer_race_fault"],
        feature_names=feature_names
    )
    print(f"\n[INFO] Matriks fitur berhasil disimpan ke: {out_features_file.resolve()}")

if __name__ == "__main__":
    run_feature_pipeline()
