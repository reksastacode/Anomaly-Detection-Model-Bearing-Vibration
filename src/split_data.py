"""
ZENTRA TinyML - Data Splitting Module (Tahap 4)
Implements temporal block splitting (chrono-split) to prevent data leakage.
Normal data is split chronologically (70% Train, 15% Val, 15% Test).
Anomaly/defect data is strictly excluded from Training (0% Train, 50% Val, 50% Test).
Scalers are refitted EXCLUSIVELY on the 70% Train Normal subset.
"""

from pathlib import Path
import numpy as np

from src.config import DATA_PROCESSED_DIR, DATA_SCALER_DIR
from src.features import fit_scaler, apply_scaler, export_scaler

def perform_temporal_split():
    print("=" * 80)
    print("ZENTRA TinyML - TAHAP 4: TEMPORAL BLOCK SPLITTING (LEAK-FREE)")
    print("=" * 80)
    
    # 1. Load Extracted Features
    feat_file = DATA_PROCESSED_DIR / "features_extracted.npz"
    if not feat_file.exists():
        raise FileNotFoundError(f"Missing {feat_file}. Run 'python run_stage3_features.py' first.")
        
    data = np.load(feat_file)
    X_norm = data["X_normal"]
    X_ir = data["X_inner_race"]
    X_or = data["X_outer_race"]
    feature_names = data["feature_names"]
    
    n_norm = len(X_norm)
    n_ir = len(X_ir)
    n_or = len(X_or)
    
    print(f"[*] Total Raw Windows Loaded:")
    print(f"    - Normal            : {n_norm} windows")
    print(f"    - Inner Race Fault  : {n_ir} windows")
    print(f"    - Outer Race Fault  : {n_or} windows")

    # 2. Chronological Split Indices
    # Normal: 70% Train, 15% Val, 15% Test (Sequential in time)
    n_train_norm = int(0.70 * n_norm)
    n_val_norm = int(0.15 * n_norm)
    n_test_norm = n_norm - n_train_norm - n_val_norm
    
    X_train_norm_raw = X_norm[:n_train_norm]
    X_val_norm_raw = X_norm[n_train_norm : n_train_norm + n_val_norm]
    X_test_norm_raw = X_norm[n_train_norm + n_val_norm :]
    
    # Anomaly Classes: 0% Train, 50% Val, 50% Test
    n_val_ir = int(0.50 * n_ir)
    X_val_ir_raw = X_ir[:n_val_ir]
    X_test_ir_raw = X_ir[n_val_ir:]
    
    n_val_or = int(0.50 * n_or)
    X_val_or_raw = X_or[:n_val_or]
    X_test_or_raw = X_or[n_val_or:]

    # 3. Fit Scaler ONLY on X_train_norm_raw
    # Absolute rule: No validation or test data must leak into the scaler!
    print("\n[1] PENYETELAN SKALER NORMALISASI:")
    print(f"    --> Fitting StandardScaler strictly on {len(X_train_norm_raw)} Train Normal samples...")
    scaler = fit_scaler(X_train_norm_raw)
    export_scaler(scaler)

    # 4. Transform All Subsets Using Train-Fitted Scaler
    X_train_scaled = apply_scaler(X_train_norm_raw, scaler)
    y_train = np.zeros(len(X_train_scaled), dtype=np.int32) # Normal = 0
    train_classes = np.array(["normal"] * len(X_train_scaled))

    # Assemble Validation Set
    X_val_scaled = np.vstack([
        apply_scaler(X_val_norm_raw, scaler),
        apply_scaler(X_val_ir_raw, scaler),
        apply_scaler(X_val_or_raw, scaler)
    ])
    y_val = np.concatenate([
        np.zeros(len(X_val_norm_raw), dtype=np.int32), # Normal = 0
        np.ones(len(X_val_ir_raw), dtype=np.int32),    # Anomaly = 1
        np.ones(len(X_val_or_raw), dtype=np.int32)     # Anomaly = 1
    ])
    val_classes = np.concatenate([
        np.array(["normal"] * len(X_val_norm_raw)),
        np.array(["inner_race"] * len(X_val_ir_raw)),
        np.array(["outer_race"] * len(X_val_or_raw))
    ])

    # Assemble Test Set
    X_test_scaled = np.vstack([
        apply_scaler(X_test_norm_raw, scaler),
        apply_scaler(X_test_ir_raw, scaler),
        apply_scaler(X_test_or_raw, scaler)
    ])
    y_test = np.concatenate([
        np.zeros(len(X_test_norm_raw), dtype=np.int32), # Normal = 0
        np.ones(len(X_test_ir_raw), dtype=np.int32),    # Anomaly = 1
        np.ones(len(X_test_or_raw), dtype=np.int32)     # Anomaly = 1
    ])
    test_classes = np.concatenate([
        np.array(["normal"] * len(X_test_norm_raw)),
        np.array(["inner_race"] * len(X_test_ir_raw)),
        np.array(["outer_race"] * len(X_test_or_raw))
    ])

    # 5. Display Split Summary Table
    print("\n[2] RINGKASAN PEMBAGIAN DATASET (TEMPORAL CHRONO-SPLIT):")
    print("-" * 80)
    print(f"{'Subset':<12} | {'Normal (0)':<12} | {'Inner Race (1)':<16} | {'Outer Race (1)':<16} | {'Total':<10}")
    print("-" * 80)
    print(f"{'Train':<12} | {len(X_train_norm_raw):<12} | {0:<16} | {0:<16} | {len(X_train_scaled):<10}")
    print(f"{'Validation':<12} | {len(X_val_norm_raw):<12} | {len(X_val_ir_raw):<16} | {len(X_val_or_raw):<16} | {len(X_val_scaled):<10}")
    print(f"{'Test':<12} | {len(X_test_norm_raw):<12} | {len(X_test_ir_raw):<16} | {len(X_test_or_raw):<16} | {len(X_test_scaled):<10}")
    print("-" * 80)

    # 6. Save Partitioned Data
    out_split_file = DATA_PROCESSED_DIR / "split_data.npz"
    np.savez_compressed(
        out_split_file,
        X_train=X_train_scaled,
        y_train=y_train,
        train_classes=train_classes,
        X_val=X_val_scaled,
        y_val=y_val,
        val_classes=val_classes,
        X_test=X_test_scaled,
        y_test=y_test,
        test_classes=test_classes,
        feature_names=feature_names
    )
    print(f"\n[INFO] Data split berhasil disimpan ke: {out_split_file.resolve()}")

if __name__ == "__main__":
    perform_temporal_split()
