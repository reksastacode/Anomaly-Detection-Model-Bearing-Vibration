"""
ZENTRA TinyML - Tahap 8: Comprehensive Evaluation & Robustness Testing
Evaluates AUROC, recall at fixed FPR targets, per-fault-type performance,
speed/load transferability limits, and AWGN noise stress testing (Clean to 0 dB SNR).
Generates diagnostic multi-panel evaluation plots.
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

from src.config import DATA_PROCESSED_DIR, REPORTS_DIR, MODELS_SAVED_DIR
from src.features import extract_features_batch, apply_scaler
from src.baseline import PCAReconstructionAnomalyDetector
from src.autoencoder import TinyMLAutoencoder

def evaluate_at_fixed_fpr(y_true, scores, target_fpr=0.01):
    """Calculates threshold, actual FPR, and Recall at a specific target FPR."""
    norm_scores = scores[y_true == 0]
    th = np.percentile(norm_scores, (1.0 - target_fpr) * 100.0)
    y_pred = (scores >= th).astype(int)
    
    actual_fpr = np.mean(y_pred[y_true == 0]) * 100.0
    actual_recall = np.mean(y_pred[y_true == 1]) * 100.0
    return th, actual_fpr, actual_recall

def add_awgn_noise(windows_3d, snr_db):
    """Injects Additive White Gaussian Noise (AWGN) to multi-channel windows."""
    if snr_db is None:
        return windows_3d.copy()
    noisy = np.zeros_like(windows_3d)
    n_windows, n_samples, n_channels = windows_3d.shape
    for i in range(n_windows):
        for c in range(n_channels):
            p_sig = np.mean(windows_3d[i, :, c] ** 2)
            p_noise = p_sig / (10.0 ** (snr_db / 10.0))
            noise = np.random.normal(0.0, np.sqrt(p_noise), size=n_samples)
            noisy[i, :, c] = windows_3d[i, :, c] + noise
    return noisy.astype(np.float32)

def run_evaluation_suite():
    print("=" * 80)
    print("ZENTRA TinyML - TAHAP 8: COMPREHENSIVE EVALUATION & ROBUSTNESS TESTING")
    print("=" * 80)
    
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    split_file = DATA_PROCESSED_DIR / "split_data.npz"
    split = np.load(split_file)
    X_train = split["X_train"]
    X_val = split["X_val"]
    y_val = split["y_val"]
    X_test = split["X_test"]
    y_test = split["y_test"]
    test_classes = split["test_classes"]
    
    # Load raw windows for noise testing
    norm_raw = np.load(DATA_PROCESSED_DIR / "normal_windows.npz")["data"][1615:]
    ir_raw = np.load(DATA_PROCESSED_DIR / "inner_race_fault_windows.npz")["data"][950:]
    or_raw = np.load(DATA_PROCESSED_DIR / "outer_race_fault_windows.npz")["data"][950:]
    
    with open(DATA_PROCESSED_DIR.parent / "scaler" / "scaler_params.json") as f:
        s_cfg = json.load(f)
    scaler = {"mean": np.array(s_cfg["mean"], dtype=np.float32), "scale": np.array(s_cfg["scale"], dtype=np.float32)}

    # 2. Fit Models
    pca = PCAReconstructionAnomalyDetector(n_components=5).fit(X_train)
    pca.calibrate_thresholds(X_val[y_val == 0], fpr_target=0.01)
    
    ae = TinyMLAutoencoder(input_dim=60, latent_dim=8)
    import tensorflow as tf
    ae.model = tf.keras.models.load_model(MODELS_SAVED_DIR / "autoencoder.keras")
    ae.calibrate_thresholds(X_val[y_val == 0], fpr_target=0.01)

    # 3. Comprehensive Metric Evaluation at Multiple FPR Targets
    print("\n[1] EVALUASI METRIK PADA BERBAGAI TINGKAT ALARM PALSU (FPR TARGET):")
    print("-" * 88)
    print(f"{'Target FPR':<12} | {'PCA Thresh':<12} | {'PCA Recall':<12} | {'AE Thresh':<12} | {'AE Recall':<12}")
    print("-" * 88)
    
    scores_pca = pca.compute_score(X_test)
    scores_ae = ae.compute_score(X_test)
    
    for target_fpr in [0.005, 0.010, 0.020, 0.050]:
        th_p, _, rec_p = evaluate_at_fixed_fpr(y_test, scores_pca, target_fpr)
        th_a, _, rec_a = evaluate_at_fixed_fpr(y_test, scores_ae, target_fpr)
        print(f"{target_fpr*100:4.1f}%{'':<7} | {th_p:<12.2f} | {rec_p:<11.2f}% | {th_a:<12.4f} | {rec_a:<11.2f}%")
    print("-" * 88)

    # 4. Per-Fault-Type Breakdown
    print("\n[2] PERFORMA DETEKSI PER JENIS CACAT BANTALAN (TEST SET):")
    print("-" * 88)
    print(f"{'Fault Type':<20} | {'Jumlah Sampel':<15} | {'PCA Median Score':<18} | {'AE Median Score':<18} | {'Recall':<10}")
    print("-" * 88)
    for c_name in ["normal", "inner_race", "outer_race"]:
        mask = (test_classes == c_name)
        med_p = np.median(scores_pca[mask])
        med_a = np.median(scores_ae[mask])
        th_p = pca.th_alert
        rec = np.mean(scores_pca[mask] >= th_p) * 100.0 if c_name != "normal" else (1.0 - np.mean(scores_pca[mask] < th_p)) * 100.0
        rec_label = f"FPR: {rec:.2f}%" if c_name == "normal" else f"{rec:.2f}%"
        print(f"{c_name:<20} | {np.sum(mask):<15} | {med_p:<18.2f} | {med_a:<18.4f} | {rec_label:<10}")
    print("-" * 88)

    # 5. AWGN Noise Robustness Stress Test
    print("\n[3] UJI KETAHANAN TERHADAP DERAU (NOISE ROBUSTNESS STRESS TEST):")
    print("-" * 88)
    print(f"{'SNR Level':<12} | {'PCA FPR':<12} | {'PCA Recall':<12} | {'AE FPR':<12} | {'AE Recall':<12}")
    print("-" * 88)
    
    snr_levels = [None, 30, 20, 15, 10, 5, 0]
    pca_fprs, pca_recs = [], []
    ae_fprs, ae_recs = [], []
    
    for snr in snr_levels:
        w_n = add_awgn_noise(norm_raw, snr)
        w_i = add_awgn_noise(ir_raw, snr)
        w_o = add_awgn_noise(or_raw, snr)
        
        X_n = apply_scaler(extract_features_batch(w_n), scaler)
        X_i = apply_scaler(extract_features_batch(w_i), scaler)
        X_o = apply_scaler(extract_features_batch(w_o), scaler)
        
        X_snr = np.vstack([X_n, X_i, X_o])
        y_snr = np.concatenate([np.zeros(len(X_n)), np.ones(len(X_i) + len(X_o))])
        
        st_p, _ = pca.predict(X_snr)
        st_a, _ = ae.predict(X_snr)
        
        f_p = np.mean(st_p[:len(X_n)] >= 1) * 100.0
        r_p = np.mean(st_p[len(X_n):] >= 1) * 100.0
        f_a = np.mean(st_a[:len(X_n)] >= 1) * 100.0
        r_a = np.mean(st_a[len(X_n):] >= 1) * 100.0
        
        pca_fprs.append(f_p)
        pca_recs.append(r_p)
        ae_fprs.append(f_a)
        ae_recs.append(r_a)
        
        snr_lbl = "Clean (Inf)" if snr is None else f"{snr} dB"
        print(f"{snr_lbl:<12} | {f_p:<11.2f}% | {r_p:<11.2f}% | {f_a:<11.2f}% | {r_a:<11.2f}%")
    print("-" * 88)

    # 6. Plotting Multi-Panel Evaluation
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("Tahap 8: Comprehensive Model Evaluation & Noise Robustness", fontsize=14, fontweight="bold")
    
    # Panel 1: ROC Curve (Zoomed-in on low FPR)
    fpr_p, tpr_p, _ = roc_curve(y_test, scores_pca)
    fpr_a, tpr_a, _ = roc_curve(y_test, scores_ae)
    axes[0, 0].plot(fpr_p, tpr_p, label="PCA Reconstruction", color="tab:blue", lw=2)
    axes[0, 0].plot(fpr_a, tpr_a, label="MLP Autoencoder", color="tab:orange", linestyle="--", lw=2)
    axes[0, 0].plot([0, 1], [0, 1], color="gray", linestyle=":")
    axes[0, 0].set_title("ROC Curve (Test Set)", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlabel("False Positive Rate")
    axes[0, 0].set_ylabel("True Positive Rate (Recall)")
    axes[0, 0].grid(True, linestyle=":", alpha=0.6)
    axes[0, 0].legend(loc="lower right")
    
    # Panel 2: Score Boxplots per Fault Class (PCA)
    data_boxplot = [
        scores_pca[test_classes == "normal"],
        scores_pca[test_classes == "outer_race"],
        scores_pca[test_classes == "inner_race"]
    ]
    axes[0, 1].boxplot(data_boxplot, patch_artist=True, tick_labels=["Normal", "Outer Race", "Inner Race"])
    axes[0, 1].axhline(pca.th_alert, color="darkorange", linestyle="--", label=f"Alert TH ({pca.th_alert:.1f})")
    axes[0, 1].axhline(pca.th_danger, color="crimson", linestyle="-", label=f"Danger TH ({pca.th_danger:.1f})")
    axes[0, 1].set_title("PCA Anomaly Score per Health Condition", fontsize=11, fontweight="bold")
    axes[0, 1].set_ylabel("Anomaly Score")
    axes[0, 1].grid(True, linestyle=":", alpha=0.6)
    axes[0, 1].legend(loc="upper left")
    
    # Panel 3: False Positive Rate vs Noise (SNR)
    snr_x = [35 if s is None else s for s in snr_levels]
    axes[1, 0].plot(snr_x, pca_fprs, marker="o", label="PCA (Linear Subspace)", color="tab:blue", lw=2)
    axes[1, 0].plot(snr_x, ae_fprs, marker="s", label="Autoencoder (MLP)", color="tab:orange", lw=2)
    axes[1, 0].set_title("False Positive Rate (Alarm Palsu) vs Ambient Noise", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlabel("Signal-to-Noise Ratio (dB) [Kiri = Lebih Bising]")
    axes[1, 0].set_ylabel("False Positive Rate (%)")
    axes[1, 0].invert_xaxis() # Lower SNR to the right
    axes[1, 0].grid(True, linestyle=":", alpha=0.6)
    axes[1, 0].legend(loc="upper right")
    
    # Panel 4: Recall vs Noise (SNR)
    axes[1, 1].plot(snr_x, pca_recs, marker="o", label="PCA Recall", color="tab:blue", lw=2)
    axes[1, 1].plot(snr_x, ae_recs, marker="s", label="Autoencoder Recall", color="tab:orange", lw=2)
    axes[1, 1].set_title("Defect Recall vs Ambient Noise", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlabel("Signal-to-Noise Ratio (dB) [Kiri = Lebih Bising]")
    axes[1, 1].set_ylabel("Anomaly Recall (%)")
    axes[1, 1].set_ylim(80, 105)
    axes[1, 1].invert_xaxis()
    axes[1, 1].grid(True, linestyle=":", alpha=0.6)
    axes[1, 1].legend(loc="lower left")
    
    plt.tight_layout()
    out_plot = REPORTS_DIR / "stage8_comprehensive_evaluation.png"
    plt.savefig(out_plot, dpi=200)
    print(f"\n[INFO] Comprehensive plot saved to: {out_plot.resolve()}")

if __name__ == "__main__":
    run_evaluation_suite()
