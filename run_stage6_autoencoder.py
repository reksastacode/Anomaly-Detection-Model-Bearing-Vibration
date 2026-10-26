"""
ZENTRA TinyML - Tahap 6: Autoencoder Training & Baseline Comparison Script
Trains MLP Autoencoder on healthy data, evaluates performance on test set,
compares head-to-head with PCA baseline, and generates diagnostic plots.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

from src.config import DATA_PROCESSED_DIR, REPORTS_DIR, MODELS_SAVED_DIR
from src.autoencoder import TinyMLAutoencoder
from src.baseline import PCAReconstructionAnomalyDetector

def run_autoencoder_pipeline():
    print("=" * 80)
    print("ZENTRA TinyML - TAHAP 6: TINY MLP AUTOENCODER ANOMALY DETECTION")
    print("=" * 80)
    
    # 1. Load Data
    split_file = DATA_PROCESSED_DIR / "split_data.npz"
    if not split_file.exists():
        raise FileNotFoundError(f"Missing {split_file}. Run 'python -m src.split_data' first.")
        
    data = np.load(split_file)
    X_train = data["X_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    test_classes = data["test_classes"]
    
    X_val_norm = X_val[y_val == 0]
    
    print(f"[*] Loaded Data:")
    print(f"    - X_train (Normal only) : {X_train.shape}")
    print(f"    - X_val_norm (Normal)   : {X_val_norm.shape}")
    print(f"    - X_test (Mixed)        : {X_test.shape}")

    # 2. Build & Train Autoencoder
    print("\n[1] MEMBANGUN DAN MELATIH MODEL AUTOENCODER:")
    ae = TinyMLAutoencoder(input_dim=60, latent_dim=8)
    ae.model.summary()
    
    print("\n    --> Memulai pelatihan Autoencoder (max 60 epoch)...")
    history = ae.fit(X_train, X_val_norm, epochs=60, batch_size=32, verbose=0)
    final_epoch = len(history.history["loss"])
    print(f"    --> Pelatihan selesai pada epoch {final_epoch}. Train Loss: {history.history['loss'][-1]:.4f}, Val Loss: {history.history['val_loss'][-1]:.4f}")

    # 3. Calibrate Thresholds
    th_alert, th_danger = ae.calibrate_thresholds(X_val_norm, fpr_target=0.01)
    ae.save()

    # 4. Evaluate Autoencoder on Test Set
    test_status, test_scores = ae.predict(X_test)
    test_auc = roc_auc_score(y_test, test_scores)
    y_pred_binary = (test_status >= 1).astype(int)
    
    norm_test_mask = (y_test == 0)
    ir_test_mask = (test_classes == "inner_race")
    or_test_mask = (test_classes == "outer_race")
    
    ae_fpr = np.mean(y_pred_binary[norm_test_mask]) * 100.0
    ae_ir_recall = np.mean(y_pred_binary[ir_test_mask]) * 100.0
    ae_or_recall = np.mean(y_pred_binary[or_test_mask]) * 100.0
    ae_f1 = f1_score(y_test, y_pred_binary) * 100.0

    # 5. Run PCA Baseline for Direct Head-to-Head Comparison
    pca = PCAReconstructionAnomalyDetector(n_components=5).fit(X_train)
    pca.calibrate_thresholds(X_val_norm, fpr_target=0.01)
    pca_status, pca_scores = pca.predict(X_test)
    pca_auc = roc_auc_score(y_test, pca_scores)
    pca_pred = (pca_status >= 1).astype(int)
    pca_fpr = np.mean(pca_pred[norm_test_mask]) * 100.0
    pca_ir_recall = np.mean(pca_pred[ir_test_mask]) * 100.0
    pca_or_recall = np.mean(pca_pred[or_test_mask]) * 100.0
    pca_f1 = f1_score(y_test, pca_pred) * 100.0

    # 6. Print Comparison Table
    print("\n[2] PERBANDINGAN HEAD-TO-HEAD: BASELINE PCA vs MLP AUTOENCODER:")
    print("-" * 88)
    print(f"{'Parameter / Metrik':<32} | {'PCA Reconstruction (K=5)':<25} | {'TinyML Autoencoder (MLP)':<25}")
    print("-" * 88)
    print(f"{'Jumlah Parameter':<32} | {'300 parameter':<25} | {'5,284 parameter':<25}")
    print(f"{'Ukuran Bobot Memori':<32} | {'1.2 KB (Flash)':<25} | {'21.1 KB (FP32) / 5.3 KB (int8)':<25}")
    print(f"{'Kebutuhan Tensor Arena':<32} | {'0 KB (C Stack murni)':<25} | {'~ 8 - 15 KB (SRAM)':<25}")
    print(f"{'Kompleksitas Runtime':<32} | {'20 baris kode C murni':<25} | {'TensorFlow Lite Micro (TFLM)':<25}")
    print(f"{'Estimasi Latensi di 240 MHz':<32} | {'< 5 mikrosekon':<25} | {'~ 350 - 800 mikrosekon':<25}")
    print(f"{'Test AUROC':<32} | {pca_auc:<25.4f} | {test_auc:<25.4f}")
    print(f"{'False Positive Rate (Alarm Palsu)':<32} | {pca_fpr:<24.2f}% | {ae_fpr:<24.2f}%")
    print(f"{'Inner Race Fault Recall':<32} | {pca_ir_recall:<24.2f}% | {ae_ir_recall:<24.2f}%")
    print(f"{'Outer Race Fault Recall':<32} | {pca_or_recall:<24.2f}% | {ae_or_recall:<24.2f}%")
    print(f"{'F1-Score Total':<32} | {pca_f1:<24.2f}% | {ae_f1:<24.2f}%")
    print("-" * 88)

    # 7. Generate Diagnostic Plots
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Tahap 6: TinyML Autoencoder Performance & Score Distributions", fontsize=13, fontweight="bold")
    
    # Left: Training Loss Curve
    axes[0].plot(history.history["loss"], label="Train Loss (MSE)", color="tab:blue", lw=1.5)
    axes[0].plot(history.history["val_loss"], label="Val Loss (MSE)", color="tab:orange", linestyle="--", lw=1.5)
    axes[0].set_title("Autoencoder Training & Validation Loss", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Reconstruction MSE Loss")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(loc="upper right")
    
    # Right: Test Score Histogram
    bins = np.linspace(0, np.percentile(test_scores, 98), 50)
    axes[1].hist(test_scores[norm_test_mask], bins=bins, alpha=0.6, label="Normal (Test)", color="tab:blue", density=True)
    axes[1].hist(test_scores[or_test_mask], bins=bins, alpha=0.5, label="Outer Race Fault", color="tab:red", density=True)
    axes[1].hist(test_scores[ir_test_mask], bins=bins, alpha=0.5, label="Inner Race Fault", color="tab:orange", density=True)
    axes[1].axvline(th_alert, color="darkorange", linestyle="--", lw=1.8, label=f"Alert TH ({th_alert:.2f})")
    axes[1].axvline(th_danger, color="crimson", linestyle="-", lw=2.0, label=f"Danger TH ({th_danger:.2f})")
    axes[1].set_title("Test Anomaly Score Distribution (Reconstruction Error)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Reconstruction MSE")
    axes[1].set_ylabel("Density")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(loc="upper right", fontsize=8)
    
    plt.tight_layout()
    out_plot = REPORTS_DIR / "stage6_autoencoder_evaluation.png"
    plt.savefig(out_plot, dpi=200)
    print(f"\n[INFO] Diagnostic plot saved to: {out_plot.resolve()}")

if __name__ == "__main__":
    run_autoencoder_pipeline()
