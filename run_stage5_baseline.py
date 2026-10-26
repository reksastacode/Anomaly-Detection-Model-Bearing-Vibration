"""
ZENTRA TinyML - Tahap 5: Baseline Evaluation Script
Trains Mahalanobis Distance & PCA Reconstruction Error on healthy data,
calibrates thresholds, evaluates metrics on test set, and generates diagnostic plots.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

from src.config import DATA_PROCESSED_DIR, REPORTS_DIR, MODELS_BASELINE_DIR
from src.baseline import PCAReconstructionAnomalyDetector, MahalanobisAnomalyDetector

def evaluate_detector(model, name, X_val, y_val, X_test, y_test, test_classes):
    # 1. Validation evaluation
    val_scores = model.compute_score(X_val)
    val_auc = roc_auc_score(y_val, val_scores)
    
    # Calibrate thresholds on normal validation samples
    norm_val = X_val[y_val == 0]
    th_alert, th_danger = model.calibrate_thresholds(norm_val, fpr_target=0.01)
    
    # 2. Test evaluation
    test_status, test_scores = model.predict(X_test)
    test_auc = roc_auc_score(y_test, test_scores)
    
    # Binary detection at Alert threshold (Status >= 1 is flagged as Anomaly)
    y_pred_binary = (test_status >= 1).astype(int)
    
    # Metrics
    norm_test_mask = (y_test == 0)
    ir_test_mask = (test_classes == "inner_race")
    or_test_mask = (test_classes == "outer_race")
    
    test_fpr = np.mean(y_pred_binary[norm_test_mask]) * 100.0
    ir_recall = np.mean(y_pred_binary[ir_test_mask]) * 100.0
    or_recall = np.mean(y_pred_binary[or_test_mask]) * 100.0
    total_recall = recall_score(y_test, y_pred_binary) * 100.0
    total_precision = precision_score(y_test, y_pred_binary) * 100.0
    total_f1 = f1_score(y_test, y_pred_binary) * 100.0
    
    return {
        "name": name,
        "val_auc": val_auc,
        "test_auc": test_auc,
        "th_alert": th_alert,
        "th_danger": th_danger,
        "test_fpr": test_fpr,
        "ir_recall": ir_recall,
        "or_recall": or_recall,
        "total_recall": total_recall,
        "total_precision": total_precision,
        "total_f1": total_f1,
        "test_scores": test_scores
    }

def run_baseline_pipeline():
    print("=" * 80)
    print("ZENTRA TinyML - TAHAP 5: BASELINE ANOMALY DETECTION (MAHALANOBIS & PCA)")
    print("=" * 80)
    
    # 1. Load Data
    split_file = DATA_PROCESSED_DIR / "split_data.npz"
    if not split_file.exists():
        raise FileNotFoundError(f"Missing {split_file}. Run 'python -m src.split_data' first.")
        
    data = np.load(split_file)
    X_train = data["X_train"] # Normal only
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    test_classes = data["test_classes"]
    
    print(f"[*] Loaded Datasets:")
    print(f"    - Train Normal  : {X_train.shape[0]} samples (features={X_train.shape[1]})")
    print(f"    - Validation    : {X_val.shape[0]} samples (Normal={np.sum(y_val==0)}, Anomaly={np.sum(y_val==1)})")
    print(f"    - Test          : {X_test.shape[0]} samples (Normal={np.sum(y_test==0)}, Anomaly={np.sum(y_test==1)})")

    # 2. Train Models
    print("\n[1] TRAINING STATISTICAL BASELINE MODELS ON HEALTHY DATA:")
    
    # Model A: PCA Reconstruction (K = 5 principal components)
    pca_model = PCAReconstructionAnomalyDetector(n_components=5)
    pca_model.fit(X_train)
    res_pca = evaluate_detector(pca_model, "PCA (K=5 Components)", X_val, y_val, X_test, y_test, test_classes)
    pca_model.export_c_header()
    
    # Model B: Mahalanobis Distance (Ledoit-Wolf covariance)
    mahal_model = MahalanobisAnomalyDetector()
    mahal_model.fit(X_train)
    res_mahal = evaluate_detector(mahal_model, "Mahalanobis Distance", X_val, y_val, X_test, y_test, test_classes)

    # 3. Print Results Comparison Table
    print("\n[2] HASIL EVALUASI BASELINE PADA TEST SET TERKUNCI:")
    print("-" * 88)
    print(f"{'Metrik Evaluasi':<32} | {'PCA Reconstruction (K=5)':<25} | {'Mahalanobis Distance':<25}")
    print("-" * 88)
    print(f"{'Validation AUROC':<32} | {res_pca['val_auc']:<25.4f} | {res_mahal['val_auc']:<25.4f}")
    print(f"{'Test AUROC':<32} | {res_pca['test_auc']:<25.4f} | {res_mahal['test_auc']:<25.4f}")
    print(f"{'Ambang Batas Alert (99th pct)':<32} | {res_pca['th_alert']:<25.2f} | {res_mahal['th_alert']:<25.2f}")
    print(f"{'Ambang Batas Danger (Safety Max)':<32} | {res_pca['th_danger']:<25.2f} | {res_mahal['th_danger']:<25.2f}")
    print(f"{'Test FPR (Normal Terdeteksi Anomali)':<32} | {res_pca['test_fpr']:<24.2f}% | {res_mahal['test_fpr']:<24.2f}%")
    print(f"{'Inner Race Fault Recall':<32} | {res_pca['ir_recall']:<24.2f}% | {res_mahal['ir_recall']:<24.2f}%")
    print(f"{'Outer Race Fault Recall':<32} | {res_pca['or_recall']:<24.2f}% | {res_mahal['or_recall']:<24.2f}%")
    print(f"{'Total F1-Score':<32} | {res_pca['total_f1']:<24.2f}% | {res_mahal['total_f1']:<24.2f}%")
    print("-" * 88)

    # 4. Embedded Constraints Comparison
    print("\n[3] PERBANDINGAN SUMBER DAYA EMBEDDED (ESP32-S3):")
    print("-" * 88)
    print(f"{'Karakteristik Embedded':<32} | {'PCA Reconstruction (K=5)':<25} | {'Mahalanobis Distance':<25}")
    print("-" * 88)
    print(f"{'Parameter Disimpan':<32} | {'300 float (1.2 KB)':<25} | {'3600 float (14.4 KB)':<25}")
    print(f"{'Operasi Floating-Point / Window':<32} | {'~780 FLOPs':<25} | {'~3660 FLOPs':<25}")
    print(f"{'Estimasi Latensi di 240 MHz':<32} | {'< 5 mikrosekon':<25} | {'~ 15 mikrosekon':<25}")
    print(f"{'Alokasi Tensor Arena':<32} | {'0 KB (Matematika C Biasa)':<25} | {'0 KB (Matematika C Biasa)':<25}")
    print("-" * 88)

    # 5. Plot Score Distributions
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Tahap 5: Baseline Anomaly Score Distributions (Test Set)", fontsize=13, fontweight="bold")
    
    for ax, res in zip(axes, [res_pca, res_mahal]):
        scores = res["test_scores"]
        bins = np.linspace(0, np.percentile(scores, 98), 50)
        
        ax.hist(scores[test_classes == "normal"], bins=bins, alpha=0.6, label="Normal (Test)", color="tab:blue", density=True)
        ax.hist(scores[test_classes == "outer_race"], bins=bins, alpha=0.5, label="Outer Race Fault", color="tab:red", density=True)
        ax.hist(scores[test_classes == "inner_race"], bins=bins, alpha=0.5, label="Inner Race Fault", color="tab:orange", density=True)
        
        ax.axvline(res["th_alert"], color="darkorange", linestyle="--", lw=1.8, label=f"Alert Threshold ({res['th_alert']:.1f})")
        ax.axvline(res["th_danger"], color="crimson", linestyle="-", lw=2.0, label=f"Danger Threshold ({res['th_danger']:.1f})")
        
        ax.set_title(f"{res['name']}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Anomaly Score")
        ax.set_ylabel("Density")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right", fontsize=8)
        
    plt.tight_layout()
    out_plot = REPORTS_DIR / "stage5_baseline_evaluation.png"
    plt.savefig(out_plot, dpi=200)
    print(f"\n[INFO] Diagnostic plot saved to: {out_plot.resolve()}")

if __name__ == "__main__":
    run_baseline_pipeline()
