"""
ZENTRA TinyML - Baseline Anomaly Detection Module (Tahap 5)
Implements Mahalanobis Distance and PCA Reconstruction Error for vibration anomaly detection.
Includes threshold calibration on healthy validation data and C header export for ESP32-S3.
"""

import json
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf

from src.config import MODELS_BASELINE_DIR

class PCAReconstructionAnomalyDetector:
    """
    PCA Subspace Reconstruction Error Anomaly Detector.
    Learns the principal subspace of healthy machine vibration.
    Anomaly Score: S(x) = ||x - P * P^T * x||^2
    """
    def __init__(self, n_components=5):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.components_ = None # Shape (K, D)
        self.mean_ = None       # Shape (D,)
        self.th_alert = None
        self.th_danger = None
        
    def fit(self, X_train_normal):
        """Fits PCA strictly on healthy training feature vectors."""
        self.pca.fit(X_train_normal)
        self.components_ = self.pca.components_.astype(np.float32) # (K, D)
        self.mean_ = self.pca.mean_.astype(np.float32)             # (D,)
        return self
        
    def compute_score(self, X):
        """Computes Euclidean squared reconstruction error."""
        X_centered = X - self.mean_
        # Project: z = X_centered @ P^T
        z = X_centered @ self.components_.T # (N, K)
        # Reconstruct: X_hat = z @ P
        X_recon_centered = z @ self.components_ # (N, D)
        # Squared error
        error = np.sum((X_centered - X_recon_centered) ** 2, axis=1)
        return error.astype(np.float32)
        
    def calibrate_thresholds(self, X_val_normal, fpr_target=0.01):
        """Calibrates Alert (99th percentile) and Danger thresholds on normal validation data."""
        val_scores = self.compute_score(X_val_normal)
        self.th_alert = float(np.percentile(val_scores, (1.0 - fpr_target) * 100.0))
        self.th_danger = float(np.max(val_scores) * 1.1) # 10% safety margin above max normal
        return self.th_alert, self.th_danger
        
    def predict(self, X):
        """Returns anomaly label: 0 (Normal), 1 (Alert), 2 (Danger)."""
        scores = self.compute_score(X)
        status = np.zeros(len(scores), dtype=np.int32)
        status[scores >= self.th_alert] = 1
        status[scores >= self.th_danger] = 2
        return status, scores

    def export_c_header(self, output_path=None):
        """Exports PCA projection matrix and thresholds to C Header for ESP32-S3."""
        if output_path is None:
            output_path = MODELS_BASELINE_DIR / "pca_baseline.h"
        MODELS_BASELINE_DIR.mkdir(parents=True, exist_ok=True)
        
        K, D = self.components_.shape
        # Flatten components row-major
        comp_flat = self.components_.flatten()
        comp_str = ", ".join([f"{v:.7e}f" for v in comp_flat])
        mean_str = ", ".join([f"{v:.7e}f" for v in self.mean_])
        
        header_content = f"""/*
 * ZENTRA TinyML - PCA Reconstruction Anomaly Detector Header
 * Auto-generated embedded C baseline for ESP32-S3.
 * Flash footprint: ~{K * D * 4} bytes.
 */

#ifndef ZENTRA_PCA_BASELINE_H_
#define ZENTRA_PCA_BASELINE_H_

#define ZENTRA_PCA_K {K}
#define ZENTRA_PCA_D {D}

static const float PCA_THRESHOLD_ALERT  = {self.th_alert:.7e}f;
static const float PCA_THRESHOLD_DANGER = {self.th_danger:.7e}f;

static const float PCA_MEAN[ZENTRA_PCA_D] = {{
    {mean_str}
}};

// PCA Projection Matrix P of shape [{K}][{D}] (row-major)
static const float PCA_COMPONENTS[ZENTRA_PCA_K * ZENTRA_PCA_D] = {{
    {comp_str}
}};

#endif // ZENTRA_PCA_BASELINE_H_
"""
        with open(output_path, "w") as f:
            f.write(header_content)
        print(f"[OK] PCA C header exported to: {output_path}")

class MahalanobisAnomalyDetector:
    """
    Mahalanobis Distance Anomaly Detector with Ledoit-Wolf Covariance Regularization.
    Anomaly Score: D_M^2(x) = (x - mu)^T * Sigma^{-1} * (x - mu)
    """
    def __init__(self):
        self.cov_model = LedoitWolf()
        self.precision_ = None # Precision matrix Sigma^{-1} (D, D)
        self.mean_ = None      # Mean vector (D,)
        self.th_alert = None
        self.th_danger = None
        
    def fit(self, X_train_normal):
        self.cov_model.fit(X_train_normal)
        self.precision_ = self.cov_model.precision_.astype(np.float32)
        self.mean_ = self.cov_model.location_.astype(np.float32)
        return self
        
    def compute_score(self, X):
        diff = X - self.mean_
        scores = np.sum((diff @ self.precision_) * diff, axis=1)
        return scores.astype(np.float32)
        
    def calibrate_thresholds(self, X_val_normal, fpr_target=0.01):
        val_scores = self.compute_score(X_val_normal)
        self.th_alert = float(np.percentile(val_scores, (1.0 - fpr_target) * 100.0))
        self.th_danger = float(np.max(val_scores) * 1.1)
        return self.th_alert, self.th_danger
        
    def predict(self, X):
        scores = self.compute_score(X)
        status = np.zeros(len(scores), dtype=np.int32)
        status[scores >= self.th_alert] = 1
        status[scores >= self.th_danger] = 2
        return status, scores
