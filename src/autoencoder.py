"""
ZENTRA TinyML - Tiny Autoencoder Module (Tahap 6)
Implements a compact 5,284-parameter MLP Autoencoder for vibration anomaly detection.
Architecture: 60 -> 32 -> 16 -> 8 (bottleneck) -> 16 -> 32 -> 60.
Trained strictly on healthy baseline data (X_train normal).
"""

import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.config import MODELS_SAVED_DIR

class TinyMLAutoencoder:
    def __init__(self, input_dim=60, latent_dim=8):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.model = self._build_architecture()
        self.th_alert = None
        self.th_danger = None
        
    def _build_architecture(self):
        """Constructs symmetric bottleneck MLP autoencoder."""
        inputs = layers.Input(shape=(self.input_dim,), name="feature_input")
        
        # Encoder
        e1 = layers.Dense(32, activation="relu", name="enc_dense1")(inputs)
        e2 = layers.Dense(16, activation="relu", name="enc_dense2")(e1)
        bottleneck = layers.Dense(self.latent_dim, activation="relu", name="bottleneck")(e2)
        
        # Decoder
        d1 = layers.Dense(16, activation="relu", name="dec_dense1")(bottleneck)
        d2 = layers.Dense(32, activation="relu", name="dec_dense2")(d1)
        outputs = layers.Dense(self.input_dim, activation="linear", name="reconstruction_output")(d2)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name="Zentra_TinyML_Autoencoder")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.003),
            loss="mse"
        )
        return model

    def fit(self, X_train_normal, X_val_normal, epochs=60, batch_size=32, verbose=0):
        """Trains autoencoder strictly on healthy baseline data with early stopping."""
        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            mode="min"
        )
        
        history = self.model.fit(
            X_train_normal, X_train_normal,
            validation_data=(X_val_normal, X_val_normal),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=verbose
        )
        return history

    def compute_score(self, X):
        """Computes Mean Squared Error (MSE) reconstruction loss per sample."""
        X_recon = self.model.predict(X, verbose=0)
        mse_scores = np.mean((X - X_recon) ** 2, axis=1)
        return mse_scores.astype(np.float32)

    def calibrate_thresholds(self, X_val_normal, fpr_target=0.01):
        """Calibrates Alert (99th percentile) and Danger thresholds on normal validation data."""
        val_scores = self.compute_score(X_val_normal)
        self.th_alert = float(np.percentile(val_scores, (1.0 - fpr_target) * 100.0))
        self.th_danger = float(np.max(val_scores) * 1.1)
        return self.th_alert, self.th_danger

    def predict(self, X):
        """Returns anomaly label: 0 (Normal), 1 (Alert), 2 (Danger)."""
        scores = self.compute_score(X)
        status = np.zeros(len(scores), dtype=np.int32)
        status[scores >= self.th_alert] = 1
        status[scores >= self.th_danger] = 2
        return status, scores

    def save(self, model_path=None, config_path=None):
        """Saves Keras model and calibrated thresholds."""
        MODELS_SAVED_DIR.mkdir(parents=True, exist_ok=True)
        if model_path is None:
            model_path = MODELS_SAVED_DIR / "autoencoder.keras"
        if config_path is None:
            config_path = MODELS_SAVED_DIR / "ae_thresholds.json"
            
        self.model.save(model_path)
        print(f"[OK] Keras model saved to: {model_path}")
        
        cfg = {
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "th_alert": self.th_alert,
            "th_danger": self.th_danger
        }
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"[OK] Autoencoder thresholds saved to: {config_path}")
