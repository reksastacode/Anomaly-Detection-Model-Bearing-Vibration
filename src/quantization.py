"""
ZENTRA TinyML - Post-Training Quantization Module (Tahap 9)
Converts trained Keras Autoencoder into:
  1. Float32 TFLite baseline model.
  2. Full INT8 Quantized TFLite model with representative dataset.
Evaluates accuracy drift and exports aligned C header array for ESP32-S3 firmware.
"""

from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, f1_score

from src.config import MODELS_SAVED_DIR, MODELS_TFLITE_DIR, REPORTS_DIR

def create_representative_dataset_generator(X_train_normal, num_samples=200):
    """Creates a generator feeding healthy baseline features for dynamic range calibration."""
    def generator():
        samples = X_train_normal[:num_samples]
        for i in range(len(samples)):
            yield [samples[i:i+1].astype(np.float32)]
    return generator

def convert_to_float32_tflite(keras_model, output_path):
    """Converts Keras model to unquantized Float32 TFLite model."""
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    tflite_model = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    return tflite_model

def convert_to_int8_tflite(keras_model, rep_dataset_gen, output_path):
    """
    Performs full INT8 Post-Training Quantization (PTQ).
    Quantizes weights and internal activations to signed 8-bit integers (int8).
    Maintains float32 input/output interfaces for direct sensor feature ingestion.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    
    tflite_model = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    return tflite_model

def evaluate_tflite_model(tflite_model_bytes, X_test, y_test):
    """Runs batch inference using TFLite Interpreter and computes test AUROC & MSE scores."""
    interpreter = tf.lite.Interpreter(model_content=tflite_model_bytes)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    inp_idx = input_details["index"]
    out_idx = output_details["index"]
    
    predictions = []
    for i in range(len(X_test)):
        sample = X_test[i:i+1].astype(np.float32)
        interpreter.set_tensor(inp_idx, sample)
        interpreter.invoke()
        out_tensor = interpreter.get_tensor(out_idx)
        predictions.append(out_tensor[0])
        
    predictions = np.array(predictions, dtype=np.float32)
    reconstruction_mse = np.mean((X_test - predictions) ** 2, axis=1)
    auc = roc_auc_score(y_test, reconstruction_mse)
    
    return predictions, reconstruction_mse, auc

def export_tflite_to_c_header(tflite_path, c_header_path, array_name="g_autoencoder_model_data"):
    """
    Exports .tflite binary into 16-byte aligned C byte array (.h) for ESP32-S3.
    Replaces need for external Linux 'xxd -i' utility.
    """
    with open(tflite_path, "rb") as f:
        bytes_data = f.read()
        
    hex_lines = []
    line_bytes = []
    for b in bytes_data:
        line_bytes.append(f"0x{b:02x}")
        if len(line_bytes) == 12:
            hex_lines.append("  " + ", ".join(line_bytes))
            line_bytes = []
    if line_bytes:
        hex_lines.append("  " + ", ".join(line_bytes))
        
    hex_body = ",\n".join(hex_lines)
    length = len(bytes_data)
    
    header_content = f"""/*
 * ZENTRA TinyML - TensorFlow Lite Micro Model Data Header (Auto-Generated)
 * Embedded C model byte array for ESP32-S3 firmware.
 * File: {Path(tflite_path).name}
 * Model size: {length} bytes ({length / 1024.0:.2f} KB)
 */

#ifndef ZENTRA_AUTOENCODER_MODEL_DATA_H_
#define ZENTRA_AUTOENCODER_MODEL_DATA_H_

#include <stdint.h>

#define ZENTRA_MODEL_SIZE_BYTES {length}

// 16-byte alignment is required for optimized SIMD execution on ESP32-S3
extern const unsigned char {array_name}[];
extern const unsigned int {array_name}_len;

const unsigned char {array_name}[] __attribute__((aligned(16))) = {{
{hex_body}
}};

const unsigned int {array_name}_len = {length};

#endif // ZENTRA_AUTOENCODER_MODEL_DATA_H_
"""
    with open(c_header_path, "w") as f:
        f.write(header_content)
    print(f"[OK] C Header model array saved to: {c_header_path}")
