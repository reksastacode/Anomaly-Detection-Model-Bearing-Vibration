"""
ZENTRA TinyML - Tahap 1: Setup & Environment Verification Script
Validates Python version, checks library compatibility, creates directory tree,
and tests TFLite Converter availability.
"""

import sys
import os
from pathlib import Path

def print_header(text):
    print("\n" + "=" * 70)
    print(f" {text}")
    print("=" * 70)

def create_directory_structure():
    """Creates the standardized folder structure for the TinyML project."""
    print_header("1. CREATING PROJECT DIRECTORY STRUCTURE")
    
    # Root directory is the parent folder where setup_env.py resides
    root = Path(__file__).resolve().parent
    
    directories = [
        root / "data" / "raw",
        root / "data" / "processed",
        root / "data" / "scaler",
        root / "models" / "baseline",
        root / "models" / "saved_models",
        root / "models" / "tflite",
        root / "reports",
        root / "src",
        root / "firmware"
    ]
    
    for d in directories:
        d.mkdir(parents=True, exist_ok=True)
        print(f"  [OK] Directory verified/created: {d.relative_to(root)}")
        
    # Check if raw CSV Files exist
    csv_raw = root / "CSV Files"
    if csv_raw.exists():
        print(f"  [OK] Found raw dataset directory: {csv_raw.relative_to(root)}")
    else:
        print(f"  [WARNING] 'CSV Files' directory not found in {root}. Place raw files here.")

def check_python_version():
    """Checks Python version against TinyML / TensorFlow compatibility."""
    print_header("2. CHECKING PYTHON ENVIRONMENT")
    ver = sys.version_info
    print(f"  Current Python Version: {ver.major}.{ver.minor}.{ver.micro}")
    print(f"  Platform: {sys.platform}")
    
    if ver.major == 3 and 9 <= ver.minor <= 12:
        print("  [STATUS: PASS] Python version is within the optimal range (3.9 - 3.12).")
    elif ver.major == 3 and ver.minor >= 13:
        print("  [STATUS: NOTICE] Python 3.13 detected. TensorFlow 2.21+ wheels are available,")
        print("                   but some legacy packages may require C++ build tools.")
    else:
        print("  [STATUS: WARNING] Python version < 3.9 is deprecated for TensorFlow.")

def check_dependencies():
    """Validates required scientific and ML libraries."""
    print_header("3. CHECKING INSTALLED PACKAGES")
    
    required_packages = [
        ("numpy", "Core Array Processing"),
        ("scipy", "Signal Processing & Filters"),
        ("pandas", "Dataframe & CSV I/O"),
        ("sklearn", "Scikit-Learn Machine Learning Baseline"),
        ("matplotlib", "Plotting & Visualizations"),
        ("tensorflow", "Deep Learning & TFLite TinyML Converter")
    ]
    
    missing_packages = []
    
    for pkg_name, description in required_packages:
        try:
            mod = __import__(pkg_name)
            ver = getattr(mod, "__version__", "Installed")
            print(f"  [PASS] {pkg_name:12} : v{ver:10} ({description})")
        except ImportError:
            print(f"  [FAIL] {pkg_name:12} : NOT INSTALLED  ({description})")
            missing_packages.append(pkg_name)
            
    if missing_packages:
        print("\n  [ACTION REQUIRED] Some packages are missing. Install them via:")
        print(f"  pip install {' '.join(missing_packages)}")
    else:
        print("\n  [STATUS: ALL CORE PACKAGES AVAILABLE]")

def test_tflite_conversion_capability():
    """Quick smoke test to verify TensorFlow Lite conversion works."""
    print_header("4. TESTING TFLITE CONVERTER CAPABILITY")
    try:
        import tensorflow as tf
        print(f"  TensorFlow Version: {tf.__version__}")
        
        # Build a miniature dummy Dense model
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(16,)),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dense(16)
        ])
        
        # Test basic conversion
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        print(f"  [PASS] TFLite Converter functional. Smoke model size: {len(tflite_model)} bytes.")
    except ImportError:
        print("  [SKIP] TensorFlow not yet installed. Install tensorflow to test TFLite conversion.")
    except Exception as e:
        print(f"  [FAIL] Error during TFLite conversion test: {e}")

if __name__ == "__main__":
    create_directory_structure()
    check_python_version()
    check_dependencies()
    test_tflite_conversion_capability()
