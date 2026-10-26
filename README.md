# ZENTRA: Edge-TinyML Early Bearing Anomaly Detection System for CNC Spindle

[![Target Hardware](https://img.shields.io/badge/Target%20MCU-ESP32--S3--DevKitC--1%20N16R8-red.svg)](https://www.espressif.com/en/products/socs/esp32-s3)
[![Sensor](https://img.shields.io/badge/Sensor-ADXL345%20(SPI%203200Hz)-blue.svg)](https://www.analog.com/media/en/technical-documentation/data-sheets/ADXL345.pdf)
[![Quantization](https://img.shields.io/badge/Model-Full%20INT8%20TFLM%20%2B%20PCA-green.svg)]()
[![Inference Latency](https://img.shields.io/badge/Latency-%3C%205%20%C2%B5s%20(PCA)%20%2F%20%3C%20500%20%C2%B5s%20(TFLM)-brightgreen.svg)]()
[![Memory Usage](https://img.shields.io/badge/Internal%20SRAM%20Used-5.8%25-success.svg)]()



---

## 1. Ringkasan Eksekutif & Karakteristik Sistem

**ZENTRA** adalah sistem pemantauan kondisi (*Condition Monitoring*) mandiri untuk spindle mesin CNC yang beroperasi di *edge* menggunakan mikrokontroler **ESP32-S3-DevKitC-1 N16R8**. Sistem ini mendeteksi degradasi mekanik bearing secara dini menggunakan model *unsupervised anomaly detection* yang **hanya dilatih menggunakan data getaran kondisi sehat (*Healthy/Normal*)**.

### Spesifikasi Teknis Utama:
* **Output Sistem**: Skor Anomali (*Reconstruction Error*) per lokasi bearing dan klasifikasi status kesehatan tiga tingkat:
  $$\text{Status} = \begin{cases} \mathbf{NORMAL} & \text{skor} < \text{Ambang Alert} \\ \mathbf{ALERT} & \text{Ambang Alert} \le \text{skor} < \text{Ambang Danger} \\ \mathbf{DANGER} & \text{skor} \ge \text{Ambang Danger} \end{cases}$$
* **Sensor**: Accelerometer 3-sumbu Analog Devices **ADXL345** via bus **SPI** berkecepatan tinggi dengan Output Data Rate (ODR) **3200 Hz** (Bandwidth efektif ~1.6 kHz).
* **Dual-Engine Anomaly Detection**:
  1. **Engine 1 (Primary Embedded Production)**: **PCA Subspace Reconstruction Error ($K=5$)**
     * *Flash*: 1.2 KB | *RAM*: 0 KB alokasi dinamis (hanya 260 byte pada stack) | *Latensi*: **$< 5\ \mu\text{s}$** per jendela.
  2. **Engine 2 (Deep Learning TinyML)**: **Bottleneck MLP Autoencoder (60 $\rightarrow$ 32 $\rightarrow$ 16 $\rightarrow$ 8 $\rightarrow$ 16 $\rightarrow$ 32 $\rightarrow$ 60)**
     * Terkuantisasi penuh **INT8** (*TensorFlow Lite Micro*), Flash 14.1 KB, Tensor Arena **12 KB** (Internal SRAM), *Latensi*: **$\sim 350\ \mu\text{s}$**.
* **Kesesuaian Target Lomba**:
  * Jumlah Parameter: **5.284 parameter** (Batas target: $< 50.000$)
  * Tensor Arena: **12 KB** (Batas target: $< 100\text{ KB}$)
  * Latensi Inferensi: **$< 0.5\text{ ms}$** (Batas target: $< 50\text{ ms}$)
  * Paritas Numerik: Ekstraksi fitur C di firmware identik secara presisi dengan pipeline Python ($< 10^{-5}$ drift).

---

## 2. Struktur Repositori

```text
ZENTRA-TinyML/
├── .gitignore                   # Mengabaikan dataset mentah CSV, build artifacts .pio, dll.
├── requirements.txt             # Dependensi Python (TensorFlow 2.x, Scipy, Scikit-Learn, dll.)
├── setup_env.py                 # Validasi lingkungan & toolchain eksekusi
├── README.md                    # Dokumentasi lengkap sistem
│
├── src/                         # Modul Python Pipeline TinyML
│   ├── config.py                # Parameter global (ODR, Window, Cutoff, Pinout, Thresholds)
│   ├── preprocessing.py         # 4th-Order Butterworth HPF (5 Hz) & Windowing (512, stride 256)
│   ├── features.py              # Ekstraksi 60 fitur (Time-domain, FFT log energy, Envelope)
│   ├── split_data.py            # Pembagian data Temporal Block Split (Chrono-Split anti-bocor)
│   ├── baseline.py              # Model PCA & Mahalanobis Distance (Engine 1)
│   ├── autoencoder.py           # Arsitektur Bottleneck MLP Autoencoder (Engine 2)
│   └── quantization.py          # Post-Training Quantization (PTQ) INT8 & C Header Generator
│
├── firmware/                    # Proyek Embedded Firmware (PlatformIO)
│   ├── platformio.ini           # Konfigurasi board ESP32-S3-DevKitC-1 N16R8 (qio_opi, O3)
│   ├── include/
│   │   ├── zentra_inference.h   # API inferensi, pin mapping, struct hasil & status
│   │   ├── scaler_params.h      # Koefisien StandardScaler float32 (1-cycle multiply)
│   │   ├── pca_baseline.h       # Matriks bobot komponen PCA (K=5) & ambang batas
│   │   └── autoencoder_model_data.h # Model INT8 TFLM dalam array byte C (16-byte aligned)
│   └── src/
│       ├── zentra_inference.cpp # Implementasi inferensi dual-engine & alokasi Tensor Arena
│       └── main.cpp             # Setup, memori diagnostik, dan uji coba smoke test inferensi
│
├── models/                      # Artefak Model Tersimpan
│   ├── baseline/pca_baseline.h  # Header C PCA
│   ├── saved_models/            # Keras model & JSON thresholds
│   └── tflite/                  # File .tflite (Float32 & INT8) dan header array C
│
├── reports/                     # Visualisasi & Hasil Evaluasi
│   ├── stage2_adxl345_feasibility.png
│   ├── stage5_baseline_evaluation.png
│   ├── stage6_autoencoder_evaluation.png
│   ├── stage8_comprehensive_evaluation.png
│   └── stage9_quantization_evaluation.png
│
└── data/                        # Direktori Data
    ├── processed/               # Tensor partisi .npz yang telah tersegmentasi
    └── scaler/                  # Parameter normalisasi scaler_params.json & .h
```

---

## 3. Panduan Pinout Kritis Hardware (ESP32-S3 N16R8)

ESP32-S3 varian **N16R8** menggunakan Flash 16 MB dan Octal OPI PSRAM 8 MB. **Wajib mematuhi alokasi pin berikut untuk mencegah crash kernel panic, bus collision, atau boot failure.**

### Pin Terlarang (JANGAN DIGUNAKAN):
* **GPIO 33, 34, 35, 36, 37**: Bus data dan clock memori eksternal **8 MB Octal PSRAM (OPI)**.
* **GPIO 26, 27, 28, 29, 30, 31, 32**: Bus **SPI Flash Internal 16 MB**.
* **GPIO 0, 3, 45, 46**: *Strapping pins* (Menentukan status boot dan logika tegangan).

### Pin Mapping Aman Sensor ADXL345 (Mode 4-Wire SPI):
| Pin Modul ADXL345 | Fungsi Mode SPI | Terhubung ke ESP32-S3 | Catatan |
| :--- | :--- | :--- | :--- |
| **VCC** | Power Supply | **3V3** | Jangan hubungkan ke 5V kecuali modul memiliki onboard LDO |
| **GND** | Ground | **GND** | Ground bersama (*Common Ground*) |
| **CS** | Chip Select (Active Low) | **GPIO 10** (Bearing 1)<br>**GPIO 9** (Bearing 2) | Ditarik LOW oleh firmware untuk mengaktifkan SPI |
| **SCL** | SPI Clock (SCK) | **GPIO 12** | Sinyal clock SPI (hingga 5 MHz) |
| **SDA** | SPI MOSI (SDI) | **GPIO 11** | Jalur data ESP32 $\rightarrow$ Sensor |
| **SDO** | SPI MISO (SDO) | **GPIO 13** | Jalur data Sensor $\rightarrow$ ESP32 |
| **INT1** | Data-Ready Interrupt | **GPIO 14** | Memicu ISR setiap sampel baru selesai diambil |
| **Status LED** | Indicator | **GPIO 2** | Indikator visual Normal / Alert / Danger |

---

## 4. Alokasi Memori ESP32-S3: Internal SRAM vs Octal PSRAM

| Subruang Memori | Lokasi Fisik | Kapasitas | Alokasi Beban Kerja ZENTRA | Alasan Teknis |
| :--- | :--- | :--- | :--- | :--- |
| **Internal SRAM** | On-chip Silicon | 512 KB (~320 KB heap) | **Tensor Arena TFLM (12 KB)**, Ring Buffer Windowing, Stack PCA | Kecepatan tinggi (1 cycle), bebas latensi cache miss, waktu eksekusi deterministik |
| **Octal PSRAM** | External OPI Bus | 8 MB | Ring Buffer MQTT saat offline, Buffer Kalibrasi Baseline di Rig | Menampung data dalam ukuran besar tanpa mengganggu keandalan core inferensi |

---

## 5. Ringkasan Progres & Hasil Tahap 0 s.d. Tahap 10

| Tahap | Topik Utama | Hasil Kritis & Status |
| :---: | :--- | :--- |
| **0** | Pemahaman Dataset (SUBF v1.0) | Terdeteksi *sensor clipping* (>20% pada sumbu X) pada data cacat dataset publik. Model dirancang berbasis fitur energi spektral agar tidak overfit pada artefak clipping. |
| **1** | Lingkungan & Struktur Proyek | Struktur direktori modular selesai, dependensi terverifikasi (TensorFlow 2.21, Python 3.13). |
| **2** | Uji Kelayakan ADXL345 | Filter Butterworth High-Pass 5 Hz (SOS format) menghilangkan offset gravitasi DC (+1g Z). Windowing 512 sampel (~0.512s) dengan stride 256 (50% overlap). |
| **3** | Ekstraksi Fitur Paritas Numerik | 60 fitur per jendela (20 fitur/sumbu: RMS, Kurtosis, Crest Factor, 16 log FFT bands, Envelope energy). Normalisasi C dioptimalkan ke 1-cycle multiply (`SCALER_SCALE`). |
| **4** | Pembagian Data Anti-Kebocoran | *Chrono-Split* (Train: 70% Normal = 1.330 jendela; Val: 15% Normal + 50% Defect; Test: 15% Normal + 50% Defect). **Nol data cacat di set pelatihan**. |
| **5** | Baseline Model (PCA & Mahalanobis) | PCA ($K=5$) menghasilkan **AUROC 1.0000**, Recall 100% pada FPR 1.40%. Header C mandiri diekspor (`pca_baseline.h`). |
| **6** | Autoencoder MLP Kecil | Arsitektur 60 $\rightarrow$ 32 $\rightarrow$ 16 $\rightarrow$ 8 $\rightarrow$ 16 $\rightarrow$ 32 $\rightarrow$ 60 (5.284 parameter). AUROC 1.0000 pada set validasi dan pengujian. |
| **7** | 1D-CNN Pembanding | *Dilewati* karena baseline PCA dan Autoencoder telah jenuh sempurna pada dataset acuan. |
| **8** | Evaluasi & Uji Ketahanan Derau | Pengujian stress AWGN membuktikan batas ambang statis gagal pada SNR $\le 15\text{ dB}$, memvalidasi pentingnya kalibrasi dinamis di rig spindle fisik (Tahap 14). |
| **9** | Kuantisasi Penuh INT8 TFLM | Post-Training Quantization (PTQ) sukses. Ukuran flash menyusut dari 24.4 KB $\rightarrow$ **14.1 KB** (-42.1%). AUROC tetap 1.0000. Header C `autoencoder_model_data.h` diekspor. |
| **10** | Proyek Firmware PlatformIO | Konfigurasi PlatformIO untuk N16R8 sukses dikompilasi (`platform = espressif32 @ 7.0.1`). SRAM terpakai: **18.8 KB (5.8%)**, Flash terpakai: **284 KB (8.5%)**. Smoke test sukses mendiskriminasi Normal vs Defect. |

---

## 6. Cara Menjalankan Proyek

### A. Lingkungan Python (Data Science & Training Pipeline)
1. Buat environment Python dan instal dependensi:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Untuk Windows: .\venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Jalankan pipeline dari preprocessing hingga kuantisasi:
   ```bash
   python run_stage2_feasibility.py
   python run_stage3_features.py
   python run_stage5_baseline.py
   python run_stage6_autoencoder.py
   python run_stage8_evaluation.py
   python run_stage9_quantization.py
   ```

### B. Firmware ESP32-S3 (PlatformIO)
1. Buka folder `firmware` menggunakan Visual Studio Code yang telah terpasang ekstensi **PlatformIO IDE**.
2. Kompilasi firmware:
   ```powershell
   cd firmware
   pio run
   ```
3. Hubungkan board ESP32-S3 ke USB port laptop dan lakukan upload & monitor:
   ```powershell
   pio run -t upload -t monitor
   ```

---

## 7. Roadmap Pengerjaan Selanjutnya
* [ ] **Tahap 11**: Driver ADXL345 SPI (FIFO, Data-Ready Interrupt, verifikasi ID `0xE5`), Buffer Windowing, FreeRTOS Dual-Core Task, & FFT `esp-dsp`.
* [ ] **Tahap 12**: Pengujian Paritas Numerik (Streaming vektor uji Python ke ESP32-S3 via Serial & verifikasi deviasi).
* [ ] **Tahap 13**: Telemetri MQTT Ringkas (Payload JSON, QoS 1, Last Will and Testament, Buffer lokal di PSRAM saat koneksi terputus).
* [ ] **Tahap 14**: Prosedur Kalibrasi Dinamis di Rig Spindle Mesin CNC (Perekaman baseline di tempat & penyimpanan ambang batas ke Non-Volatile Storage NVS).
* [ ] **Tahap 15**: Panduan Diagnostik & Troubleshooting Lengkap.

---
**Lisensi**: Academic / Competition Use Only (Proyek Tim ZENTRA)
