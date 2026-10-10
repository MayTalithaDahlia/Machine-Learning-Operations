# Hands-on 1 — Komponen MLflow

Repository ini berisi implementasi Hands-on 1 mata kuliah **Machine Learning Operations (MLOps)**, dengan fokus pada experiment tracking, pencatatan metrik training, autologging, hyperparameter tuning, pencarian run, dan Model Registry menggunakan MLflow.

## Tujuan

Hands-on ini bertujuan memahami penggunaan MLflow untuk mencatat dan mengelola eksperimen machine learning, membandingkan hasil eksperimen, melakukan tuning hiperparameter, serta mendaftarkan dan memuat model melalui alias.

## Teknologi yang Digunakan

- Python 3.11
- MLflow 3.17.0
- scikit-learn 1.9.0
- PyTorch 2.x
- Optuna 5
- Dataset Breast Cancer Wisconsin dari scikit-learn

## Struktur Repository

```text
Machine-Learning-Operations/
├── report/
│   ├── screenshot/
│   ├── 123450009_May_Talitha_Dahlia_RA_Laporan_HandsOn1_MLflow.md
│   └── 123450009_May_Talitha_Dahlia_RA_Laporan_HandsOn1_MLflow.pdf
├── src/
│   ├── common.py
│   ├── logging_dasar.py
│   ├── pytorch_per_epoch.py
│   ├── autolog.py
│   ├── optuna_nested.py
│   ├── search_runs.py
│   └── registry.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Komponen yang Diimplementasikan

| Script | Komponen | Keterangan |
|---|---|---|
| `src/logging_dasar.py` | Logging dasar | Mencatat parameter, metrik, tag, dan model Random Forest. |
| `src/pytorch_per_epoch.py` | Logging per epoch | Mencatat metrik training, checkpoint, dan informasi eksperimen model MLP. |
| `src/autolog.py` | Autologging | Membandingkan pencatatan otomatis MLflow dengan pencatatan manual. |
| `src/optuna_nested.py` | Hyperparameter tuning | Menggunakan Optuna dengan 50 trial dan nested runs. |
| `src/search_runs.py` | Pencarian run | Mencari eksperimen berdasarkan kriteria metrik dan parameter. |
| `src/registry.py` | Model Registry | Mendaftarkan model, menggunakan alias, memuat model, dan melakukan prediksi. |

Fungsi pendukung eksperimen diletakkan di `src/common.py`.

## Dataset dan Eksperimen

Dataset yang digunakan adalah Breast Cancer Wisconsin dari scikit-learn, dengan 569 sampel dan 30 fitur numerik. Data dibagi menjadi 80% data latih dan 20% data validasi menggunakan stratified split dengan `random_state=42`.

Eksperimen mencakup Random Forest dan Multi-Layer Perceptron (MLP). MLflow digunakan untuk melacak konfigurasi, metrik evaluasi, artefak model, dan informasi eksperimen.

## Hasil Utama

- **Random Forest:** run `rf-baseline` memperoleh `val_f1` sekitar 0,9655 dan `val_auc` sekitar 0,9940.
- **Logging per epoch:** run `mlp64-lr0.003` mencapai `best_val_f1` sekitar 0,9718.
- **Autologging:** pencatatan otomatis merekam parameter dan informasi dataset yang melengkapi pencatatan manual.
- **Hyperparameter tuning:** 50 trial Optuna menghasilkan `best_val_f1` sebesar 0,9790 pada data validasi.
- **Model Registry:** model `breast-cancer-clf` didaftarkan sebagai versi 1 dan dimuat melalui alias `@champion`.

Nilai tersebut merupakan hasil eksperimen pada data validasi, bukan jaminan performa pada data baru.

## Persiapan Environment

Jalankan perintah berikut melalui PowerShell dari folder project:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Menjalankan MLflow Tracking Server

Buka terminal pertama dan aktifkan environment, kemudian jalankan:

```powershell
.\.venv\Scripts\Activate.ps1
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --port 5000
```

Buka terminal kedua, aktifkan environment, lalu atur alamat tracking server:

```powershell
.\.venv\Scripts\Activate.ps1
$env:MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
```

Buka MLflow UI melalui browser pada alamat `http://127.0.0.1:5000`.

## Menjalankan Eksperimen

Jalankan script dari folder utama project:

```powershell
python src/logging_dasar.py
python src/pytorch_per_epoch.py
python src/autolog.py
python src/optuna_nested.py
python src/search_runs.py
python src/registry.py
```

Pastikan MLflow Tracking Server aktif sebelum menjalankan script yang mencatat eksperimen.

## Dokumentasi

Laporan lengkap, hasil eksperimen, dan screenshot tersedia di folder `report/`.

## Catatan

- Jalankan eksperimen dalam environment Python yang sudah dipersiapkan.
- Jangan mengunggah kredensial, file environment rahasia, atau file lokal yang tidak diperlukan.
- Database tracking dan artefak model dapat berukuran besar sehingga perlu dikelola sesuai kebutuhan pengumpulan tugas.