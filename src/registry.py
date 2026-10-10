"""
Slide 50 — Model Registry: register -> alias -> load.

Menjalankan:
    python .\src\registry.py

Serving REST (opsional):
    mlflow models serve -m "models:/breast-cancer-clf@champion" -p 8080 --env-manager local
"""

import mlflow
from mlflow import MlflowClient

from common import load_data


# ============================================================
# 1. Koneksi ke MLflow Tracking Server
# ============================================================
mlflow.set_tracking_uri("http://127.0.0.1:5000")

client = MlflowClient()
name = "breast-cancer-clf"

print("=" * 60)
print("MODEL REGISTRY — MLflow")
print("=" * 60)


# ============================================================
# 2. Cari run Random Forest terbaik
# ============================================================
runs = mlflow.search_runs(
    experiment_names=[name],
    filter_string=(
        "metrics.val_f1 > 0.95 "
        "and params.max_depth = '8'"
    ),
    order_by=["metrics.val_f1 DESC"],
    max_results=1,
)

if runs.empty:
    raise ValueError(
        "Tidak ditemukan run yang sesuai. "
        "Pastikan eksperimen breast-cancer-clf memiliki "
        "run dengan val_f1 > 0.95 dan max_depth = 8."
    )

best_run_id = runs.iloc[0]["run_id"]
best_val_f1 = runs.iloc[0]["metrics.val_f1"]

print(f"Run terbaik : {best_run_id}")
print(f"Validation F1: {best_val_f1:.4f}")


# ============================================================
# 3. Registrasi model dari run terbaik
# ============================================================
model_uri = f"runs:/{best_run_id}/model"

print("\nMendaftarkan model ke Model Registry...")

mv = mlflow.register_model(
    model_uri=model_uri,
    name=name,
)

# Tunggu hingga versi model siap digunakan
client = MlflowClient()
client.set_model_version_tag(
    name,
    mv.version,
    "validation_status",
    "passed",
)

client.update_model_version(
    name,
    mv.version,
    description="Random Forest depth=8, data v1",
)

print(f"Model terdaftar: {name}, versi {mv.version}")


# ============================================================
# 4. Buat alias challenger dan champion
# ============================================================
client.set_registered_model_alias(
    name,
    "challenger",
    mv.version,
)

print(f"Alias challenger -> versi {mv.version}")

# Pada latihan ini, model yang lolos validasi
# langsung dipromosikan menjadi champion.
client.set_registered_model_alias(
    name,
    "champion",
    mv.version,
)

print(f"Alias champion  -> versi {mv.version}")


# ============================================================
# 5. Muat model melalui alias champion
# ============================================================
X_tr, X_val, y_tr, y_val = load_data()

X_new = X_val[:5]

model = mlflow.pyfunc.load_model(
    f"models:/{name}@champion"
)

preds = model.predict(X_new)

print("\n" + "=" * 60)
print("HASIL PREDIKSI")
print("=" * 60)

print("Prediksi :", preds.tolist())
print("Label    :", y_val[:5].tolist())
print("(1 = benign, menggunakan label asli sklearn)")

print("\nProses registrasi, alias, dan pemuatan model selesai.")
