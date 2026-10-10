"""Slide 46 — Autologging.

    python s46_autolog.py

Bandingkan run 'rf-autolog' dengan 'rf-baseline' (slide 42) di UI:
autolog mencatat SEMUA hyperparameter RF + metrik training + model + signature
secara otomatis, tapi TIDAK tahu versi data & metrik validasi
-> itu yang ditambahkan manual (praktik terbaik di slide).
"""

import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from common import dataset_md5, load_data

mlflow.set_experiment("breast-cancer-clf")
mlflow.sklearn.autolog()                     # atau mlflow.autolog() untuk deteksi otomatis

X_tr, X_val, y_tr, y_val = load_data()

with mlflow.start_run(run_name="rf-autolog"):
    model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    model.fit(X_tr, y_tr)                    # <- di sini autolog bekerja

    # Log manual: hal yang autolog tidak tahu
    mlflow.set_tags({"stage": "baseline", "data_md5": dataset_md5(X_tr, y_tr)})
    mlflow.log_metric("val_f1", f1_score(y_val, model.predict(X_val)))

mlflow.sklearn.autolog(disable=True)
print("rf-autolog selesai. Buka run-nya: hitung berapa params yang tercatat otomatis.")
