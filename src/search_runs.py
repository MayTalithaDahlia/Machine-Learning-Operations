"""Slide 48/49 — Membandingkan dan mencari run secara programatik."""

import mlflow

# Hubungkan ke MLflow Tracking Server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Cari run berdasarkan metrik, parameter, dan tag
runs = mlflow.search_runs(
    experiment_names=["breast-cancer-clf"],
    filter_string=(
        "metrics.val_f1 > 0.95 "
        "and params.max_depth = '8' "
        "and tags.stage != 'debug'"
    ),
    order_by=["metrics.val_f1 DESC"],
    max_results=10,
)

# Periksa apakah pencarian menemukan run
if runs.empty:
    print("Tidak ada run yang memenuhi filter.")
    print("Periksa eksperimen, parameter, metrik, dan tag.")
else:
    kolom = [
        "run_id",
        "tags.mlflow.runName",
        "params.n_estimators",
        "metrics.val_f1",
        "metrics.val_auc",
    ]

    # Tampilkan hanya kolom yang tersedia
    kolom_tersedia = [k for k in kolom if k in runs.columns]
    print(runs[kolom_tersedia].to_string(index=False))

    # Ambil run terbaik dari hasil pencarian
    best_run_id = runs.iloc[0]["run_id"]
    print("\nbest_run_id =", best_run_id)
