"""Slide 42–43 — Logging dasar (Random Forest).

    python s42_logging_dasar.py

Yang dilatih: params, tags, 2 metrik, dan model + input_example dalam satu run.
"""

import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score

from common import load_data

mlflow.set_experiment("breast-cancer-clf")

X_tr, X_val, y_tr, y_val = load_data()
params = {"n_estimators": 200, "max_depth": 8, "random_state": 42}

with mlflow.start_run(run_name="rf-baseline"):
    mlflow.set_tags({"stage": "baseline", "author": "may"})       # ganti dengan namamu
    mlflow.log_params(params)

    model = RandomForestClassifier(**params).fit(X_tr, y_tr)
    proba = model.predict_proba(X_val)[:, 1]

    mlflow.log_metric("val_f1", f1_score(y_val, proba > 0.5))
    mlflow.log_metric("val_auc", roc_auc_score(y_val, proba))
    mlflow.sklearn.log_model(model, name="model", input_example=X_val[:5],
                             skops_trusted_types=["sklearn.tree._tree.Tree"])   # RF (MLflow 3)

print(f"rf-baseline  val_f1={f1_score(y_val, proba > 0.5):.4f}  "
      f"val_auc={roc_auc_score(y_val, proba):.4f}")
