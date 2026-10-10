"""Slide 44–45 — Deep learning: logging per epoch (PyTorch).

    python s44_pytorch_per_epoch.py

Yang dilatih:
  - log_params(cfg) + data_md5 (tautan ke versi data) + requirements.txt (environment)
  - metrik PER EPOCH dengan step=epoch -> learning curve di tab Model metrics
  - checkpoint terbaik sebagai artefak
"""

from pathlib import Path

import mlflow
import torch

from common import MLP, dataset_md5, evaluate, load_data, make_loaders, set_seed, train_one_epoch

mlflow.set_experiment("breast-cancer-dl")

cfg = {"lr": 3e-3, "batch_size": 64, "epochs": 20, "arch": "mlp64", "dropout": 0.2, "seed": 42}
set_seed(cfg["seed"])

X_tr, X_val, y_tr, y_val = load_data()
data_md5 = dataset_md5(X_tr, y_tr)
train_loader, val_loader = make_loaders(cfg["batch_size"], cfg["seed"])
model = MLP(dropout=cfg["dropout"])
optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

with mlflow.start_run(run_name=f"{cfg['arch']}-lr{cfg['lr']}"):
    mlflow.log_params(cfg)
    mlflow.log_param("data_md5", data_md5)                     # tautkan ke versi data
    if Path("requirements.txt").exists():
        mlflow.log_artifact("requirements.txt")                # environment

    best_f1 = 0.0
    for epoch in range(cfg["epochs"]):
        train_loss = train_one_epoch(model, train_loader, optimizer)
        val_loss, val_f1 = evaluate(model, val_loader)

        mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss,
                            "val_f1": val_f1, "lr": scheduler.get_last_lr()[0]},
                           step=epoch)
        scheduler.step()

        if val_f1 > best_f1:                                   # checkpoint terbaik
            best_f1 = val_f1
            torch.save(model.state_dict(), "best.pt")
            mlflow.log_artifact("best.pt", artifact_path="checkpoints")

    mlflow.log_metric("best_val_f1", best_f1)
    # Slide 45 menulis log_model(model, name="model") saja. Di MLflow 3.17 default-nya format
    # 'pt2' (torch.export) yang WAJIB punya input_example -> tanpa ini akan error.
    # (Ini error yang sama dengan yang kamu alami di tugas CNN.)
    example = torch.randn(3, 30).numpy()
    mlflow.pytorch.log_model(model, name="model", input_example=example)

Path("best.pt").unlink(missing_ok=True)
print(f"{cfg['arch']}  best_val_f1={best_f1:.4f}")
