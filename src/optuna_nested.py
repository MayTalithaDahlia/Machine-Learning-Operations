"""Slide 47–48 — Hyperparameter tuning dengan nested runs (Optuna + MLflow).

    python s47_optuna_nested.py              # 50 trial (seperti slide), ± 1–3 menit di CPU
    python s47_optuna_nested.py --trials 20  # lebih cepat

UI: parent 'optuna-sweep-01' dengan N child run -> pilih semua child -> Compare
-> Parallel Coordinates (lr, batch_size, dropout -> val_f1).
"""

import argparse

import mlflow
import optuna
import torch

from common import MLP, evaluate, make_loaders, set_seed, train_one_epoch

EPOCHS = 15


def train_and_eval(params):
    """Fungsi yang diasumsikan ada di slide: latih dengan params, kembalikan val_f1 terbaik."""
    set_seed(42)                                   # seed dikunci -> beda hasil hanya dari params
    train_loader, val_loader = make_loaders(params["batch_size"], seed=42)
    model = MLP(dropout=params["dropout"])
    opt = torch.optim.Adam(model.parameters(), lr=params["lr"])
    best = 0.0
    for _ in range(EPOCHS):
        train_one_epoch(model, train_loader, opt)
        best = max(best, evaluate(model, val_loader)[1])
    return best


def objective(trial):
    params = {
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
    }
    with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
        mlflow.log_params(params)
        val_f1 = train_and_eval(params)
        mlflow.log_metric("val_f1", val_f1)
    return val_f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=50)
    args = ap.parse_args()

    mlflow.set_experiment("breast-cancer-dl")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    with mlflow.start_run(run_name="optuna-sweep-01"):              # parent run
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=args.trials)
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_val_f1", study.best_value)

    print(f"{args.trials} trial selesai. best_val_f1={study.best_value:.4f}  {study.best_params}")


if __name__ == "__main__":
    main()
