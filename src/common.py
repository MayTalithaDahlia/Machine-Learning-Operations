"""Bagian yang di slide 'diasumsikan ada' tapi tidak ditulis:
set_seed, dataset_md5, model kecil, train_one_epoch, evaluate.

Supaya bisa jalan cepat di CPU, contoh deep learning (slide 44–48) memakai
MLP kecil di data Breast Cancer, bukan resnet18 di data citra.
Pola logging-nya tetap sama persis dengan slide.
"""

import hashlib
import os
import random

import mlflow
import numpy as np
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))


# ------------------------------------------------------------------ data (slide 42)
def load_data():
    X, y = load_breast_cancer(return_X_y=True)
    # Catatan: label ASLI sklearn -> 1 = benign. (Hands-on 2 membaliknya jadi 1 = malignant.)
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def dataset_md5(X, y) -> str:
    """Sidik jari data: berubah satu nilai pun -> hash berbeda."""
    h = hashlib.md5()
    h.update(np.ascontiguousarray(X).tobytes())
    h.update(np.ascontiguousarray(y).tobytes())
    return h.hexdigest()


# ------------------------------------------------------------------ deep learning helpers
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MLP(nn.Module):
    def __init__(self, n_in=30, hidden=64, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, 2))

    def forward(self, x):
        return self.net(x)


def make_loaders(batch_size, seed):
    X_tr, X_val, y_tr, y_val = load_data()
    scaler = StandardScaler().fit(X_tr)            # fit di train saja -> tanpa leakage
    X_tr, X_val = scaler.transform(X_tr), scaler.transform(X_val)
    to_t = lambda a, dt: torch.tensor(a, dtype=dt)
    train_ds = torch.utils.data.TensorDataset(to_t(X_tr, torch.float32), to_t(y_tr, torch.long))
    val_ds = torch.utils.data.TensorDataset(to_t(X_val, torch.float32), to_t(y_val, torch.long))
    g = torch.Generator().manual_seed(seed)
    return (torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g),
            torch.utils.data.DataLoader(val_ds, batch_size=256))


def train_one_epoch(model, loader, optimizer):
    model.train()
    total = 0.0
    for xb, yb in loader:
        loss = nn.functional.cross_entropy(model(xb), yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * len(yb)
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    logits = torch.cat([model(xb) for xb, _ in loader])
    y = torch.cat([yb for _, yb in loader])
    loss = nn.functional.cross_entropy(logits, y).item()
    return loss, f1_score(y.numpy(), logits.argmax(1).numpy())
