"""Datasets as in-memory uint8 arrays.

Everything is held as (N, C, H, W) uint8 plus int64 labels and normalised on the
fly, so loaders never touch disk mid-training and augmentation is a seeded tensor
op rather than a per-worker PIL transform (one less source of nondeterminism).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from pof.seed import derive_seed


def _split_val_test(x, y, val_size: int, seed: int):
    perm = np.random.default_rng(derive_seed(seed, "val_test")).permutation(len(y))
    v, t = perm[:val_size], perm[val_size:]
    return x[v], y[v], x[t], y[t]


def _cifar10(root: str):
    from torchvision.datasets import CIFAR10

    tr = CIFAR10(root, train=True, download=True)
    te = CIFAR10(root, train=False, download=True)
    to_nchw = lambda a: np.ascontiguousarray(a.transpose(0, 3, 1, 2))
    return (to_nchw(tr.data), np.array(tr.targets, dtype=np.int64),
            to_nchw(te.data), np.array(te.targets, dtype=np.int64), 10)


def _pathmnist(root: str):
    import medmnist

    Path(root).mkdir(parents=True, exist_ok=True)
    out = []
    for split in ("train", "test"):
        ds = medmnist.PathMNIST(split=split, download=True, root=root, size=64)
        out += [np.ascontiguousarray(ds.imgs.transpose(0, 3, 1, 2)),
                ds.labels.reshape(-1).astype(np.int64)]
    return (*out, 9)


def _synthetic(seed: int, n_train: int = 1200, n_test: int = 600, num_classes: int = 4):
    """Class-dependent colour blobs plus noise: learnable in a few CPU epochs and
    small enough that a model can memorise it, so membership signal exists."""
    rng = np.random.default_rng(derive_seed(seed, "synthetic"))
    means = rng.uniform(40, 215, size=(num_classes, 3, 1, 1))

    def make(n):
        y = rng.integers(0, num_classes, size=n)
        x = means[y] + rng.normal(0, 60, size=(n, 3, 32, 32))
        return np.clip(x, 0, 255).astype(np.uint8), y.astype(np.int64)

    return (*make(n_train), *make(n_test), num_classes)


def load_dataset(cfg: dict, seed: int) -> dict:
    d = cfg["data"]
    name = d["name"]
    if name == "cifar10":
        xtr, ytr, xte, yte, k = _cifar10(d["root"])
    elif name == "pathmnist":
        xtr, ytr, xte, yte, k = _pathmnist(d["root"])
    elif name == "synthetic":
        xtr, ytr, xte, yte, k = _synthetic(seed)
    else:
        raise ValueError(f"unknown dataset '{name}'")

    xval, yval, xtest, ytest = _split_val_test(xte, yte, d["val_size"], seed)
    mean = xtr.reshape(len(xtr), xtr.shape[1], -1).mean(axis=(0, 2)) / 255.0
    std = xtr.reshape(len(xtr), xtr.shape[1], -1).std(axis=(0, 2)) / 255.0
    return {"train_x": xtr, "train_y": ytr, "val_x": xval, "val_y": yval,
            "test_x": xtest, "test_y": ytest, "num_classes": k,
            "mean": mean.astype(np.float32), "std": std.astype(np.float32)}
