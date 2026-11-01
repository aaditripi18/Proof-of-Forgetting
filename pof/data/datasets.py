"""Datasets as in-memory uint8 arrays.

Everything is held as (N, C, H, W) uint8 plus int64 labels and normalised on the
fly, so loaders never touch disk mid-training and augmentation is a seeded tensor
op rather than a per-worker PIL transform (one less source of nondeterminism).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from pof.seed import derive_seed


def _split_held_out(x, y, val_size: int, aux_size: int, seed: int):
    """Official test split -> validation | server auxiliary | test, disjoint.

    `aux` is data the *server* holds (e.g. public data); only server-side methods
    such as canary_suppressor may use it. Keeping it apart from val (the V1
    reference) and test (accuracy, V2) stops a server method from fitting the
    very images a verifier queries.
    """
    perm = np.random.default_rng(derive_seed(seed, "val_test")).permutation(len(y))
    v, a, t = perm[:val_size], perm[val_size:val_size + aux_size], perm[val_size + aux_size:]
    return x[v], y[v], x[a], y[a], x[t], y[t]


def _cifar10(root: str):
    """CIFAR-10 from a local mirror if one is set and verifies, else downloaded to root.

    POF_DATA_MIRROR names a directory holding an extracted `cifar-10-batches-py`
    (e.g. a Kaggle input). It is an environment variable, not a config key, so
    machine-specific paths never enter the config hash. torchvision checks every
    file's md5 against the official release, so a modified copy is rejected.
    """
    from torchvision.datasets import CIFAR10

    def load(where: str, download: bool):
        return (CIFAR10(where, train=True, download=download),
                CIFAR10(where, train=False, download=download))

    mirror = os.environ.get("POF_DATA_MIRROR")
    tr = te = None
    if mirror:
        try:
            tr, te = load(mirror, download=False)  # raises if absent or md5 mismatch
            print(f"CIFAR-10: using verified mirror {mirror}")
        except RuntimeError:
            print(f"CIFAR-10: mirror {mirror} missing or failed md5 check; downloading")
    if tr is None:
        tr, te = load(root, download=True)
    to_nchw = lambda a: np.ascontiguousarray(a.transpose(0, 3, 1, 2))
    return (to_nchw(tr.data), np.array(tr.targets, dtype=np.int64),
            to_nchw(te.data), np.array(te.targets, dtype=np.int64), 10)


def _pathmnist(root: str, size: int, train_size: int | None, seed: int):
    """PathMNIST (9 tissue classes) at `size` x `size` pixels.

    The official val and test splits are pooled as the held-out data: the test
    split alone (7180) would leave little once val and aux are carved from it.
    `train_size` subsamples the training split (seeded), so the federation can
    match CIFAR-10's 50000 and client sizes stay comparable.
    """
    import medmnist

    Path(root).mkdir(parents=True, exist_ok=True)
    split = {s: medmnist.PathMNIST(split=s, download=True, root=root, size=size)
             for s in ("train", "val", "test")}
    to_nchw = lambda ds: np.ascontiguousarray(ds.imgs.transpose(0, 3, 1, 2))
    labels = lambda ds: ds.labels.reshape(-1).astype(np.int64)
    xtr, ytr = to_nchw(split["train"]), labels(split["train"])
    if train_size is not None and train_size < len(ytr):
        keep = np.sort(np.random.default_rng(derive_seed(seed, "train_subsample"))
                       .choice(len(ytr), train_size, replace=False))
        xtr, ytr = xtr[keep], ytr[keep]
    xte = np.concatenate([to_nchw(split["val"]), to_nchw(split["test"])])
    yte = np.concatenate([labels(split["val"]), labels(split["test"])])
    return xtr, ytr, xte, yte, 9


def _synthetic(seed: int, n_train: int = 1200, n_test: int = 600, num_classes: int = 4,
               noise: float = 60.0):
    """Class-dependent colour blobs plus noise: learnable in a few CPU epochs and
    small enough that a model can memorise it, so membership signal exists.

    At the default noise a random-pixel trigger is indistinguishable from the
    noise and no backdoor can be planted; tests of V2 and canary_suppressor use
    data.synthetic_noise to lower it."""
    rng = np.random.default_rng(derive_seed(seed, "synthetic"))
    means = rng.uniform(40, 215, size=(num_classes, 3, 1, 1))

    def make(n):
        y = rng.integers(0, num_classes, size=n)
        x = means[y] + rng.normal(0, noise, size=(n, 3, 32, 32))
        return np.clip(x, 0, 255).astype(np.uint8), y.astype(np.int64)

    return (*make(n_train), *make(n_test), num_classes)


def load_dataset(cfg: dict, seed: int) -> dict:
    d = cfg["data"]
    name = d["name"]
    if name == "cifar10":
        xtr, ytr, xte, yte, k = _cifar10(d["root"])
    elif name == "pathmnist":
        # optional keys, absent from default.yaml so CIFAR-10 config hashes are unchanged
        xtr, ytr, xte, yte, k = _pathmnist(d["root"], d.get("image_size", 28),
                                           d.get("train_size"), seed)
    elif name == "synthetic":
        xtr, ytr, xte, yte, k = _synthetic(seed, noise=d.get("synthetic_noise", 60.0))
    else:
        raise ValueError(f"unknown dataset '{name}'")

    xval, yval, xaux, yaux, xtest, ytest = _split_held_out(xte, yte, d["val_size"],
                                                           d.get("aux_size", 0), seed)
    mean = xtr.reshape(len(xtr), xtr.shape[1], -1).mean(axis=(0, 2)) / 255.0
    std = xtr.reshape(len(xtr), xtr.shape[1], -1).std(axis=(0, 2)) / 255.0
    return {"train_x": xtr, "train_y": ytr, "val_x": xval, "val_y": yval,
            "aux_x": xaux, "aux_y": yaux, "test_x": xtest, "test_y": ytest, "num_classes": k,
            "mean": mean.astype(np.float32), "std": std.astype(np.float32)}
