"""In-memory image tensors with seeded, vectorised augmentation."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class TensorData:
    """uint8 images + labels on `device`; batches are normalised on the fly."""

    def __init__(self, x: np.ndarray, y: np.ndarray, mean: np.ndarray, std: np.ndarray,
                 device: torch.device):
        self.x = torch.from_numpy(x).to(device)
        self.y = torch.from_numpy(y).to(device)
        self.mean = torch.from_numpy(mean).view(1, -1, 1, 1).to(device)
        self.std = torch.from_numpy(std).view(1, -1, 1, 1).to(device)
        self.device = device

    def __len__(self) -> int:
        return len(self.y)

    def batch(self, idx: np.ndarray, augment: bool = False,
              gen: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        i = torch.from_numpy(np.asarray(idx, dtype=np.int64)).to(self.device)
        x = self.x[i].float().div_(255.0)
        if augment:
            x = random_crop_flip(x, gen)
        return (x - self.mean) / self.std, self.y[i]


def random_crop_flip(x: torch.Tensor, gen: torch.Generator | None, pad: int = 4) -> torch.Tensor:
    """Per-sample random crop (zero padding `pad`) and horizontal flip.
    Random draws come from a CPU generator so results do not depend on the device."""
    b, c, h, w = x.shape
    oy = torch.randint(0, 2 * pad + 1, (b,), generator=gen).to(x.device)
    ox = torch.randint(0, 2 * pad + 1, (b,), generator=gen).to(x.device)
    flip = (torch.rand(b, generator=gen) < 0.5).to(x.device)
    xp = F.pad(x, (pad, pad, pad, pad))
    rows = (oy[:, None] + torch.arange(h, device=x.device))[:, None, :, None]
    cols = (ox[:, None] + torch.arange(w, device=x.device))
    cols = torch.where(flip[:, None], cols.flip(1), cols)[:, None, None, :]
    bi = torch.arange(b, device=x.device)[:, None, None, None]
    ci = torch.arange(c, device=x.device)[None, :, None, None]
    return xp[bi, ci, rows, cols]
