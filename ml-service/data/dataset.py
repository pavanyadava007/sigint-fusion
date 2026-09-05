"""torch Dataset over normalised I/Q arrays (training only; runtime services never import torch)."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class IQDataset(Dataset):
    def __init__(self, X, y, snr=None, augment: bool = False):
        self.X, self.y, self.snr, self.augment = X, y, snr, augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = self.X[i]
        if self.augment:  # random phase rotation (I/Q-consistent)
            th = np.random.uniform(0, 2 * np.pi)
            c, s = np.cos(th), np.sin(th)
            x = np.stack([c * x[0] - s * x[1], s * x[0] + c * x[1]]).astype(np.float32)
        return torch.from_numpy(np.ascontiguousarray(x)), int(self.y[i])
