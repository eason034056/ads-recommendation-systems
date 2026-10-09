from __future__ import annotations

import copy
from collections.abc import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def loader(*arrays: np.ndarray, batch_size: int = 1024, shuffle: bool = False) -> DataLoader:
    return DataLoader(TensorDataset(*[torch.tensor(value) for value in arrays]), batch_size=batch_size, shuffle=shuffle)


def fit(model: nn.Module, training: tuple[np.ndarray, ...], batch_loss: Callable[[nn.Module, list[torch.Tensor]], torch.Tensor], validation_loss: Callable[[nn.Module], float],
        max_epochs: int, patience: int, learning_rate: float = 2e-3, batch_size: int = 1024) -> dict[str, float]:
    """Train with Adam and restore the epoch with the lowest validation loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_loss, best_epoch, best_state, stale = float("inf"), 0, copy.deepcopy(model.state_dict()), 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch in loader(*training, batch_size=batch_size, shuffle=True):
            optimizer.zero_grad()
            batch_loss(model, batch).backward()
            optimizer.step()
        model.eval()  # Dropout must be off whenever the model is scored.
        current = validation_loss(model)
        if current < best_loss:
            best_loss, best_epoch, best_state, stale = current, epoch, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return {"best_epoch": best_epoch, "validation_loss": best_loss}
