"""Evaluation helpers for trained quantum classifiers."""

from __future__ import annotations

import numpy as np
import torch

from .models import QuantumTicTacToeModel


def vector_mse_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.sum((predictions - targets) ** 2, dim=1).mean()


def accuracy_from_predictions(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    pred_classes = torch.argmax(predictions, dim=1)
    target_classes = torch.argmax(targets, dim=1)
    return float((pred_classes == target_classes).to(torch.float64).mean().item())


def predict_numpy(
    model: QuantumTicTacToeModel,
    x: np.ndarray,
    *,
    batch_size: int = 64,
) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.as_tensor(x[start : start + batch_size], dtype=torch.float64)
            outputs.append(model(batch).detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def evaluate_arrays(
    model: QuantumTicTacToeModel,
    x: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int = 64,
) -> dict[str, float]:
    predictions = predict_numpy(model, x, batch_size=batch_size)
    pred_tensor = torch.as_tensor(predictions, dtype=torch.float64)
    target_tensor = torch.as_tensor(y, dtype=torch.float64)
    return {
        "loss": float(vector_mse_loss(pred_tensor, target_tensor).item()),
        "accuracy": accuracy_from_predictions(pred_tensor, target_tensor),
    }

