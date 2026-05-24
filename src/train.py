"""Training loop and single-run orchestration."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data_tictactoe import DatasetSplit, make_approximate_symmetry_split, make_balanced_split
from .evaluate import accuracy_from_predictions, evaluate_arrays, vector_mse_loss
from .groups_d4 import subgroup_order
from .models import QuantumTicTacToeModel, count_parameters
from .utils import ExperimentConfig, config_to_dict, seed_everything


def _epoch_batches(
    n_items: int,
    *,
    batch_size: int,
    steps_per_epoch: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    batches: list[np.ndarray] = []
    pool = np.asarray([], dtype=int)
    for _ in range(steps_per_epoch):
        while len(pool) < batch_size:
            pool = np.concatenate([pool, rng.permutation(n_items)])
        batches.append(pool[:batch_size])
        pool = pool[batch_size:]
    return batches


def train_model(
    config: ExperimentConfig,
    *,
    split: DatasetSplit | None = None,
    eval_every: int = 0,
) -> tuple[
    QuantumTicTacToeModel,
    dict[str, int | float | str | bool | tuple[int, ...]],
    pd.DataFrame,
]:
    """Train one model and return the model, final metrics, and optional history."""
    seed_everything(config.seed)
    if split is None:
        if config.epsilon > 0.0:
            split = make_approximate_symmetry_split(
                train_size=config.train_size,
                test_size=config.test_size,
                seed=config.seed,
                disjoint=True,
                allow_overlap_if_needed=config.allow_overlap_if_needed,
                epsilon=config.epsilon,
                broken_K=config.broken_K,
            )
        else:
            split = make_balanced_split(
                train_size=config.train_size,
                test_size=config.test_size,
                seed=config.seed,
                disjoint=True,
                allow_overlap_if_needed=config.allow_overlap_if_needed,
            )

    model = QuantumTicTacToeModel(config)
    expected_params = count_parameters(config)
    if model.parameter_count != expected_params:
        raise RuntimeError(
            f"Parameter mismatch: actual={model.parameter_count}, expected={expected_params}"
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    x_train = torch.as_tensor(split.x_train, dtype=torch.float64)
    y_train = torch.as_tensor(split.y_train, dtype=torch.float64)
    rng = np.random.default_rng(config.seed + 17)
    history_rows: list[dict[str, int | float]] = []
    start_time = time.perf_counter()

    for epoch in range(config.epochs):
        model.train()
        epoch_losses: list[float] = []
        for indices in _epoch_batches(
            len(x_train),
            batch_size=config.batch_size,
            steps_per_epoch=config.steps_per_epoch,
            rng=rng,
        ):
            optimizer.zero_grad()
            predictions = model(x_train[indices])
            loss = vector_mse_loss(predictions, y_train[indices])
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().item()))

        if eval_every and ((epoch + 1) % eval_every == 0 or epoch == config.epochs - 1):
            train_pred = model(x_train).detach()
            train_loss = float(vector_mse_loss(train_pred, y_train).item())
            train_acc = accuracy_from_predictions(train_pred, y_train)
            history_rows.append(
                {
                    "epoch": epoch + 1,
                    "minibatch_loss": float(np.mean(epoch_losses)),
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                }
            )

    train_metrics = evaluate_arrays(
        model,
        split.x_train,
        split.y_train,
        batch_size=max(config.batch_size, 64),
    )
    test_metrics = evaluate_arrays(
        model,
        split.x_test,
        split.y_test,
        batch_size=max(config.batch_size, 64),
    )
    elapsed = time.perf_counter() - start_time

    row: dict[str, int | float | str | bool | tuple[int, ...]] = {
        **config_to_dict(config),
        "train_loss": train_metrics["loss"],
        "test_loss": test_metrics["loss"],
        "train_accuracy": train_metrics["accuracy"],
        "test_accuracy": test_metrics["accuracy"],
        "generalization_gap": train_metrics["accuracy"] - test_metrics["accuracy"],
        "num_parameters": model.parameter_count,
        "subgroup_order": subgroup_order(config.subgroup),
        "sharing_type": "random" if config.random_sharing else "symmetry",
        "elapsed_seconds": elapsed,
        **split.metadata,
    }
    return model, row, pd.DataFrame(history_rows)


def run_and_save_rows(
    rows: list[dict[str, int | float | str | bool | tuple[int, ...]]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
