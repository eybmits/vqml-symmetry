"""Shared configuration and small utilities."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
CSV_DIR = RESULTS_DIR / "csv"
FIGURES_DIR = RESULTS_DIR / "figures"
LOGS_DIR = RESULTS_DIR / "logs"


@dataclass(frozen=True)
class ExperimentConfig:
    subgroup: str = "D4"
    L: int = 3
    p: int = 2
    seed: int = 0
    train_size: int = 450
    test_size: int = 600
    batch_size: int = 15
    epochs: int = 100
    steps_per_epoch: int = 30
    lr: float = 0.01
    random_sharing: bool = False
    device: str = "cpu"
    pl_device: str = "lightning.qubit"
    diff_method: str = "adjoint"
    single_qubit_block: str = "paper"
    circuit_family: str = "edge"
    init_scale: float = 0.1
    allow_overlap_if_needed: bool = True
    epsilon: float = 0.0
    broken_K: str = "C4"

    @property
    def n_blocks(self) -> int:
        return self.L * self.p


def config_to_dict(
    config: ExperimentConfig,
) -> dict[str, int | float | str | bool]:
    return asdict(config)


def ensure_results_dirs() -> None:
    for path in (CSV_DIR, FIGURES_DIR, LOGS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_default_dtype(torch.float64)


def parse_int_list(value: str | Iterable[int]) -> list[int]:
    if isinstance(value, str):
        if not value:
            return []
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    return [int(item) for item in value]


def parse_float_list(value: str | Iterable[float]) -> list[float]:
    if isinstance(value, str):
        if not value:
            return []
        return [float(part.strip()) for part in value.split(",") if part.strip()]
    return [float(item) for item in value]


def seed_range(n_seeds: int) -> list[int]:
    return list(range(n_seeds))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
