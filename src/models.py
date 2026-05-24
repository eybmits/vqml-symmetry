"""PyTorch wrappers around PennyLane Tic-Tac-Toe circuits."""

from __future__ import annotations

import torch

from .circuits import (
    ALL_CIRCUIT_FAMILIES,
    create_qnode,
    line_parameter_channels,
    uses_edge_pairs,
)
from .groups_d4 import SharingPattern, make_sharing_pattern
from .utils import ExperimentConfig


def count_parameters(
    config: ExperimentConfig,
    subgroup: str | None = None,
    *,
    random_sharing: bool | None = None,
) -> int:
    selected_subgroup = subgroup or config.subgroup
    selected_random = config.random_sharing if random_sharing is None else random_sharing
    sharing = make_sharing_pattern(
        selected_subgroup,
        random_sharing=selected_random,
        seed=config.seed,
    )
    n_single_rotations = 2 if config.single_qubit_block == "paper" else 1
    if config.circuit_family not in ALL_CIRCUIT_FAMILIES:
        raise ValueError(f"circuit_family must be one of {sorted(ALL_CIRCUIT_FAMILIES)}")
    interaction_parameters = 0
    if uses_edge_pairs(config.circuit_family):
        interaction_parameters += sharing.n_pair_parameters
    interaction_parameters += line_parameter_channels(config.circuit_family) * sharing.n_line_parameters
    return config.n_blocks * (n_single_rotations * sharing.n_single_parameters + interaction_parameters)


class QuantumTicTacToeModel(torch.nn.Module):
    """A data-reuploading variational quantum classifier."""

    def __init__(self, config: ExperimentConfig):
        super().__init__()
        if config.device != "cpu":
            raise ValueError("Only CPU execution is supported for PennyLane default.qubit.")
        self.config = config
        self.sharing: SharingPattern = make_sharing_pattern(
            config.subgroup,
            random_sharing=config.random_sharing,
            seed=config.seed + 100_003,
        )
        if config.single_qubit_block not in {"ry", "paper"}:
            raise ValueError("single_qubit_block must be 'ry' or 'paper'")
        if config.circuit_family not in ALL_CIRCUIT_FAMILIES:
            raise ValueError(f"circuit_family must be one of {sorted(ALL_CIRCUIT_FAMILIES)}")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(config.seed)
        n_single_rotations = 2 if config.single_qubit_block == "paper" else 1
        self.single_params = torch.nn.Parameter(
            config.init_scale
            * torch.randn(
                config.n_blocks,
                self.sharing.n_single_parameters,
                n_single_rotations,
                generator=generator,
                dtype=torch.float64,
            )
        )
        if uses_edge_pairs(config.circuit_family):
            self.pair_params = torch.nn.Parameter(
                config.init_scale
                * torch.randn(
                    config.n_blocks,
                    self.sharing.n_pair_parameters,
                    generator=generator,
                    dtype=torch.float64,
                )
            )
        else:
            self.register_buffer(
                "pair_params",
                torch.zeros(config.n_blocks, self.sharing.n_pair_parameters, dtype=torch.float64),
            )

        n_line_channels = line_parameter_channels(config.circuit_family)
        if n_line_channels == 0:
            self.register_buffer(
                "line_params",
                torch.zeros(
                    config.n_blocks,
                    self.sharing.n_line_parameters,
                    1,
                    dtype=torch.float64,
                ),
            )
        else:
            self.line_params = torch.nn.Parameter(
                config.init_scale
                * torch.randn(
                    config.n_blocks,
                    self.sharing.n_line_parameters,
                    n_line_channels,
                    generator=generator,
                    dtype=torch.float64,
                )
            )
        self.qnode = create_qnode(
            L=config.L,
            p=config.p,
            sharing=self.sharing,
            device_name=config.pl_device,
            diff_method=config.diff_method,
            single_qubit_block=config.single_qubit_block,
            circuit_family=config.circuit_family,
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def _forward_one(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.qnode(x, self.single_params, self.pair_params, self.line_params)
        return torch.stack(tuple(outputs)).to(dtype=torch.float64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(dtype=torch.float64)
        if x.ndim == 1:
            return self._forward_one(x)
        return torch.stack([self._forward_one(sample) for sample in x], dim=0)
