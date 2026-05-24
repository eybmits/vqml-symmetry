"""Deterministic Tic-Tac-Toe oracle and reversible-circuit blueprint.

This module is intentionally not a learning model. It encodes the exact
Tic-Tac-Toe winner rule and is useful as an oracle baseline, dataset sanity
check, and source of architecture ideas such as winner-line interactions.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .data_tictactoe import CLASS_NAMES, LABEL_VECTORS, WIN_LINES, generate_all_states
from .groups_d4 import PERMUTATIONS, apply_permutation_to_board

LINE_NAMES: tuple[str, ...] = (
    "top_row",
    "middle_row",
    "bottom_row",
    "left_column",
    "middle_column",
    "right_column",
    "main_diagonal",
    "anti_diagonal",
)


def winner_line_indices(board: Iterable[int], mark: int) -> tuple[int, ...]:
    """Return indices of completed winning lines for ``mark``.

    ``mark`` is ``+1`` for cross/X and ``-1`` for circle/O.
    """
    if mark not in {-1, 1}:
        raise ValueError(f"mark must be -1 or +1, got {mark}")
    values = np.asarray(tuple(board), dtype=int)
    if values.shape != (9,):
        raise ValueError(f"board must contain 9 entries, got shape {values.shape}")
    return tuple(
        line_idx
        for line_idx, line in enumerate(WIN_LINES)
        if all(values[position] == mark for position in line)
    )


def oracle_label_name(board: Iterable[int], *, strict: bool = True) -> str:
    """Classify a board using the exact Tic-Tac-Toe rule."""
    circle_lines = winner_line_indices(board, -1)
    cross_lines = winner_line_indices(board, 1)
    if circle_lines and cross_lines:
        if strict:
            raise ValueError(
                "Both players have a completed line. This should not occur for legal play."
            )
        return "draw"
    if circle_lines:
        return "circle"
    if cross_lines:
        return "cross"
    return "draw"


def oracle_label_vector(board: Iterable[int], *, strict: bool = True) -> np.ndarray:
    """Return the exact label vector in the project's {-1,+1} convention."""
    return np.asarray(LABEL_VECTORS[oracle_label_name(board, strict=strict)], dtype=np.float64)


def oracle_predict_names(x: np.ndarray, *, strict: bool = True) -> np.ndarray:
    """Predict class names for a batch of boards."""
    boards = np.asarray(x, dtype=int)
    if boards.ndim == 1:
        boards = boards[None, :]
    return np.asarray([oracle_label_name(board, strict=strict) for board in boards], dtype=object)


def oracle_predict_vectors(x: np.ndarray, *, strict: bool = True) -> np.ndarray:
    """Predict project label vectors for a batch of boards."""
    names = oracle_predict_names(x, strict=strict)
    return np.asarray([LABEL_VECTORS[str(name)] for name in names], dtype=np.float64)


def labels_from_vectors(y: np.ndarray) -> np.ndarray:
    """Convert {-1,+1} target vectors to class names by argmax."""
    targets = np.asarray(y)
    if targets.ndim != 2 or targets.shape[1] != len(CLASS_NAMES):
        raise ValueError(f"Expected target shape (n, 3), got {targets.shape}")
    return np.asarray([CLASS_NAMES[int(index)] for index in np.argmax(targets, axis=1)], dtype=object)


def line_feature_matrix(x: np.ndarray) -> np.ndarray:
    """Return deterministic winner-line flags for every board.

    The 16 output columns are ordered as 8 circle-line flags followed by 8
    cross-line flags. These exact three-cell features are a natural inspiration
    for trainable 3-qubit ansatz blocks over Tic-Tac-Toe winning lines.
    """
    boards = np.asarray(x, dtype=int)
    if boards.ndim == 1:
        boards = boards[None, :]
    features = np.zeros((len(boards), 2 * len(WIN_LINES)), dtype=np.float64)
    for row, board in enumerate(boards):
        for line_idx in winner_line_indices(board, -1):
            features[row, line_idx] = 1.0
        offset = len(WIN_LINES)
        for line_idx in winner_line_indices(board, 1):
            features[row, offset + line_idx] = 1.0
    return features


def oracle_accuracy(
    x: np.ndarray,
    y: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    *,
    strict: bool = True,
) -> float:
    """Evaluate exact-rule accuracy against either labels or target vectors."""
    if labels is None:
        if y is None:
            raise ValueError("Provide either labels or y target vectors")
        labels = labels_from_vectors(y)
    predictions = oracle_predict_names(x, strict=strict)
    return float(np.mean(predictions == np.asarray(labels, dtype=object)))


def evaluate_oracle(
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
    labels: np.ndarray | None = None,
) -> dict[str, int | float]:
    """Evaluate the exact oracle and return compact summary statistics."""
    if x is None:
        x, y, labels = generate_all_states()
    if labels is None:
        if y is None:
            raise ValueError("Provide labels or y if x is supplied")
        labels = labels_from_vectors(y)

    labels = np.asarray(labels, dtype=object)
    predictions = oracle_predict_names(np.asarray(x), strict=True)
    summary: dict[str, int | float] = {
        "n_examples": int(len(predictions)),
        "accuracy": float(np.mean(predictions == labels)),
        "num_trainable_parameters": 0,
    }
    for class_name in CLASS_NAMES:
        mask = labels == class_name
        summary[f"n_{class_name}"] = int(np.sum(mask))
        summary[f"accuracy_{class_name}"] = float(np.mean(predictions[mask] == labels[mask]))
    return summary


def oracle_is_d4_invariant() -> bool:
    """Check that the exact rule is invariant under every D4 board transform."""
    x, _, _ = generate_all_states()
    for board in x.astype(int):
        base = oracle_label_name(board)
        for perm in PERMUTATIONS.values():
            transformed = apply_permutation_to_board(board, perm)
            if oracle_label_name(transformed) != base:
                return False
    return True


def reversible_oracle_blueprint() -> dict[str, object]:
    """Return a logical reversible-circuit blueprint for exact classification.

    The blueprint uses an orthogonal two-bit encoding per cell. This is separate
    from the RX data embedding used by the variational experiments; the RX
    embedding maps three values into one qubit and is not an exact classical
    readout encoding.
    """
    line_checks = []
    for line_name, line in zip(LINE_NAMES, WIN_LINES):
        line_checks.append(
            {
                "mark": "circle",
                "line": line_name,
                "controls": tuple(f"o_{idx}" for idx in line),
                "target": f"circle_win_{line_name}",
                "gate": "MultiControlledX",
            }
        )
        line_checks.append(
            {
                "mark": "cross",
                "line": line_name,
                "controls": tuple(f"x_{idx}" for idx in line),
                "target": f"cross_win_{line_name}",
                "gate": "MultiControlledX",
            }
        )

    return {
        "encoding": {
            "empty": "(x_i=0, o_i=0)",
            "cross": "(x_i=1, o_i=0)",
            "circle": "(x_i=0, o_i=1)",
            "requires_qubits": 18,
        },
        "ancillas": {
            "line_flags": 16,
            "readout": "OR all circle line flags and OR all cross line flags",
        },
        "line_checks": tuple(line_checks),
        "trainable_parameters": 0,
        "note": (
            "This is an exact oracle/sanity circuit, not a variational model. "
            "For a trainable architecture, use these line triples as equivariant "
            "3-qubit interaction sites."
        ),
    }


def format_reversible_oracle_blueprint() -> str:
    """Format the logical circuit blueprint for CLI output."""
    blueprint = reversible_oracle_blueprint()
    lines = [
        "Exact reversible Tic-Tac-Toe oracle blueprint",
        "Encoding per board cell: empty=(0,0), cross=(1,0), circle=(0,1)",
        "For each winning line, apply one 3-control MultiControlledX for circle and one for cross:",
    ]
    for check in blueprint["line_checks"]:  # type: ignore[index]
        lines.append(
            f"- {check['target']} <- MCX({', '.join(check['controls'])})"
        )
    lines.extend(
        [
            "Readout: circle if any circle_win flag is 1; cross if any cross_win flag is 1; otherwise draw.",
            "Trainable parameters: 0",
        ]
    )
    return "\n".join(lines)
