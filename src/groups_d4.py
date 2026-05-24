"""Dihedral group actions and orbit utilities for the Tic-Tac-Toe board."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, TypeVar

import numpy as np

N_QUBITS = 9

COORDS: dict[int, tuple[int, int]] = {
    0: (-1, 1),
    1: (0, 1),
    2: (1, 1),
    3: (1, 0),
    4: (1, -1),
    5: (0, -1),
    6: (-1, -1),
    7: (-1, 0),
    8: (0, 0),
}

CORNERS = (0, 2, 4, 6)
EDGES = (1, 3, 5, 7)
CENTER = 8

DIRECTED_PAIRS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (2, 1),
    (2, 3),
    (4, 3),
    (4, 5),
    (6, 5),
    (6, 7),
    (0, 7),
    (1, 8),
    (3, 8),
    (5, 8),
    (7, 8),
    (8, 0),
    (8, 2),
    (8, 4),
    (8, 6),
)

WIN_LINE_TRIPLES: tuple[tuple[int, int, int], ...] = (
    (0, 1, 2),
    (3, 7, 8),
    (4, 5, 6),
    (0, 6, 7),
    (1, 5, 8),
    (2, 3, 4),
    (0, 4, 8),
    (2, 6, 8),
)

TRANSFORM_NAMES = (
    "identity",
    "rot90",
    "rot180",
    "rot270",
    "reflect_vertical",
    "reflect_horizontal",
    "reflect_diagonal",
    "reflect_antidiagonal",
)

SUBGROUPS: dict[str, tuple[str, ...]] = {
    "none": ("identity",),
    "Z2_rot180": ("identity", "rot180"),
    "Z2_reflection": ("identity", "reflect_vertical"),
    "C4": ("identity", "rot90", "rot180", "rot270"),
    "D2_V4": ("identity", "rot180", "reflect_vertical", "reflect_horizontal"),
    "D4": TRANSFORM_NAMES,
}

COMPRESSION_VARIANTS = (
    "D4_pair_tied",
    "D4_qubit_tied",
    "D4_all_tied",
)

_INDEX_BY_COORD = {coord: idx for idx, coord in COORDS.items()}
T = TypeVar("T")


def transform_coord(name: str, coord: tuple[int, int]) -> tuple[int, int]:
    """Apply a named D4 transform to an ``(x, y)`` coordinate."""
    x, y = coord
    if name == "identity":
        return x, y
    if name == "rot90":
        return -y, x
    if name == "rot180":
        return -x, -y
    if name == "rot270":
        return y, -x
    if name == "reflect_vertical":
        return -x, y
    if name == "reflect_horizontal":
        return x, -y
    if name == "reflect_diagonal":
        return y, x
    if name == "reflect_antidiagonal":
        return -y, -x
    raise KeyError(f"Unknown D4 transform: {name}")


def permutation(name: str) -> tuple[int, ...]:
    """Return a permutation mapping each old index to its transformed index."""
    return tuple(_INDEX_BY_COORD[transform_coord(name, COORDS[i])] for i in range(N_QUBITS))


PERMUTATIONS: dict[str, tuple[int, ...]] = {name: permutation(name) for name in TRANSFORM_NAMES}


def subgroup_names() -> tuple[str, ...]:
    return tuple(SUBGROUPS)


def subgroup_order(subgroup: str) -> int:
    if subgroup in COMPRESSION_VARIANTS:
        return len(SUBGROUPS["D4"])
    return len(SUBGROUPS[subgroup])


def subgroup_permutations(subgroup: str) -> tuple[tuple[int, ...], ...]:
    return tuple(PERMUTATIONS[name] for name in SUBGROUPS[subgroup])


def apply_permutation_to_board(
    board: Sequence[int] | np.ndarray,
    perm: Sequence[int],
) -> np.ndarray:
    """Transform a board using a permutation that maps old positions to new positions."""
    arr = np.asarray(board)
    out = np.empty_like(arr)
    for old_idx, new_idx in enumerate(perm):
        out[new_idx] = arr[old_idx]
    return out


def apply_transform_to_board(
    board: Sequence[int] | np.ndarray,
    transform_name: str,
) -> np.ndarray:
    return apply_permutation_to_board(board, PERMUTATIONS[transform_name])


def _apply_perm_to_item(
    item: int | tuple[int, int],
    perm: Sequence[int],
) -> int | tuple[int, int]:
    if isinstance(item, tuple):
        return perm[item[0]], perm[item[1]]
    return perm[item]


def compute_orbits(
    items: Iterable[int | tuple[int, int]],
    subgroup: str,
) -> tuple[tuple[int | tuple[int, int], ...], ...]:
    """Compute item orbits under a subgroup action."""
    item_set = set(items)
    remaining = set(item_set)
    perms = subgroup_permutations(subgroup)
    orbits: list[tuple[int | tuple[int, int], ...]] = []
    while remaining:
        item = next(iter(remaining))
        orbit = {_apply_perm_to_item(item, perm) for perm in perms}
        orbit &= item_set
        ordered = tuple(sorted(orbit))
        orbits.append(ordered)
        remaining -= orbit
    return tuple(sorted(orbits, key=lambda orbit: (len(orbit), orbit)))


def qubit_orbits(subgroup: str) -> tuple[tuple[int, ...], ...]:
    return compute_orbits(range(N_QUBITS), subgroup)  # type: ignore[return-value]


def pair_orbits(subgroup: str) -> tuple[tuple[tuple[int, int], ...], ...]:
    return compute_orbits(DIRECTED_PAIRS, subgroup)  # type: ignore[return-value]


def _apply_perm_to_line(
    line: tuple[int, int, int],
    perm: Sequence[int],
) -> tuple[int, int, int]:
    return tuple(sorted(perm[index] for index in line))  # type: ignore[return-value]


def line_orbits(subgroup: str) -> tuple[tuple[tuple[int, int, int], ...], ...]:
    """Compute undirected winning-line orbits under a subgroup action."""
    item_set = {tuple(sorted(line)) for line in WIN_LINE_TRIPLES}
    remaining = set(item_set)
    perms = subgroup_permutations(subgroup)
    orbits: list[tuple[tuple[int, int, int], ...]] = []
    while remaining:
        item = next(iter(remaining))
        orbit = {_apply_perm_to_line(item, perm) for perm in perms}
        orbit &= item_set
        ordered = tuple(sorted(orbit))
        orbits.append(ordered)
        remaining -= orbit
    return tuple(sorted(orbits, key=lambda orbit: (len(orbit), orbit)))


def compression_variant_names() -> tuple[str, ...]:
    return COMPRESSION_VARIANTS


def _compressed_d4_orbits(
    variant: str,
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[tuple[int, int], ...], ...]]:
    d4_single = qubit_orbits("D4")
    d4_pairs = pair_orbits("D4")
    all_qubits = (tuple(range(N_QUBITS)),)
    all_pairs = (tuple(DIRECTED_PAIRS),)

    if variant == "D4_pair_tied":
        return d4_single, all_pairs
    if variant == "D4_qubit_tied":
        return all_qubits, d4_pairs
    if variant == "D4_all_tied":
        return all_qubits, all_pairs
    raise KeyError(f"Unknown compression variant: {variant}")


def _random_partition(
    items: Sequence[T],
    n_groups: int,
    rng: np.random.Generator,
) -> tuple[tuple[T, ...], ...]:
    if n_groups < 1 or n_groups > len(items):
        raise ValueError(f"Cannot partition {len(items)} items into {n_groups} groups")
    shuffled = list(items)
    rng.shuffle(shuffled)
    groups: list[list[T]] = [[item] for item in shuffled[:n_groups]]
    for item in shuffled[n_groups:]:
        groups[int(rng.integers(0, n_groups))].append(item)
    return tuple(tuple(sorted(group)) for group in groups)


@dataclass(frozen=True)
class SharingPattern:
    """Parameter-sharing groups for single-qubit and two-qubit gates."""

    subgroup: str
    single_orbits: tuple[tuple[int, ...], ...]
    pair_orbits: tuple[tuple[tuple[int, int], ...], ...]
    line_orbits: tuple[tuple[tuple[int, int, int], ...], ...]
    random_sharing: bool = False

    @property
    def n_single_parameters(self) -> int:
        return len(self.single_orbits)

    @property
    def n_pair_parameters(self) -> int:
        return len(self.pair_orbits)

    @property
    def n_line_parameters(self) -> int:
        return len(self.line_orbits)

    @property
    def n_parameters_per_block(self) -> int:
        return self.n_single_parameters + self.n_pair_parameters


def make_sharing_pattern(
    subgroup: str,
    *,
    random_sharing: bool = False,
    seed: int = 0,
) -> SharingPattern:
    """Create symmetry-aware or type-preserving random parameter sharing."""
    if subgroup not in SUBGROUPS and subgroup not in COMPRESSION_VARIANTS:
        raise KeyError(
            f"Unknown subgroup {subgroup!r}. Options: "
            f"{sorted((*SUBGROUPS, *COMPRESSION_VARIANTS))}"
        )

    if not random_sharing:
        if subgroup in COMPRESSION_VARIANTS:
            single_orbits, compressed_pair_orbits = _compressed_d4_orbits(subgroup)
            return SharingPattern(
                subgroup=subgroup,
                single_orbits=single_orbits,
                pair_orbits=compressed_pair_orbits,
                line_orbits=line_orbits("D4"),
                random_sharing=False,
            )
        return SharingPattern(
            subgroup=subgroup,
            single_orbits=qubit_orbits(subgroup),
            pair_orbits=pair_orbits(subgroup),
            line_orbits=line_orbits(subgroup),
            random_sharing=False,
        )

    if subgroup in COMPRESSION_VARIANTS:
        target_single, target_pairs = (
            len(orbits) for orbits in _compressed_d4_orbits(subgroup)
        )
        target_lines = len(line_orbits("D4"))
    else:
        target_single = len(qubit_orbits(subgroup))
        target_pairs = len(pair_orbits(subgroup))
        target_lines = len(line_orbits(subgroup))
    rng = np.random.default_rng(seed)
    return SharingPattern(
        subgroup=subgroup,
        single_orbits=_random_partition(tuple(range(N_QUBITS)), target_single, rng),
        pair_orbits=_random_partition(DIRECTED_PAIRS, target_pairs, rng),
        line_orbits=_random_partition(WIN_LINE_TRIPLES, target_lines, rng),
        random_sharing=True,
    )


EXPECTED_PARAMETERS_PER_BLOCK = {
    "none": 25,
    "Z2_rot180": 13,
    "Z2_reflection": 15,
    "C4": 7,
    "D2_V4": 9,
    "D4": 6,
}
