"""PennyLane circuit construction for subgroup-tied variational models."""

from __future__ import annotations

import math

import pennylane as qml

from .groups_d4 import CENTER, CORNERS, EDGES, N_QUBITS, SharingPattern

LINE_CIRCUIT_FAMILIES = {
    "line_zzz",
    "line_ccrz",
    "line_zzz_ccrz",
    "line_pair_crz",
    "edge_line_zzz",
    "edge_line_ccrz",
    "edge_line_zzz_ccrz",
}
ALL_CIRCUIT_FAMILIES = {"edge", *LINE_CIRCUIT_FAMILIES}


def line_parameter_channels(circuit_family: str) -> int:
    """Number of trainable interaction channels per winning-line orbit."""
    if circuit_family in {
        "line_zzz",
        "line_ccrz",
        "line_pair_crz",
        "edge_line_zzz",
        "edge_line_ccrz",
    }:
        return 1
    if circuit_family in {"line_zzz_ccrz", "edge_line_zzz_ccrz"}:
        return 2
    if circuit_family == "edge":
        return 0
    raise ValueError(f"Unknown circuit_family={circuit_family!r}")


def uses_edge_pairs(circuit_family: str) -> bool:
    """Whether the family includes the original directed-edge CRY layer."""
    return circuit_family in {"edge", "edge_line_zzz", "edge_line_ccrz", "edge_line_zzz_ccrz"}


def _apply_symmetric_ccrz(theta, line: tuple[int, int, int]) -> None:
    """Apply a trainable two-control phase on every target in a line triple."""
    for target in line:
        controls = tuple(wire for wire in line if wire != target)
        qml.ctrl(qml.RZ, control=controls)(theta, wires=target)


def _apply_line_pair_crz(theta, line: tuple[int, int, int]) -> None:
    """Apply controlled-Z rotations on all directed pairs inside a line triple."""
    for control in line:
        for target in line:
            if control != target:
                qml.CRZ(theta, wires=(control, target))


def create_qnode(
    *,
    L: int,
    p: int,
    sharing: SharingPattern,
    device_name: str = "lightning.qubit",
    diff_method: str = "adjoint",
    single_qubit_block: str = "ry",
    circuit_family: str = "edge",
):
    """Create a QNode returning the three invariant Tic-Tac-Toe observables."""
    if circuit_family not in ALL_CIRCUIT_FAMILIES:
        raise ValueError(f"circuit_family must be one of {sorted(ALL_CIRCUIT_FAMILIES)}")
    dev = qml.device(device_name, wires=N_QUBITS)

    circle_observable = qml.Hamiltonian([0.25] * len(CORNERS), [qml.PauliZ(i) for i in CORNERS])
    cross_observable = qml.Hamiltonian([0.25] * len(EDGES), [qml.PauliZ(i) for i in EDGES])

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def circuit(x, single_params, pair_params, line_params):
        block_idx = 0
        for _layer in range(L):
            for wire in range(N_QUBITS):
                qml.RX((2.0 * math.pi / 3.0) * x[wire], wires=wire)

            for _rep in range(p):
                for orbit_idx, orbit in enumerate(sharing.single_orbits):
                    for wire in orbit:
                        if single_qubit_block == "paper":
                            qml.RX(single_params[block_idx, orbit_idx, 0], wires=wire)
                            qml.RY(single_params[block_idx, orbit_idx, 1], wires=wire)
                        else:
                            qml.RY(single_params[block_idx, orbit_idx, 0], wires=wire)

                if uses_edge_pairs(circuit_family):
                    for orbit_idx, orbit in enumerate(sharing.pair_orbits):
                        theta = pair_params[block_idx, orbit_idx]
                        for control, target in orbit:
                            qml.CRY(theta, wires=(control, target))

                if circuit_family in {"line_zzz", "edge_line_zzz"}:
                    for orbit_idx, orbit in enumerate(sharing.line_orbits):
                        theta = line_params[block_idx, orbit_idx, 0]
                        for line in orbit:
                            qml.MultiRZ(theta, wires=line)
                elif circuit_family in {"line_ccrz", "edge_line_ccrz"}:
                    for orbit_idx, orbit in enumerate(sharing.line_orbits):
                        theta = line_params[block_idx, orbit_idx, 0]
                        for line in orbit:
                            _apply_symmetric_ccrz(theta, line)
                elif circuit_family in {"line_zzz_ccrz", "edge_line_zzz_ccrz"}:
                    for orbit_idx, orbit in enumerate(sharing.line_orbits):
                        zzz_theta = line_params[block_idx, orbit_idx, 0]
                        ccrz_theta = line_params[block_idx, orbit_idx, 1]
                        for line in orbit:
                            qml.MultiRZ(zzz_theta, wires=line)
                            _apply_symmetric_ccrz(ccrz_theta, line)
                elif circuit_family == "line_pair_crz":
                    for orbit_idx, orbit in enumerate(sharing.line_orbits):
                        theta = line_params[block_idx, orbit_idx, 0]
                        for line in orbit:
                            _apply_line_pair_crz(theta, line)
                block_idx += 1

        return (
            qml.expval(circle_observable),
            qml.expval(qml.PauliZ(CENTER)),
            qml.expval(cross_observable),
        )

    return circuit
