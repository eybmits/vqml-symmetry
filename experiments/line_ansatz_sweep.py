"""Compare edge-based ansatz blocks with winner-line 3-qubit interactions."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from experiments.common import add_common_args, default_csv_path, run_configs, seeds_from_args
from src.circuits import ALL_CIRCUIT_FAMILIES
from src.utils import CSV_DIR, ExperimentConfig, parse_int_list


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.set_defaults(single_qubit_block="paper")
    parser.add_argument("--train-sizes", default="30,120,450")
    parser.add_argument("--L", type=int, default=3)
    parser.add_argument("--p", type=int, default=2)
    parser.add_argument("--subgroups", default="none,C4,D4")
    parser.add_argument(
        "--circuit-families",
        default="edge,line_zzz",
        help=f"Comma-separated list from {','.join(sorted(ALL_CIRCUIT_FAMILIES))}.",
    )
    parser.add_argument(
        "--output",
        default=str(default_csv_path("results_line_ansatz_sweep.csv")),
    )
    parser.add_argument("--summary-output", default=str(CSV_DIR / "table_line_ansatz_sweep.csv"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    circuit_families = [
        item.strip() for item in args.circuit_families.split(",") if item.strip()
    ]
    unknown = sorted(set(circuit_families) - ALL_CIRCUIT_FAMILIES)
    if unknown:
        raise ValueError(f"Unknown circuit families: {unknown}")

    subgroups = [item.strip() for item in args.subgroups.split(",") if item.strip()]
    configs = [
        ExperimentConfig(
            subgroup=subgroup,
            L=args.L,
            p=args.p,
            seed=seed,
            train_size=train_size,
            test_size=args.test_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            lr=args.lr,
            pl_device=args.pl_device,
            diff_method=args.diff_method,
            single_qubit_block=args.single_qubit_block,
            circuit_family=circuit_family,
            allow_overlap_if_needed=not args.strict_disjoint,
        )
        for circuit_family in circuit_families
        for subgroup in subgroups
        for train_size in parse_int_list(args.train_sizes)
        for seed in seeds_from_args(args)
    ]
    csv_path = Path(args.output)
    df = run_configs(configs, csv_path, resume=args.resume)

    summary = (
        df.groupby(["circuit_family", "subgroup", "train_size"], dropna=False)
        .agg(
            n=("seed", "nunique"),
            train_accuracy=("train_accuracy", "mean"),
            test_accuracy=("test_accuracy", "mean"),
            generalization_gap=("generalization_gap", "mean"),
            num_parameters=("num_parameters", "mean"),
        )
        .reset_index()
        .sort_values(["train_size", "subgroup", "circuit_family"])
    )
    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    print(summary.round(4).to_string(index=False))
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
