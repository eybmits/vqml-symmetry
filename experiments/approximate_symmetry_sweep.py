"""Stage 2: approximate symmetry sweep with epsilon-corrupted labels."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.common import add_common_args, default_csv_path, run_configs, seeds_from_args
from src.groups_d4 import subgroup_names
from src.plotting import plot_approximate_symmetry
from src.utils import ExperimentConfig, CSV_DIR, parse_float_list, parse_int_list


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.set_defaults(single_qubit_block="paper")
    parser.add_argument(
        "--eps-values",
        default="0.0,0.05,0.10,0.20,0.40",
        help="Comma-separated epsilon values for orbit-label corruption.",
    )
    parser.add_argument("--train-sizes", default="30,60,120,240,450")
    parser.add_argument("--L", type=int, default=3)
    parser.add_argument("--p", type=int, default=2)
    parser.add_argument(
        "--subgroups",
        default=",".join(subgroup_names()),
        help="Comma-separated list of subgroups to sweep.",
    )
    parser.add_argument(
        "--broken-K",
        default="C4",
        choices=list(subgroup_names()),
        help="Subgroup K under which the corrupted task is still invariant.",
    )
    parser.add_argument(
        "--output",
        default=str(default_csv_path("results_approximate_symmetry.csv")),
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    eps_values = parse_float_list(args.eps_values)
    train_sizes = parse_int_list(args.train_sizes)
    subgroups = [item.strip() for item in args.subgroups.split(",") if item.strip()]
    seeds = seeds_from_args(args)

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
            circuit_family=args.circuit_family,
            allow_overlap_if_needed=not args.strict_disjoint,
            epsilon=eps,
            broken_K=args.broken_K,
        )
        for subgroup in subgroups
        for train_size in train_sizes
        for eps in eps_values
        for seed in seeds
    ]

    csv_path = Path(args.output)
    run_configs(configs, csv_path, resume=args.resume)

    if not args.no_plots:
        plot_approximate_symmetry(csv_path)


if __name__ == "__main__":
    main()
