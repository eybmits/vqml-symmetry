"""Stage 2: train-size sweep over partial and full equivariance groups."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.common import add_common_args, default_csv_path, run_configs, seeds_from_args
from src.groups_d4 import subgroup_names
from src.plotting import plot_parameter_count, plot_partial_equivariance
from src.utils import ExperimentConfig, parse_int_list


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--train-sizes", default="30,60,120,240,450")
    parser.add_argument("--L", type=int, default=3)
    parser.add_argument("--p", type=int, default=2)
    parser.add_argument(
        "--subgroups",
        default=",".join(subgroup_names()),
        help="Comma-separated subgroup list to sweep.",
    )
    parser.add_argument("--output", default=str(default_csv_path("results_partial_equivariance.csv")))
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
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
            circuit_family=args.circuit_family,
            allow_overlap_if_needed=not args.strict_disjoint,
        )
        for subgroup in subgroups
        for train_size in parse_int_list(args.train_sizes)
        for seed in seeds_from_args(args)
    ]
    csv_path = Path(args.output)
    run_configs(configs, csv_path, resume=args.resume)
    if not args.no_plots:
        plot_partial_equivariance(csv_path)
        plot_parameter_count(csv_path)


if __name__ == "__main__":
    main()
