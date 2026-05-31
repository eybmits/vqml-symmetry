"""Run and validate the consistent short-paper experiment matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from experiments.common import run_configs
from src.groups_d4 import subgroup_names
from src.utils import CSV_DIR, ExperimentConfig, config_to_dict, ensure_results_dirs


TRAIN_SIZES = [30, 60, 120, 240, 450, 600]
EDGE_SUBGROUPS = subgroup_names()
LINE_SUBGROUPS = ["C4", "D4"]
AUDIT_SEEDS = [0, 1, 2]
HEADLINE_SEEDS = list(range(10))
CONTROL_SEEDS = list(range(10))

EDGE_LINES_FAMILY = "edge_line_zzz_ccrz"
ABLATION_FAMILIES = [
    "edge",
    EDGE_LINES_FAMILY,
    "edge_line_zzz",
    "edge_line_ccrz",
    "line_zzz_ccrz",
    "line_zzz",
    "line_ccrz",
    "line_pair_crz",
]

AUDIT_CSV = CSV_DIR / "results_paper_training_budget_audit_L3p2_train600.csv"
AUDIT_DECISION_JSON = CSV_DIR / "paper_training_budget_decision_L3p2.json"
EDGE_CSV = CSV_DIR / "results_paper_consistent_edge_L3p2.csv"
EDGE_LINES_CSV = CSV_DIR / "results_paper_consistent_edge_lines_L3p2.csv"
ABLATION_CSV = CSV_DIR / "results_paper_consistent_ablation_L3p2_train600.csv"
RANDOM_CSV = CSV_DIR / "results_paper_consistent_random_sharing_L3p2_train600.csv"
SUMMARY_CSV = CSV_DIR / "table_paper_consistent_summary.csv"
STAGE_OUTPUTS = [AUDIT_CSV, EDGE_CSV, EDGE_LINES_CSV, ABLATION_CSV, RANDOM_CSV]


BASE_CONFIG = {
    "L": 3,
    "p": 2,
    "test_size": 600,
    "batch_size": 15,
    "steps_per_epoch": 30,
    "lr": 0.01,
    "pl_device": "lightning.qubit",
    "diff_method": "adjoint",
    "single_qubit_block": "paper",
    "allow_overlap_if_needed": True,
}


def _config(
    *,
    subgroup: str,
    seed: int,
    train_size: int,
    epochs: int,
    circuit_family: str = "edge",
    random_sharing: bool = False,
) -> ExperimentConfig:
    return ExperimentConfig(
        subgroup=subgroup,
        seed=seed,
        train_size=train_size,
        epochs=epochs,
        circuit_family=circuit_family,
        random_sharing=random_sharing,
        **BASE_CONFIG,
    )


def _row_key(row: pd.Series) -> tuple:
    return (
        str(row.get("subgroup")),
        int(row.get("L", 3)),
        int(row.get("p", 2)),
        int(row.get("seed")),
        int(row.get("train_size")),
        int(row.get("test_size", 600)),
        int(row.get("batch_size", 15)),
        int(row.get("epochs", 100)),
        int(row.get("steps_per_epoch", 30)),
        round(float(row.get("lr", 0.01)), 12),
        bool(row.get("random_sharing", False)),
        str(row.get("pl_device", "lightning.qubit")),
        str(row.get("diff_method", "adjoint")),
        str(row.get("single_qubit_block", "paper")),
        str(row.get("circuit_family", "edge")),
        round(float(row.get("epsilon", 0.0) if pd.notna(row.get("epsilon", 0.0)) else 0.0), 12),
    )


def _config_key(config: ExperimentConfig) -> tuple:
    return _row_key(pd.Series(config_to_dict(config)))


def _shard_path(path: Path, shard_index: int, shard_count: int) -> Path:
    return path.with_name(f"{path.stem}.shard{shard_index:02d}of{shard_count:02d}{path.suffix}")


def _read_existing(path: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for candidate in [path, *sorted(path.parent.glob(f"{path.stem}.shard*of*{path.suffix}"))]:
        if candidate.exists() and candidate.stat().st_size > 0:
            frames.append(pd.read_csv(candidate))
    return _dedupe(frames)


def _run_stage_configs(
    configs: list[ExperimentConfig],
    output_path: Path,
    *,
    resume: bool,
    shard_index: int | None = None,
    shard_count: int = 1,
) -> pd.DataFrame:
    if shard_count == 1:
        return run_configs(configs, output_path, resume=resume)
    if shard_index is None or shard_index < 0 or shard_index >= shard_count:
        raise ValueError("--shard-index must be in [0, shard_count) when --shard-count > 1.")

    existing = _read_existing(output_path) if resume else pd.DataFrame()
    existing_keys = {_row_key(row) for _, row in existing.iterrows()} if not existing.empty else set()
    missing = [config for config in configs if _config_key(config) not in existing_keys]
    shard_configs = [config for index, config in enumerate(missing) if index % shard_count == shard_index]
    shard_path = _shard_path(output_path, shard_index, shard_count)
    print(
        f"Shard {shard_index}/{shard_count}: {len(shard_configs)} configs "
        f"({len(missing)} missing before sharding) -> {shard_path}"
    )
    return run_configs(shard_configs, shard_path, resume=resume)


def _dedupe(rows: list[pd.DataFrame]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    for column, value in {
        "circuit_family": "edge",
        "epsilon": 0.0,
        "random_sharing": False,
        "pl_device": "lightning.qubit",
        "diff_method": "adjoint",
        "single_qubit_block": "paper",
    }.items():
        if column not in df.columns:
            df[column] = value
        else:
            df[column] = df[column].fillna(value)
    df["_key"] = df.apply(_row_key, axis=1)
    df = df.drop_duplicates("_key", keep="first").drop(columns=["_key"])
    return df


def _write_seeded(path: Path, df: pd.DataFrame, *, overwrite: bool) -> None:
    if df.empty:
        return
    if path.exists() and not overwrite:
        existing = pd.read_csv(path)
        df = _dedupe([existing, df])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Seeded {path} with {len(df)} rows.")


def _compatible(df: pd.DataFrame, *, epochs: int = 100) -> pd.DataFrame:
    filtered = df.copy()
    checks = {
        "L": 3,
        "p": 2,
        "epochs": epochs,
        "steps_per_epoch": 30,
        "batch_size": 15,
        "lr": 0.01,
        "test_size": 600,
        "single_qubit_block": "paper",
        "pl_device": "lightning.qubit",
        "diff_method": "adjoint",
    }
    for column, value in checks.items():
        if column in filtered.columns:
            filtered = filtered[filtered[column] == value]
    return filtered


def seed_existing(*, overwrite: bool = False) -> None:
    """Populate new consistent CSVs with compatible historical rows."""
    ensure_results_dirs()
    partial_path = CSV_DIR / "results_partial_data_sweep_full_L3p2.csv"
    oracle_path = CSV_DIR / "results_oracle_inspired_robust_C4D4_L3p2_train450_600_750.csv"

    partial = _compatible(pd.read_csv(partial_path))
    partial["circuit_family"] = "edge"
    partial = partial[
        partial["subgroup"].isin(EDGE_SUBGROUPS)
        & partial["train_size"].isin(TRAIN_SIZES)
        & partial["seed"].isin(HEADLINE_SEEDS)
    ]

    oracle = _compatible(pd.read_csv(oracle_path))
    oracle_no_750 = oracle[oracle["train_size"].isin(TRAIN_SIZES)]

    edge_from_oracle = oracle_no_750[
        (oracle_no_750["circuit_family"] == "edge")
        & oracle_no_750["subgroup"].isin(["C4", "D4"])
        & oracle_no_750["seed"].isin(HEADLINE_SEEDS)
    ]
    _write_seeded(EDGE_CSV, _dedupe([partial, edge_from_oracle]), overwrite=overwrite)

    edge_lines = oracle_no_750[
        (oracle_no_750["circuit_family"] == EDGE_LINES_FAMILY)
        & oracle_no_750["subgroup"].isin(LINE_SUBGROUPS)
        & oracle_no_750["seed"].isin(HEADLINE_SEEDS)
    ]
    _write_seeded(EDGE_LINES_CSV, _dedupe([edge_lines]), overwrite=overwrite)

    ablation = oracle_no_750[
        (oracle_no_750["train_size"] == 600)
        & oracle_no_750["circuit_family"].isin(ABLATION_FAMILIES)
        & oracle_no_750["subgroup"].isin(LINE_SUBGROUPS)
        & oracle_no_750["seed"].isin(CONTROL_SEEDS)
    ]
    _write_seeded(ABLATION_CSV, _dedupe([ablation]), overwrite=overwrite)

    audit = oracle_no_750[
        (oracle_no_750["train_size"] == 600)
        & oracle_no_750["circuit_family"].isin(["edge", EDGE_LINES_FAMILY])
        & oracle_no_750["subgroup"].isin(LINE_SUBGROUPS)
        & oracle_no_750["seed"].isin(AUDIT_SEEDS)
    ]
    _write_seeded(AUDIT_CSV, _dedupe([audit]), overwrite=overwrite)


def run_audit(
    *, resume: bool = True, shard_index: int | None = None, shard_count: int = 1
) -> pd.DataFrame:
    configs = [
        _config(subgroup=subgroup, seed=seed, train_size=600, epochs=epochs, circuit_family=family)
        for epochs in [100, 200]
        for family, subgroup in [
            ("edge", "none"),
            ("edge", "C4"),
            ("edge", "D4"),
            (EDGE_LINES_FAMILY, "C4"),
            (EDGE_LINES_FAMILY, "D4"),
        ]
        for seed in AUDIT_SEEDS
    ]
    return _run_stage_configs(configs, AUDIT_CSV, resume=resume, shard_index=shard_index, shard_count=shard_count)


def choose_budget() -> dict[str, object]:
    df = pd.read_csv(AUDIT_CSV)
    required = 2 * 5 * len(AUDIT_SEEDS)
    if len(df.drop_duplicates(["circuit_family", "subgroup", "epochs", "seed"])) < required:
        raise RuntimeError(f"Audit incomplete: need {required} distinct rows before choosing budget.")

    summary = (
        df.groupby(["circuit_family", "subgroup", "epochs"], as_index=False)
        .agg(test_accuracy=("test_accuracy", "mean"), generalization_gap=("generalization_gap", "mean"))
        .sort_values(["circuit_family", "subgroup", "epochs"])
    )
    choose_200 = False
    comparisons: list[dict[str, object]] = []
    for (family, subgroup), block in summary.groupby(["circuit_family", "subgroup"]):
        by_epoch = block.set_index("epochs")
        if 100 not in by_epoch.index or 200 not in by_epoch.index:
            raise RuntimeError(f"Missing 100/200 comparison for {family}/{subgroup}.")
        delta_test = float(by_epoch.loc[200, "test_accuracy"] - by_epoch.loc[100, "test_accuracy"])
        delta_gap = float(by_epoch.loc[200, "generalization_gap"] - by_epoch.loc[100, "generalization_gap"])
        passes = delta_test >= 0.015 and delta_gap <= 0.02
        choose_200 = choose_200 or passes
        comparisons.append(
            {
                "circuit_family": family,
                "subgroup": subgroup,
                "delta_test_accuracy": delta_test,
                "delta_generalization_gap": delta_gap,
                "selects_200": passes,
            }
        )

    decision = {
        "epochs": 200 if choose_200 else 100,
        "steps_per_epoch": 30,
        "rule": "use 200x30 if any audited headline model gains >=0.015 test accuracy without >0.02 gap increase",
        "comparisons": comparisons,
    }
    AUDIT_DECISION_JSON.write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    prepare_outputs_for_epochs(int(decision["epochs"]))
    seed_chosen_audit_rows(int(decision["epochs"]))
    print(json.dumps(decision, indent=2))
    return decision


def _chosen_epochs(default: int | None = None) -> int:
    if default is not None:
        return default
    if AUDIT_DECISION_JSON.exists():
        return int(json.loads(AUDIT_DECISION_JSON.read_text(encoding="utf-8"))["epochs"])
    raise RuntimeError("No budget decision found. Run --stage audit and --stage choose-budget first.")


def run_headline_edge(
    *, epochs: int, resume: bool = True, shard_index: int | None = None, shard_count: int = 1
) -> pd.DataFrame:
    configs = [
        _config(subgroup=subgroup, seed=seed, train_size=train_size, epochs=epochs, circuit_family="edge")
        for subgroup in EDGE_SUBGROUPS
        for train_size in TRAIN_SIZES
        for seed in HEADLINE_SEEDS
    ]
    return _run_stage_configs(configs, EDGE_CSV, resume=resume, shard_index=shard_index, shard_count=shard_count)


def run_headline_edge_lines(
    *, epochs: int, resume: bool = True, shard_index: int | None = None, shard_count: int = 1
) -> pd.DataFrame:
    configs = [
        _config(subgroup=subgroup, seed=seed, train_size=train_size, epochs=epochs, circuit_family=EDGE_LINES_FAMILY)
        for subgroup in LINE_SUBGROUPS
        for train_size in TRAIN_SIZES
        for seed in HEADLINE_SEEDS
    ]
    return _run_stage_configs(configs, EDGE_LINES_CSV, resume=resume, shard_index=shard_index, shard_count=shard_count)


def run_ablation(
    *, epochs: int, resume: bool = True, shard_index: int | None = None, shard_count: int = 1
) -> pd.DataFrame:
    configs = [
        _config(subgroup=subgroup, seed=seed, train_size=600, epochs=epochs, circuit_family=family)
        for family in ABLATION_FAMILIES
        for subgroup in LINE_SUBGROUPS
        for seed in CONTROL_SEEDS
    ]
    return _run_stage_configs(configs, ABLATION_CSV, resume=resume, shard_index=shard_index, shard_count=shard_count)


def run_random(
    *, epochs: int, resume: bool = True, shard_index: int | None = None, shard_count: int = 1
) -> pd.DataFrame:
    configs: list[ExperimentConfig] = []
    for seed in CONTROL_SEEDS:
        configs.append(_config(subgroup="none", seed=seed, train_size=600, epochs=epochs))
    for subgroup in ["Z2_reflection", "Z2_rot180", "C4", "D2_V4", "D4"]:
        for random_sharing in [False, True]:
            for seed in CONTROL_SEEDS:
                configs.append(
                    _config(
                        subgroup=subgroup,
                        seed=seed,
                        train_size=600,
                        epochs=epochs,
                        random_sharing=random_sharing,
                    )
                )
    return _run_stage_configs(configs, RANDOM_CSV, resume=resume, shard_index=shard_index, shard_count=shard_count)


def merge_outputs() -> None:
    for path in STAGE_OUTPUTS:
        merged = _read_existing(path)
        if merged.empty:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(path, index=False)
        print(f"Merged {len(merged)} rows -> {path}")


def prepare_outputs_for_epochs(epochs: int) -> None:
    for path in [EDGE_CSV, EDGE_LINES_CSV, ABLATION_CSV, RANDOM_CSV]:
        for candidate in [path, *sorted(path.parent.glob(f"{path.stem}.shard*of*{path.suffix}"))]:
            if not candidate.exists() or candidate.stat().st_size == 0:
                continue
            df = pd.read_csv(candidate)
            if "epochs" not in df.columns:
                candidate.unlink()
                print(f"Removed incompatible {candidate}")
                continue
            filtered = df[df["epochs"] == epochs].copy()
            if filtered.empty:
                candidate.unlink()
                print(f"Removed non-{epochs}-epoch {candidate}")
            elif len(filtered) != len(df):
                filtered.to_csv(candidate, index=False)
                print(f"Filtered {candidate} to {len(filtered)} rows with epochs={epochs}")


def seed_chosen_audit_rows(epochs: int) -> None:
    """Reuse completed audit rows in the final paper sweep outputs."""
    audit = _read_existing(AUDIT_CSV)
    if audit.empty:
        return
    audit = audit[audit["epochs"] == epochs].copy()
    edge = audit[audit["circuit_family"] == "edge"]
    edge_lines = audit[audit["circuit_family"] == EDGE_LINES_FAMILY]
    if not edge.empty:
        _write_seeded(EDGE_CSV, _dedupe([edge]), overwrite=False)
    if not edge_lines.empty:
        _write_seeded(EDGE_LINES_CSV, _dedupe([edge_lines]), overwrite=False)


def _validate_file(
    path: Path,
    *,
    expected_rows: int,
    expected_epochs: int,
    train_sizes: list[int],
    seeds: list[int],
) -> dict[str, object]:
    if not path.exists():
        raise RuntimeError(f"Missing required result file: {path}")
    df = pd.read_csv(path)
    distinct = df.drop_duplicates(
        ["circuit_family", "subgroup", "train_size", "seed", "random_sharing"], keep="first"
    )
    protocols = df[
        ["L", "p", "epochs", "steps_per_epoch", "batch_size", "lr", "test_size", "single_qubit_block"]
    ].drop_duplicates()
    if len(protocols) != 1:
        raise RuntimeError(f"{path} has mixed protocols:\n{protocols.to_string(index=False)}")
    protocol = protocols.iloc[0].to_dict()
    if int(protocol["epochs"]) != expected_epochs:
        raise RuntimeError(f"{path} has epochs={protocol['epochs']}, expected {expected_epochs}.")
    if sorted(df["train_size"].unique().tolist()) != train_sizes:
        raise RuntimeError(f"{path} has train sizes {sorted(df['train_size'].unique().tolist())}, expected {train_sizes}.")
    if sorted(df["seed"].unique().tolist()) != seeds:
        raise RuntimeError(f"{path} has seeds {sorted(df['seed'].unique().tolist())}, expected {seeds}.")
    if len(distinct) != expected_rows:
        raise RuntimeError(f"{path} has {len(distinct)} distinct rows, expected {expected_rows}.")
    return {"file": str(path), "rows": len(distinct), "protocol": protocol}


def validate(*, epochs: int) -> None:
    rows = [
        _validate_file(EDGE_CSV, expected_rows=360, expected_epochs=epochs, train_sizes=TRAIN_SIZES, seeds=HEADLINE_SEEDS),
        _validate_file(EDGE_LINES_CSV, expected_rows=120, expected_epochs=epochs, train_sizes=TRAIN_SIZES, seeds=HEADLINE_SEEDS),
        _validate_file(ABLATION_CSV, expected_rows=160, expected_epochs=epochs, train_sizes=[600], seeds=CONTROL_SEEDS),
        _validate_file(RANDOM_CSV, expected_rows=110, expected_epochs=epochs, train_sizes=[600], seeds=CONTROL_SEEDS),
    ]
    summary_rows: list[pd.DataFrame] = []
    for path in [EDGE_CSV, EDGE_LINES_CSV, ABLATION_CSV, RANDOM_CSV]:
        df = pd.read_csv(path)
        grouped = (
            df.groupby(["circuit_family", "subgroup", "train_size", "random_sharing"], dropna=False)
            .agg(
                n=("seed", "nunique"),
                test_accuracy=("test_accuracy", "mean"),
                train_accuracy=("train_accuracy", "mean"),
                generalization_gap=("generalization_gap", "mean"),
                num_parameters=("num_parameters", "mean"),
            )
            .reset_index()
        )
        grouped["source"] = path.name
        summary_rows.append(grouped)
    pd.concat(summary_rows, ignore_index=True).to_csv(SUMMARY_CSV, index=False)
    print(json.dumps(rows, indent=2))
    print(f"Wrote {SUMMARY_CSV}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=[
            "seed-existing",
            "audit",
            "choose-budget",
            "prepare-epochs",
            "headline-edge",
            "headline-lines",
            "ablation",
            "random",
            "merge",
            "validate",
            "all",
        ],
        required=True,
    )
    parser.add_argument("--epochs", type=int, choices=[100, 200], default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--overwrite-seeded", action="store_true")
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--seeds", default="", help="Optional comma-separated seed override for local debugging only.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    resume = not args.no_resume
    if args.seeds:
        raise ValueError("This paper runner fixes seed policy; do not override seeds for final runs.")

    if args.stage in {"seed-existing", "all"}:
        seed_existing(overwrite=args.overwrite_seeded)
    if args.stage in {"audit", "all"}:
        run_audit(
            resume=resume,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
        )
    if args.stage in {"choose-budget", "all"}:
        merge_outputs()
        choose_budget()
    if args.stage == "prepare-epochs":
        prepare_outputs_for_epochs(_chosen_epochs(args.epochs))

    if args.stage in {"merge", "validate"}:
        merge_outputs()

    if args.stage in {"headline-edge", "headline-lines", "ablation", "random", "validate", "all"}:
        epochs = _chosen_epochs(args.epochs)
        if args.stage in {"headline-edge", "all"}:
            run_headline_edge(
                epochs=epochs,
                resume=resume,
                shard_index=args.shard_index,
                shard_count=args.shard_count,
            )
        if args.stage in {"headline-lines", "all"}:
            run_headline_edge_lines(
                epochs=epochs,
                resume=resume,
                shard_index=args.shard_index,
                shard_count=args.shard_count,
            )
        if args.stage in {"ablation", "all"}:
            run_ablation(
                epochs=epochs,
                resume=resume,
                shard_index=args.shard_index,
                shard_count=args.shard_count,
            )
        if args.stage in {"random", "all"}:
            run_random(
                epochs=epochs,
                resume=resume,
                shard_index=args.shard_index,
                shard_count=args.shard_count,
            )
        if args.stage in {"validate", "all"}:
            validate(epochs=epochs)


if __name__ == "__main__":
    main()
