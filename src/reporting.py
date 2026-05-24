"""Tabular summaries for paper-oriented experiment outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SUBGROUP_CATEGORY = {
    "none": "none",
    "Z2_rot180": "partial",
    "Z2_reflection": "partial",
    "C4": "partial",
    "D2_V4": "partial",
    "D4": "full",
    "D4_pair_tied": "beyond_D4",
    "D4_qubit_tied": "beyond_D4",
    "D4_all_tied": "beyond_D4",
}

SUBGROUP_ORDERING = [
    "none",
    "Z2_rot180",
    "Z2_reflection",
    "C4",
    "D2_V4",
    "D4",
    "D4_pair_tied",
    "D4_qubit_tied",
    "D4_all_tied",
]


def _format_float(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Render a small DataFrame as a GitHub-flavored Markdown table."""
    headers = list(df.columns)
    rows = [[_format_float(value) for value in row] for row in df.to_numpy()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def make_main_results_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create the core subgroup table requested for the paper plan."""
    required = {
        "subgroup",
        "train_size",
        "seed",
        "test_accuracy",
        "generalization_gap",
        "num_parameters",
        "subgroup_order",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns for main table: {missing}")

    grouped = df.groupby(["subgroup", "train_size"], as_index=False).agg(
        n=("seed", "count"),
        test_accuracy=("test_accuracy", "mean"),
        generalization_gap=("generalization_gap", "mean"),
        num_parameters=("num_parameters", "first"),
        subgroup_order=("subgroup_order", "first"),
    )

    test_pivot = grouped.pivot(index="subgroup", columns="train_size", values="test_accuracy")
    gap_pivot = grouped.pivot(index="subgroup", columns="train_size", values="generalization_gap")
    seed_counts = grouped.pivot(index="subgroup", columns="train_size", values="n")

    rows = []
    for subgroup in SUBGROUP_ORDERING:
        if subgroup not in grouped["subgroup"].values:
            continue
        subgroup_rows = grouped[grouped["subgroup"] == subgroup]
        row: dict[str, object] = {
            "subgroup": subgroup,
            "category": SUBGROUP_CATEGORY.get(subgroup, "unknown"),
            "order": int(subgroup_rows["subgroup_order"].iloc[0]),
            "params": int(subgroup_rows["num_parameters"].iloc[0]),
        }
        for train_size in sorted(test_pivot.columns):
            row[f"test@{train_size}"] = test_pivot.loc[subgroup, train_size]
        row["mean_test"] = subgroup_rows["test_accuracy"].mean()
        row["mean_gap"] = subgroup_rows["generalization_gap"].mean()
        row["min_n"] = int(seed_counts.loc[subgroup].min())
        rows.append(row)

    return pd.DataFrame(rows)


def write_main_results_table(
    input_csv: Path,
    *,
    output_csv: Path,
    output_md: Path,
) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    table = make_main_results_table(df)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_csv, index=False)
    output_md.write_text(dataframe_to_markdown(table), encoding="utf-8")
    return table


def make_approximate_symmetry_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create a compact summary for epsilon sweeps.

    The table contains two record types:
      - best_at_epsilon: best subgroup for each (train_size, epsilon)
      - crossover: first epsilon where subgroup overtakes D4 for each (train_size, subgroup)
    """
    required = {
        "subgroup",
        "train_size",
        "epsilon",
        "seed",
        "test_accuracy",
        "num_parameters",
        "subgroup_order",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns for approximate table: {missing}")

    means = (
        df.assign(
            epsilon=pd.to_numeric(df["epsilon"], errors="coerce"),
            train_size=pd.to_numeric(df["train_size"], errors="coerce").astype(int),
        )
        .groupby(["train_size", "epsilon", "subgroup"], as_index=False)["test_accuracy"]
        .mean()
        .sort_values(["train_size", "epsilon", "subgroup"])
    )
    # best subgroup for each epsilon and train size
    best_rows: list[dict[str, object]] = []
    for (train_size, epsilon), block in means.groupby(["train_size", "epsilon"]):
        best = block.loc[block["test_accuracy"].idxmax()]
        d4_block = block[block["subgroup"] == "D4"]
        d4_test = float(d4_block["test_accuracy"].iloc[0]) if not d4_block.empty else float("nan")
        best_rows.append(
            {
                "kind": "best_at_epsilon",
                "train_size": int(train_size),
                "epsilon": float(epsilon),
                "best_subgroup": str(best["subgroup"]),
                "best_test_accuracy": float(best["test_accuracy"]),
                "d4_test_accuracy": d4_test,
                "best_minus_d4": (
                    float(0.0)
                    if pd.notna(d4_test) and abs(float(best["test_accuracy"]) - float(d4_test)) <= 1e-9
                    else (float(best["test_accuracy"] - d4_test) if pd.notna(d4_test) else float("nan"))
                ),
            }
        )

    # first epsilon where each subgroup beats D4 for each train size
    d4_reference = (
        means[means["subgroup"] == "D4"]
        .set_index(["train_size", "epsilon"])["test_accuracy"]
    )
    for subgroup in sorted(set(means["subgroup"]) - {"D4"}):
        sub_block = means[means["subgroup"] == subgroup].copy()
        for train_size, train_block in sub_block.groupby("train_size"):
            first = None
            max_advantage = float("-inf")
            max_epsilon = None
            for _, row in train_block.sort_values("epsilon").iterrows():
                epsilon = float(row["epsilon"])
                ref = d4_reference.get((int(train_size), float(epsilon)))
                if ref is None:
                    continue
                value = float(row["test_accuracy"])
                advantage = float(value) - float(ref)
                if abs(advantage) <= 1e-9:
                    advantage = 0.0
                if advantage > max_advantage:
                    max_advantage = advantage
                    max_epsilon = float(epsilon)
                if first is None and advantage > 1e-9:
                    first = float(epsilon)
            if max_epsilon is None:
                max_epsilon = float("nan")
                max_advantage = float("nan") if max_advantage == float("-inf") else max_advantage
            best_rows.append(
                {
                    "kind": "crossover",
                    "train_size": int(train_size),
                    "subgroup": subgroup,
                    "first_epsilon_beats_d4": first,
                    "max_advantage": max_advantage,
                    "epsilon_of_max_advantage": max_epsilon,
                }
            )

    return pd.DataFrame(best_rows)


def write_approximate_symmetry_summary_table(
    input_csv: Path,
    *,
    output_csv: Path,
    output_md: Path,
) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    table = make_approximate_symmetry_summary_table(df)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_csv, index=False)
    output_md.write_text(dataframe_to_markdown(table), encoding="utf-8")
    return table
