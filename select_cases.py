"""
Select high-risk credit cases with diverse financial profiles.

Steps:
- Load the test split and predicted default probabilities.
- Filter to cases above a probability threshold (default: 0.6).
- Pick a diverse subset across loan amount, income, and debt-to-income ratio.
- Save the selected cases and a compact summary table for reporting.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Features that drive diversity in the selection step
DIVERSITY_FEATURES: List[str] = ["loan_amount", "income", "dtir1"]


def resolve_path(base_dir: Path, path_str: str) -> Path:
    """Resolve relative paths against the repository root (this file's parent)."""
    path = Path(path_str)
    return path if path.is_absolute() else base_dir / path


def load_and_merge(test_path: Path, preds_path: Path) -> pd.DataFrame:
    """Load test data and predictions, validating shape and required columns."""
    test_df = pd.read_csv(test_path)
    preds_df = pd.read_csv(preds_path)

    if "predicted_probability" not in preds_df.columns:
        raise ValueError("Predictions file must contain a 'predicted_probability' column.")

    if len(test_df) != len(preds_df):
        raise ValueError(
            f"Row mismatch between test set ({len(test_df)}) and predictions ({len(preds_df)})."
        )

    merged = test_df.copy()
    merged["predicted_probability"] = preds_df["predicted_probability"].values

    if "predicted_label" in preds_df.columns:
        merged["predicted_label"] = preds_df["predicted_label"].values

    return merged


def select_high_risk(merged: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Return high-risk cases above the probability threshold."""
    high_risk = merged[merged["predicted_probability"] > threshold].copy()
    high_risk = high_risk.reset_index().rename(columns={"index": "case_id"})
    return high_risk


def _validate_features(df: pd.DataFrame, columns: Iterable[str]) -> None:
    """Ensure all required columns are present."""
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required feature columns: {', '.join(missing)}")


def choose_diverse_cases(
    high_risk: pd.DataFrame, top_k: int, random_state: int) -> pd.DataFrame:
    """
    Pick up to `top_k` cases that are spread across key financial dimensions.

    Uses KMeans on loan_amount, income, and dtir1 to encourage diversity, then
    picks the highest-risk example from each cluster.
    """
    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    if len(high_risk) == 0:
        raise ValueError("No cases exceed the probability threshold.")

    # If there are fewer candidates than requested, just return the sorted set.
    if len(high_risk) <= top_k:
        return high_risk.sort_values("predicted_probability", ascending=False)

    _validate_features(high_risk, DIVERSITY_FEATURES)

    n_clusters = min(top_k, len(high_risk))

    scaler = StandardScaler()
    feature_matrix = scaler.fit_transform(high_risk[DIVERSITY_FEATURES])

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
    high_risk = high_risk.copy()
    high_risk["cluster_id"] = kmeans.fit_predict(feature_matrix)

    # Take the highest-risk case from each cluster.
    selected = (
        high_risk.sort_values("predicted_probability", ascending=False)
        .groupby("cluster_id", group_keys=False)
        .head(1)
    )

    # If clustering produced fewer rows than requested (edge cases), top up by risk.
    if len(selected) < top_k:
        remaining = high_risk.drop(selected.index).sort_values(
            "predicted_probability", ascending=False
        )
        selected = pd.concat([selected, remaining.head(top_k - len(selected))])

    return selected.sort_values("predicted_probability", ascending=False).head(top_k)


def build_summary(high_risk: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    """Create a compact summary for reporting."""
    summary = selected.copy()

    # Percentile rank of selected cases relative to all high-risk candidates.
    high_risk_indexed = high_risk.set_index("case_id")
    for col in DIVERSITY_FEATURES:
        percentiles = high_risk_indexed[col].rank(pct=True)
        summary[f"{col}_percentile"] = summary["case_id"].map(percentiles)

    cols = (
        ["case_id", "predicted_probability"]
        + DIVERSITY_FEATURES
        + [f"{col}_percentile" for col in DIVERSITY_FEATURES]
    )

    optional_cols = [col for col in ["cluster_id", "predicted_label", "status"] if col in summary.columns]
    cols.extend(optional_cols)

    summary = summary[cols].sort_values("predicted_probability", ascending=False)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select diverse, high-risk credit cases from the test set."
    )
    parser.add_argument(
        "--test-path", default="data/test.csv", help="Path to the test set CSV."
    )
    parser.add_argument(
        "--predictions-path",
        default="results/mlp_test_predictions.csv",
        help="Path to predictions CSV (must contain 'predicted_probability').",
    )
    parser.add_argument(
        "--prob-threshold",
        type=float,
        default=0.6,
        help="Probability cutoff to mark a case as high risk.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of diverse cases to return.",
    )
    parser.add_argument(
        "--output-path",
        default="data/high_risk_cases.csv",
        help="Where to store the selected high-risk cases.",
    )
    parser.add_argument(
        "--summary-path",
        default="results/tables/selected_cases_summary.csv",
        help="Where to store the summary table.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for clustering-based selection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent

    test_path = resolve_path(base_dir, args.test_path)
    preds_path = resolve_path(base_dir, args.predictions_path)
    output_path = resolve_path(base_dir, args.output_path)
    summary_path = resolve_path(base_dir, args.summary_path)

    merged = load_and_merge(test_path, preds_path)
    high_risk = select_high_risk(merged, args.prob_threshold)
    selected = choose_diverse_cases(high_risk, args.top_k, args.random_state)
    summary = build_summary(high_risk, selected)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    selected.to_csv(output_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(
        f"Selected {len(selected)} cases from {len(high_risk)} "
        f"above probability {args.prob_threshold}."
    )
    print(f"Saved selected cases to: {output_path}")
    print(f"Saved summary table to: {summary_path}")


if __name__ == "__main__":
    main()