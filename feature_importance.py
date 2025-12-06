import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

MODEL_PATH = MODELS_DIR / "logistic_model.pkl"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.txt"
TOP_FEATURES_PATH = RESULTS_DIR / "top_features.csv"


def load_feature_names(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Feature names file not found at {path}")
    with path.open() as f:
        names = [line.strip() for line in f if line.strip()]
    if not names:
        raise ValueError("No feature names found in feature_names.txt")
    return names


def compute_feature_importance(model, feature_names: list[str]) -> pd.DataFrame:
    coefficients = model.coef_.ravel()
    if len(coefficients) != len(feature_names):
        raise ValueError(
            f"Mismatch between coefficients ({len(coefficients)}) "
            f"and feature names ({len(feature_names)})"
        )

    df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
        }
    )
    df["abs_coefficient"] = df["coefficient"].abs()
    df["effect"] = np.where(df["coefficient"] >= 0, "increases_risk", "decreases_risk")
    df["odds_ratio"] = np.exp(np.clip(df["coefficient"], -50, 50))
    return df.sort_values("abs_coefficient", ascending=False)


def save_top_features(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    top_df = df.head(top_n).reset_index(drop=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    top_df.to_csv(TOP_FEATURES_PATH, index=False)
    return top_df


def plot_feature_importance(df: pd.DataFrame, top_n: int) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    top_df = df.head(top_n).iloc[::-1]  # reverse for barh ordering
    colors = top_df["coefficient"].apply(lambda x: "#d62728" if x >= 0 else "#1f77b4")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_df["feature"], top_df["coefficient"], color=colors)
    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("Coefficient (log-odds)")
    ax.set_ylabel("Feature")
    ax.set_title(f"Top {top_n} Feature Importances (Logistic Regression)")
    for idx, value in enumerate(top_df["coefficient"]):
        ax.text(
            value,
            idx,
            f"{value:.2f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=9,
        )
    plt.tight_layout()

    output_path = FIGURES_DIR / "feature_importance.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main(top_n: int = 10) -> None:
    model = joblib.load(MODEL_PATH)
    feature_names = load_feature_names(FEATURE_NAMES_PATH)
    importance_df = compute_feature_importance(model, feature_names)

    top_features = save_top_features(importance_df, top_n)
    plot_feature_importance(importance_df, top_n)

    print(f"Saved top {top_n} features to {TOP_FEATURES_PATH}")
    print(top_features[["feature", "coefficient", "abs_coefficient", "effect"]])


if __name__ == "__main__":
    main()
