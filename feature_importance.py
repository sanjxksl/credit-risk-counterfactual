import logging
from pathlib import Path
from typing import List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

MODEL_PATH = MODELS_DIR / "logistic_model.pkl"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.txt"
TOP_FEATURES_PATH = RESULTS_DIR / "top_features.csv"


def load_feature_names(path: Path) -> List[str]:
    """Load feature names from text file"""
    if not path.exists():
        logger.error(f"Feature names file not found at {path}")
        raise FileNotFoundError(f"Feature names file not found at {path}")

    logger.info(f"Loading feature names from {path}")
    with path.open() as f:
        names = [line.strip() for line in f if line.strip()]

    if not names:
        logger.error("No feature names found in feature_names.txt")
        raise ValueError("No feature names found in feature_names.txt")

    logger.info(f"Loaded {len(names)} feature names")
    return names


def compute_feature_importance(model, feature_names: List[str]) -> pd.DataFrame:
    """Compute feature importance from model coefficients"""
    logger.info("Computing feature importance from model coefficients")
    coefficients = model.coef_.ravel()

    if len(coefficients) != len(feature_names):
        error_msg = (
            f"Mismatch between coefficients ({len(coefficients)}) "
            f"and feature names ({len(feature_names)})"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
        }
    )
    df["abs_coefficient"] = df["coefficient"].abs()
    df["effect"] = np.where(df["coefficient"] >= 0, "increases_risk", "decreases_risk")
    df["odds_ratio"] = np.exp(np.clip(df["coefficient"], -50, 50))

    sorted_df = df.sort_values("abs_coefficient", ascending=False)
    logger.info(f"Computed importance for {len(sorted_df)} features")
    return sorted_df


def save_top_features(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Save top N features to CSV"""
    logger.info(f"Saving top {top_n} features to {TOP_FEATURES_PATH}")
    top_df = df.head(top_n).reset_index(drop=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    top_df.to_csv(TOP_FEATURES_PATH, index=False)
    logger.info(f"Successfully saved top features to {TOP_FEATURES_PATH}")
    return top_df


def plot_feature_importance(df: pd.DataFrame, top_n: int) -> Path:
    """Generate and save feature importance plot"""
    logger.info(f"Generating feature importance plot for top {top_n} features")
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
    logger.info(f"Successfully saved plot to {output_path}")
    return output_path


def main(top_n: int = 10) -> None:
    """Main execution function"""
    logger.info("Starting feature importance analysis")
    logger.info(f"Loading model from {MODEL_PATH}")

    try:
        model = joblib.load(MODEL_PATH)
        feature_names = load_feature_names(FEATURE_NAMES_PATH)
        importance_df = compute_feature_importance(model, feature_names)

        top_features = save_top_features(importance_df, top_n)
        plot_feature_importance(importance_df, top_n)

        logger.info("=" * 60)
        logger.info(f"TOP {top_n} FEATURES")
        logger.info("=" * 60)
        print(top_features[["feature", "coefficient", "abs_coefficient", "effect"]])
        logger.info("=" * 60)
        logger.info("Feature importance analysis completed successfully")

    except Exception as e:
        logger.error(f"Error during feature importance analysis: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
