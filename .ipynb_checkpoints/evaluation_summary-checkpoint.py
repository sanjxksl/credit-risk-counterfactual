# +
import json
from pathlib import Path
import pandas as pd

# ---------- Path Setup ----------
# Use Path(__file__) when run as a script
# If running inside Jupyter Notebook, replace with: BASE_DIR = Path.cwd()
BASE_DIR = Path(__file__).resolve().parent

RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)  # Create tables folder if missing


# ---------- Helper Function: Load metrics from multiple possible locations ----------
def load_metrics(model_name: str) -> dict:
    """
    Try loading metrics file from:
    1) results/figures/{model}_metrics.json
    2) results/{model}_metrics.json
    """
    candidate_paths = [
        FIGURES_DIR / f"{model_name}_metrics.json",
        RESULTS_DIR / f"{model_name}_metrics.json",
    ]

    for path in candidate_paths:
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)

    # If not found in any location, raise error
    raise FileNotFoundError(
        f"Metrics file for '{model_name}' was not found in any of: "
        + ", ".join(str(p) for p in candidate_paths)
    )



def build_tables(logistic_metrics: dict, mlp_metrics: dict) -> None:
    """
    Build and export:
    1) Model comparison table (test set only)
    2) Performance metrics table (validation + test)
    """

    # Extract validation and test metrics for both models
    log_val = logistic_metrics["validation_metrics"]
    log_test = logistic_metrics["test_metrics"]

    mlp_val = mlp_metrics["validation_metrics"]
    mlp_test = mlp_metrics["test_metrics"]

    # ---------- 1) Model Comparison Table (Test set) ----------
    # One row per model, columns = metrics on TEST set
    comparison_df = pd.DataFrame(
        [
            {
                "Model": "Logistic Regression",
                "AUC_ROC": log_test["auc_roc"],
                "AUC_PR": log_test["auc_pr"],
                "Brier": log_test["brier_score"],
            },
            {
                "Model": "MLP",
                "AUC_ROC": mlp_test["auc_roc"],
                "AUC_PR": mlp_test["auc_pr"],
                "Brier": mlp_test["brier_score"],
            },
        ]
    )

    comparison_path = TABLES_DIR / "model_comparison_table.tex"
    comparison_df.to_latex(
        comparison_path,
        index=False,
        float_format="%.4f"
    )

    # ---------- 2) Performance Metrics Table (Val + Test) ----------
    # Rows = (dataset, metric), columns = Logistic vs MLP
    metrics_df = pd.DataFrame(
        [
            {
                "Dataset": "Validation",
                "Metric": "AUC-ROC",
                "Logistic": log_val["auc_roc"],
                "MLP": mlp_val["auc_roc"],
            },
            {
                "Dataset": "Validation",
                "Metric": "AUC-PR",
                "Logistic": log_val["auc_pr"],
                "MLP": mlp_val["auc_pr"],
            },
            {
                "Dataset": "Validation",
                "Metric": "Brier score",
                "Logistic": log_val["brier_score"],
                "MLP": mlp_val["brier_score"],
            },
            {
                "Dataset": "Test",
                "Metric": "AUC-ROC",
                "Logistic": log_test["auc_roc"],
                "MLP": mlp_test["auc_roc"],
            },
            {
                "Dataset": "Test",
                "Metric": "AUC-PR",
                "Logistic": log_test["auc_pr"],
                "MLP": mlp_test["auc_pr"],
            },
            {
                "Dataset": "Test",
                "Metric": "Brier score",
                "Logistic": log_test["brier_score"],
                "MLP": mlp_test["brier_score"],
            },
        ]
    )

    metrics_path = TABLES_DIR / "metrics_table.tex"
    metrics_df.to_latex(
        metrics_path,
        index=False,
        float_format="%.4f"
    )

    print(f"Saved model comparison table to: {comparison_path}")
    print(f"Saved metrics table to: {metrics_path}")

    
def build_figures_list() -> None:
    """
    Scan results/figures/ and generate a Markdown list of figures
    in docs/figures_list.md
    """
    # Collect all image files in results/figures
    figures = sorted([
        f for f in FIGURES_DIR.iterdir()
        if f.suffix.lower() in [".png", ".jpg", ".jpeg"]
    ])

    output_path = BASE_DIR / "docs" / "figures_list.md"

    with open(output_path, "w") as f:
        f.write("# Figures List for Report\n\n")
        for i, fig in enumerate(figures, start=1):
            # fig.stem-a file name without a suffix
            f.write(f"{i}. **{fig.stem}** — results/figures/{fig.name}\n")

    print(f"Saved figures list to: {output_path}")
    
    
    
def main():
    # Load metrics for logistic and MLP
    logistic_metrics = load_metrics("logistic")
    mlp_metrics = load_metrics("mlp")

    print("Logistic metrics:", logistic_metrics)
    print("MLP metrics:", mlp_metrics)
    
    # Build and export LaTeX tables
    build_tables(logistic_metrics, mlp_metrics)
    build_figures_list()
    
    
if __name__ == "__main__":
    main()
# -









