import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

DATA_DIR = "./data"
INPUT_FILE = os.path.join(DATA_DIR, "cleaned_loan_data.csv")

TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
VAL_FILE = os.path.join(DATA_DIR, "val.csv")
TEST_FILE = os.path.join(DATA_DIR, "test.csv")

TARGET_COL = "status"
ID_COL = "id"

# Keep these consistent with clean_data.py
NUMERIC_COLS = ["term", "credit_score", "ltv", "dtir1",
                "loan_amount", "income", "property_value", "year"]

CATEGORICAL_COLS = ["loan_limit", "gender", "approv_in_adv", "loan_type", "loan_purpose",
                    "credit_worthiness", "open_credit", "business_or_commercial", "neg_ammortization",
                    "interest_only", "lump_sum_payment", "construction_type", "occupancy_type",
                    "secured_by", "total_units", "credit_type", "co-applicant_credit_type",
                    "age", "submission_of_application", "region", "security_type"]


def load_clean_data(path: str) -> pd.DataFrame:
    """Load the cleaned dataset produced by clean_data.py."""
    return pd.read_csv(path)


def build_preprocessor():
    """
    Build a ColumnTransformer that:
    - one-hot encodes categorical features
    - standardizes numerical features
    """
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)


    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_COLS),
            ("cat", categorical_transformer, CATEGORICAL_COLS),
        ]
    )

    return preprocessor


def train_val_test_split(X, y, random_state: int = 42):
    """
    Create 80/10/10 train/val/test split with stratification on the target.
    First split into train (80%) and temp (20%), then split temp into
    validation (10%) and test (10%).
    """
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=random_state
    )

    # 10% / 10% from the original data => 50/50 split of the remaining 20%
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def apply_smote(X_train, y_train, random_state: int = 42):
    """
    Apply SMOTE to the training set only to address class imbalance.
    Validation and test sets MUST NOT be oversampled.
    """
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled


def main():
    # ----------------------------------------------------------------
    # 1. Load cleaned data
    # ----------------------------------------------------------------
    df = load_clean_data(INPUT_FILE)

    # Separate target and features
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    # Drop identifier column from features
    if ID_COL in X.columns:
        X = X.drop(columns=[ID_COL])

    # Sanity check: ensure our numeric/categorical column lists match the data
    expected_cols = set(NUMERIC_COLS + CATEGORICAL_COLS)
    missing = expected_cols - set(X.columns)
    extra = set(X.columns) - expected_cols
    if missing:
        raise ValueError(f"Columns listed in NUMERIC/CATEGORICAL_COLS not in data: {missing}")
    if extra:
        # Not a hard error, but good to know
        print(f"Warning: there are extra columns not in NUMERIC/CATEGORICAL_COLS: {extra}")

    # ----------------------------------------------------------------
    # 2. Train/val/test split (on raw features)
    # ----------------------------------------------------------------
    X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = train_val_test_split(X, y)

    # ----------------------------------------------------------------
    # 3. Fit preprocessing on training data only; transform all splits
    # ----------------------------------------------------------------
    preprocessor = build_preprocessor()

    X_train_processed = preprocessor.fit_transform(X_train_raw)
    X_val_processed = preprocessor.transform(X_val_raw)
    X_test_processed = preprocessor.transform(X_test_raw)

    # Recover feature names for interpretability (after one-hot & scaling)
    ohe = preprocessor.named_transformers_["cat"]
    cat_feature_names = list(ohe.get_feature_names_out(CATEGORICAL_COLS))
    num_feature_names = NUMERIC_COLS
    all_feature_names = num_feature_names + cat_feature_names

    # Convert to DataFrames
    X_train_df = pd.DataFrame(X_train_processed,
                              columns=all_feature_names,
                              index=X_train_raw.index)
    X_val_df = pd.DataFrame(X_val_processed,
                            columns=all_feature_names,
                            index=X_val_raw.index)
    X_test_df = pd.DataFrame(X_test_processed,
                             columns=all_feature_names,
                             index=X_test_raw.index)

    # ----------------------------------------------------------------
    # 4. Apply SMOTE to training set only
    # ----------------------------------------------------------------
    X_train_balanced, y_train_balanced = apply_smote(X_train_df, y_train)

    # Convert balanced training data to DataFrame to keep column names
    X_train_balanced_df = pd.DataFrame(X_train_balanced, columns=all_feature_names)

    # ----------------------------------------------------------------
    # 5. Save train/val/test splits to CSV
    # ----------------------------------------------------------------
    os.makedirs(DATA_DIR, exist_ok=True)

    train_out = X_train_balanced_df.copy()
    train_out[TARGET_COL] = y_train_balanced
    train_out.to_csv(TRAIN_FILE, index=False)

    val_out = X_val_df.copy()
    val_out[TARGET_COL] = y_val
    val_out.to_csv(VAL_FILE, index=False)

    test_out = X_test_df.copy()
    test_out[TARGET_COL] = y_test
    test_out.to_csv(TEST_FILE, index=False)

    print(f"Saved train set with SMOTE to {TRAIN_FILE} (shape: {train_out.shape})")
    print(f"Saved validation set to {VAL_FILE} (shape: {val_out.shape})")
    print(f"Saved test set to {TEST_FILE} (shape: {test_out.shape})")


if __name__ == "__main__":
    main()
