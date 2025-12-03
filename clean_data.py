import pandas as pd
import numpy as np

NUMERIC_COLS = ["term", "credit_score", "ltv", "dtir1", "loan_amount", "income", "property_value"]
CATEGORICAL_COLS = ["loan_limit", "gender", "approv_in_adv", "loan_type", "loan_purpose",
                    "credit_worthiness", "open_credit", "business_or_commercial", "neg_ammortization",
                    "interest_only", "lump_sum_payment", "construction_type", "occupancy_type",
                    "secured_by", "total_units", "credit_type", "co-applicant_credit_type",
                    "age", "submission_of_application", "region", "security_type"] 
LOG_TRANSFORM_COLS = ["loan_amount", "income", "property_value"]
DROP_COLS = ["rate_of_interest", "interest_rate_spread", "upfront_charges"]

# First, load the data
def load_data(path):
    return pd.read_csv(path)

# Next, we will impute the missing values depending on the situation
def impute_missing_values(df):

    # For all of the numeric columns except for the ones to drop, we replace missing values with the median
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # For the categotical columns, we will replace all missing values with the mode
    for col in CATEGORICAL_COLS:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df

# Using log transformations on a few of the numerical variables that are positively skewed
def log_transform(df):
    for col in LOG_TRANSFORM_COLS:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: np.log1p(x))
    return df

def clean_data(path):
    df = load_data(path)

    # Renaming some columns for consistency
    df = df.rename(columns={'ID': 'id', 'Gender': 'gender', 'Credit_Worthiness': 'credit_worthiness', 
                        'Interest_rate_spread': 'interest_rate_spread', 'Upfront_charges': 'upfront_charges', 
                       'Neg_ammortization': 'neg_ammortization', 'Secured_by': 'secured_by', 
                       'Credit_Score': 'credit_score', 'LTV': 'ltv', 'Region': 'region',
                       'Security_Type': 'security_type', 'Status': 'status'})

    # Dropping features that are indicative of the target variable values
    df = df.drop(columns=[col for col in DROP_COLS if col in df.columns])

    df = impute_missing_values(df)
    df = log_transform(df)
    return df

if __name__ == "__main__":
    input_path = "./data/Loan_Default.csv"
    output_path = "./data/cleaned_loan_data.csv"

    cleaned_df = clean_data(input_path)
    cleaned_df.to_csv(output_path, index=False)

    print("Cleaned data:", output_path)
