"""
Input and output shapes for the API.

Pydantic validates incoming JSON automatically — if a required field is
missing or a value is outside the allowed set, FastAPI returns a 422 error
before your code even runs. No manual if/else validation needed.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class LoanApplication(BaseModel):
    """Raw loan application as submitted by a lender or applicant.

    These are the original values before any preprocessing. The pipeline
    applies winsorization, log-transforms, and scaling internally.
    """

    # --- Numeric ---
    year: int = Field(default=2019)
    loan_amount: float = Field(..., gt=0, description="Loan amount in dollars")
    term: float = Field(..., gt=0, description="Loan term in months (e.g. 360 = 30 years)")
    property_value: float = Field(..., gt=0, description="Appraised property value in dollars")
    income: float = Field(..., ge=0, description="Annual income in dollars")
    credit_score: int = Field(..., ge=500, le=900)
    ltv: float = Field(..., gt=0, description="Loan-to-value ratio, e.g. 80.0 means 80%")
    dtir1: float = Field(..., gt=0, description="Debt-to-income ratio, e.g. 36.0 means 36%")

    # --- Categorical ---
    loan_limit: Literal["cf", "ncf"]
    gender: Literal["Male", "Female", "Joint", "Sex Not Available"]
    approv_in_adv: Literal["pre", "nopre"]
    loan_type: Literal["type1", "type2", "type3"]
    loan_purpose: Literal["p1", "p2", "p3", "p4"]
    credit_worthiness: Literal["l1", "l2"]
    open_credit: Literal["nopc", "opc"]
    business_or_commercial: Literal["nob/c", "b/c"]
    neg_ammortization: Literal["not_neg", "neg_amm"]
    interest_only: Literal["not_int", "int_only"]
    lump_sum_payment: Literal["not_lpsm", "lpsm"]
    occupancy_type: Literal["pr", "sr", "ir"]
    total_units: Literal["1U", "2U", "3U", "4U"]
    age: Literal["<25", "25-34", "35-44", "45-54", "55-64", "65-74", ">74"]
    submission_of_application: Literal["to_inst", "not_inst"]
    region: Literal["North", "North-East", "central", "south"]


class CounterfactualScenario(BaseModel):
    """One actionable scenario: what to change to avoid predicted default."""

    changes: dict  # e.g. {"loan_amount": 280000, "dtir1": 32.0}
    predicted_probability_after: float
    description: str  # human-readable, e.g. "Reduce loan amount by $40,000"


class PredictionResponse(BaseModel):
    request_id: str
    default_probability: float
    predicted_label: int  # 0 = no default, 1 = default
    risk_level: str       # "Low" / "Medium" / "High"
    counterfactuals: Optional[List[CounterfactualScenario]] = None
    counterfactuals_available: bool = False
    message: str


class DriftReport(BaseModel):
    """Summary of the Evidently drift analysis."""

    num_requests_analysed: int
    share_drifted_features: float
    drifted_features: List[str]
    predicted_default_rate: float
    reference_default_rate: float
    retrain_recommended: bool
    trigger_reasons: List[str]
