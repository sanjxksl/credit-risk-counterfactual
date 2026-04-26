"""
FastAPI application — the HTTP layer that ties everything together.

What FastAPI does:
  - Reads incoming JSON, validates it against LoanApplication (Pydantic rejects
    bad data automatically before your code runs)
  - Calls the pipeline, counterfactual engine, or monitoring layer
  - Serializes the result back to JSON

Endpoints:
  POST /predict              — fast path: probability + risk level only
  POST /predict/explain      — slow path: probability + DiCE counterfactuals
  GET  /ping                 — health check (required by SageMaker)
  POST /invocations          — SageMaker's expected prediction route (mirrors /predict)
  GET  /monitoring/drift     — run Evidently drift report on buffered requests
"""

import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException

from .counterfactuals import CounterfactualEngine
from .monitoring import DriftMonitor, MIN_REQUESTS_FOR_DRIFT, RequestLog, RetrainingTrigger
from .pipeline import InferencePipeline
from .schemas import DriftReport, LoanApplication, PredictionResponse

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/opt/ml/model"))

# Shared state — loaded once at startup, reused for every request
_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Everything in the 'before yield' block runs once when the server starts.
    This is where you load heavy objects (models, reference data) into memory.
    After 'yield', the server is live. After shutdown, any cleanup goes here.
    """
    pipeline = InferencePipeline()
    request_log = RequestLog()
    drift_monitor = DriftMonitor(MODEL_DIR / "train.csv")
    retrain_trigger = RetrainingTrigger()

    # CounterfactualEngine is heavier (loads DiCE + full training data)
    # so we load it lazily on first /predict/explain request
    _state.update(
        pipeline=pipeline,
        request_log=request_log,
        drift_monitor=drift_monitor,
        retrain_trigger=retrain_trigger,
        cf_engine=None,
    )
    yield
    # Cleanup (if needed) goes here


app = FastAPI(
    title="Credit Risk API",
    description="Predicts loan default probability and generates actionable counterfactuals.",
    version="1.0.0",
    lifespan=lifespan,
)


def _run_prediction(application: LoanApplication) -> PredictionResponse:
    """Core logic shared by /predict and /invocations."""
    pipeline: InferencePipeline = _state["pipeline"]
    raw = application.model_dump()

    prob, label = pipeline.predict(raw)

    # Log the request for drift monitoring (uses preprocessed features)
    _state["request_log"].log(raw, prob)

    return PredictionResponse(
        request_id=str(uuid.uuid4()),
        default_probability=round(prob, 4),
        predicted_label=label,
        risk_level=pipeline.risk_level(prob),
        counterfactuals_available=False,
        message=(
            "High default risk detected. Call /predict/explain for actionable recommendations."
            if label == 1
            else "Application does not indicate high default risk."
        ),
    )


@app.get("/ping", status_code=200)
def health_check():
    """SageMaker calls this every 30 seconds to confirm the container is alive."""
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict(application: LoanApplication):
    """
    Fast prediction endpoint (~5ms).
    Returns default probability and risk level. Does not run counterfactuals.
    """
    return _run_prediction(application)


@app.post("/invocations", response_model=PredictionResponse)
def invocations(application: LoanApplication):
    """
    SageMaker routes all requests to /invocations.
    This is identical to /predict — both exist so the API is usable
    both directly and via a SageMaker endpoint.
    """
    return _run_prediction(application)


@app.post("/predict/explain", response_model=PredictionResponse)
def predict_with_explanation(application: LoanApplication):
    """
    Slow prediction endpoint (~2-5 seconds).
    Runs DiCE counterfactuals for declined applications.
    Only generates counterfactuals if the application is predicted to default.
    """
    pipeline: InferencePipeline = _state["pipeline"]
    raw = application.model_dump()

    prob, label = pipeline.predict(raw)
    _state["request_log"].log(raw, prob)

    counterfactuals = None
    cf_available = False

    if label == 1:
        # Lazy-load the counterfactual engine on first explain request
        if _state["cf_engine"] is None:
            _state["cf_engine"] = CounterfactualEngine(
                pipeline, MODEL_DIR / "train.csv"
            )

        preprocessed = pipeline.preprocess(raw)
        import pandas as pd
        preprocessed_df = pd.DataFrame(
            preprocessed,
            columns=pipeline.preprocessor.get_feature_names_out()
            if hasattr(pipeline.preprocessor, "get_feature_names_out")
            else None,
        )
        preprocessed_df["status"] = label  # DiCE needs the target column present

        raw_scenarios = _state["cf_engine"].generate(preprocessed_df)
        if raw_scenarios:
            from .schemas import CounterfactualScenario
            counterfactuals = [CounterfactualScenario(**s) for s in raw_scenarios]
            cf_available = True

    if label == 1 and not cf_available:
        message = (
            "High default risk. No feasible loan restructuring found — "
            "risk is driven by the borrower's financial profile, not loan structure."
        )
    elif label == 1 and cf_available:
        message = (
            f"High default risk. {len(counterfactuals)} restructuring scenario(s) "
            "found that would change the predicted outcome."
        )
    else:
        message = "Application does not indicate high default risk."

    return PredictionResponse(
        request_id=str(uuid.uuid4()),
        default_probability=round(prob, 4),
        predicted_label=label,
        risk_level=pipeline.risk_level(prob),
        counterfactuals=counterfactuals,
        counterfactuals_available=cf_available,
        message=message,
    )


@app.get("/monitoring/drift", response_model=DriftReport)
def drift_report():
    """
    Runs Evidently on all buffered requests since the last check.
    Compares the live feature distribution to the training reference.

    Call this on a schedule (e.g. daily via a CloudWatch cron) rather than
    on every request. After running, the buffer is cleared so the next report
    reflects only new traffic.
    """
    log: RequestLog = _state["request_log"]

    if log.count < MIN_REQUESTS_FOR_DRIFT:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least {MIN_REQUESTS_FOR_DRIFT} requests to run drift analysis. "
            f"Currently have {log.count}.",
        )

    current_df = log.to_dataframe()
    drift_result = _state["drift_monitor"].run(current_df)
    result = _state["retrain_trigger"].evaluate(drift_result)

    log.clear()  # reset buffer after reporting

    return DriftReport(**result)
