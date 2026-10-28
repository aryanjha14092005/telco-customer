"""Model loader and inference utilities."""

import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import joblib
import pandas as pd
import sklearn

logger = logging.getLogger(__name__)

# Feature order strictly matching trained pipeline feature_names_in_
FEATURE_NAMES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges"
]

_model: Optional[Any] = None
_metadata: Dict[str, Any] = {}


def find_model_path() -> Path:
    """Locate model.joblib in candidate project directories."""
    candidates = [
        Path(os.environ.get("MODEL_PATH", "")) if os.environ.get("MODEL_PATH") else None,
        Path("model/model.joblib"),
        Path("model.joblib"),
        Path(__file__).resolve().parent.parent / "model" / "model.joblib",
        Path(__file__).resolve().parent.parent / "model.joblib",
    ]
    for candidate in candidates:
        if candidate and candidate.exists() and candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Trained model artifact 'model.joblib' could not be found. "
        "Please place it at 'model/model.joblib' or set the MODEL_PATH environment variable."
    )


def find_metadata_path() -> Optional[Path]:
    """Locate metadata.json in candidate project directories."""
    candidates = [
        Path("model/metadata.json"),
        Path(__file__).resolve().parent.parent / "model" / "metadata.json",
    ]
    for candidate in candidates:
        if candidate and candidate.exists() and candidate.is_file():
            return candidate.resolve()
    return None


def load_model() -> Any:
    """Load the model artifact and metadata into memory."""
    global _model, _metadata
    model_path = find_model_path()
    logger.info("Loading model artifact from: %s", model_path)
    _model = joblib.load(model_path)

    metadata_path = find_metadata_path()
    if metadata_path:
        with open(metadata_path, "r", encoding="utf-8") as f:
            _metadata = json.load(f)
    else:
        _metadata = {
            "model_name": "Telco Churn Model",
            "model_version": "1.0.0",
            "framework": "scikit-learn",
            "scikit_learn_version": getattr(_model, "_sklearn_version", sklearn.__version__),
        }
    return _model


def get_model() -> Optional[Any]:
    """Get the cached model instance."""
    return _model


def get_metadata() -> Dict[str, Any]:
    """Get loaded model metadata."""
    return _metadata


def predict_churn(customer_data: Dict[str, Any]) -> Tuple[str, float, str]:
    """
    Run inference on a single customer feature dictionary.

    Returns:
        Tuple of (prediction_label, churn_probability, model_version)
    """
    if _model is None:
        raise RuntimeError("Model is not loaded. Call load_model() first.")

    # Exclude customerID if present
    data = {k: v for k, v in customer_data.items() if k != "customerID"}

    # Validate all required features are present
    missing_features = [f for f in FEATURE_NAMES if f not in data]
    if missing_features:
        raise ValueError(f"Missing required feature(s): {missing_features}")

    # Build DataFrame with strict feature order
    df = pd.DataFrame([{feature: data[feature] for feature in FEATURE_NAMES}])

    # Execute prediction
    pred_raw = _model.predict(df)[0]

    # Calculate probability if model supports predict_proba
    if hasattr(_model, "predict_proba"):
        proba_array = _model.predict_proba(df)[0]
        # classes_ is [0, 1] where 1 is churn ("Yes")
        churn_proba = float(proba_array[1])
    else:
        churn_proba = 1.0 if pred_raw == 1 else 0.0

    # Map raw class (0 or 1) to label ("No" or "Yes")
    label_map = _metadata.get("label_mapping", {"0": "No", "1": "Yes", 0: "No", 1: "Yes"})
    pred_label = label_map.get(str(pred_raw), label_map.get(pred_raw, "Yes" if pred_raw == 1 else "No"))

    version = _metadata.get("model_version", "1.0.0")
    return pred_label, round(churn_proba, 4), version
