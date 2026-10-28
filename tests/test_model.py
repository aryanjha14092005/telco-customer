"""Unit tests for model artifact loading and inference functionality."""

import pytest
from app.model_loader import (
    load_model,
    get_model,
    get_metadata,
    predict_churn,
    FEATURE_NAMES
)


@pytest.fixture(scope="module", autouse=True)
def setup_model():
    """Ensure the model is loaded before running tests."""
    return load_model()


def test_model_artifact_loaded():
    """Verify that the model artifact loads and is not None."""
    model = get_model()
    assert model is not None, "Model failed to load"
    assert hasattr(model, "predict"), "Model does not have predict method"
    assert hasattr(model, "predict_proba"), "Model does not support predict_proba"


def test_model_feature_schema():
    """Verify expected feature names match trained pipeline expectations."""
    model = get_model()
    assert hasattr(model, "feature_names_in_")
    assert list(model.feature_names_in_) == FEATURE_NAMES
    assert len(FEATURE_NAMES) == 19


def test_predict_churn_high_risk():
    """Test inference on a customer with characteristics of high churn."""
    customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.5,
        "TotalCharges": 85.5,
        "customerID": "HIGH-CHURN-1"
    }

    pred_label, churn_proba, version = predict_churn(customer)
    assert pred_label in ["Yes", "No"]
    assert 0.0 <= churn_proba <= 1.0
    assert isinstance(version, str)
    # Month-to-month fiber optic with 1 month tenure should have high churn probability
    assert churn_proba > 0.5
    assert pred_label == "Yes"


def test_predict_churn_loyal_customer():
    """Test inference on a long-tenure customer with two-year contract."""
    customer = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 72,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 60.0,
        "TotalCharges": 4320.0
    }

    pred_label, churn_proba, version = predict_churn(customer)
    assert pred_label in ["Yes", "No"]
    assert 0.0 <= churn_proba <= 1.0
    # Long-term two year contract customer should have low churn probability
    assert churn_proba < 0.5
    assert pred_label == "No"


def test_predict_missing_feature():
    """Verify that omitting a required feature raises a ValueError."""
    incomplete_customer = {
        "gender": "Male",
        "SeniorCitizen": 0,
        # Missing Partner, Dependents, tenure, etc.
        "MonthlyCharges": 50.0,
        "TotalCharges": 100.0
    }
    with pytest.raises(ValueError, match="Missing required feature"):
        predict_churn(incomplete_customer)


def test_metadata_contents():
    """Verify metadata.json contains expected configuration."""
    metadata = get_metadata()
    assert metadata.get("framework") == "scikit-learn"
    assert metadata.get("scikit_learn_version") == "1.6.1"
    assert "feature_names" in metadata
    assert len(metadata["feature_names"]) == 19
