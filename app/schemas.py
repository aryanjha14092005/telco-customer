"""Pydantic schemas for request and response validation."""

from typing import Optional, Literal
from pydantic import BaseModel, Field, ConfigDict, field_validator


class CustomerInput(BaseModel):
    """Input features required for customer churn prediction."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": 29.85,
                "customerID": "7590-VHVEG"
            }
        }
    )

    # Identifiers (optional, excluded during inference)
    customerID: Optional[str] = Field(
        default=None,
        description="Optional customer identifier. Excluded from model features."
    )

    # Demographic features
    gender: Literal["Male", "Female"] = Field(
        ..., description="Customer gender (Male or Female)"
    )
    SeniorCitizen: int = Field(
        ..., ge=0, le=1, description="Whether customer is a senior citizen (1 or 0)"
    )
    Partner: Literal["Yes", "No"] = Field(
        ..., description="Whether customer has a partner (Yes or No)"
    )
    Dependents: Literal["Yes", "No"] = Field(
        ..., description="Whether customer has dependents (Yes or No)"
    )

    # Account and Service features
    tenure: float = Field(
        ..., ge=0, description="Number of months customer has stayed with company"
    )
    PhoneService: Literal["Yes", "No"] = Field(
        ..., description="Whether customer has phone service (Yes or No)"
    )
    MultipleLines: Literal["Yes", "No", "No phone service"] = Field(
        ..., description="Whether customer has multiple lines"
    )
    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(
        ..., description="Customer internet service provider"
    )
    OnlineSecurity: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Whether customer has online security"
    )
    OnlineBackup: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Whether customer has online backup"
    )
    DeviceProtection: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Whether customer has device protection"
    )
    TechSupport: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Whether customer has tech support"
    )
    StreamingTV: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Whether customer has streaming TV"
    )
    StreamingMovies: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Whether customer has streaming movies"
    )
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        ..., description="The contract term of the customer"
    )
    PaperlessBilling: Literal["Yes", "No"] = Field(
        ..., description="Whether customer has paperless billing"
    )
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ] = Field(..., description="Customer payment method")
    MonthlyCharges: float = Field(
        ..., ge=0, description="The amount charged to the customer monthly"
    )
    TotalCharges: float = Field(
        ..., ge=0, description="The total amount charged to the customer"
    )

    @field_validator("TotalCharges", mode="before")
    @classmethod
    def parse_total_charges(cls, v):
        if isinstance(v, str):
            v_strip = v.strip()
            if v_strip == "":
                return 0.0
            try:
                return float(v_strip)
            except ValueError:
                raise ValueError("TotalCharges must be a valid number")
        return v


class PredictionResponse(BaseModel):
    """Response schema for customer churn prediction."""

    prediction: Literal["Yes", "No"] = Field(
        ..., description="Predicted churn label (Yes = Churn, No = Retain)"
    )
    churn_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Estimated probability of churn"
    )
    model_version: str = Field(
        ..., description="Version of the trained model artifact"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prediction": "Yes",
                "churn_probability": 0.8021,
                "model_version": "1.0.0"
            }
        }
    )


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str = Field(..., description="API operational status")
    model_loaded: bool = Field(..., description="Whether model artifact is loaded")
    model_version: str = Field(..., description="Loaded model version")
    scikit_learn_version: str = Field(..., description="Runtime scikit-learn version")


class RootResponse(BaseModel):
    """Root info response schema."""

    title: str
    description: str
    version: str
    status: str
    docs_url: str
    health_url: str
    predict_url: str
