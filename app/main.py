"""FastAPI application for Telco Customer Churn Prediction."""

import logging
from contextlib import asynccontextmanager
import sklearn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import (
    CustomerInput,
    PredictionResponse,
    HealthResponse,
    RootResponse
)
from app.model_loader import (
    load_model,
    get_model,
    get_metadata,
    predict_churn
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("telco_churn_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager: loads model artifact once on startup."""
    logger.info("Initializing Telco Churn API lifespan...")
    try:
        load_model()
        logger.info("Model loaded successfully.")
    except Exception as exc:
        logger.error("Failed to load model on startup: %s", exc)
    yield
    logger.info("Shutting down Telco Churn API.")


app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description=(
        "Production-ready inference service for predicting customer churn on the "
        "Kaggle Telco Customer Churn dataset using a trained scikit-learn pipeline."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/",
    response_model=RootResponse,
    summary="API Information",
    tags=["General"]
)
def root() -> RootResponse:
    """Return general API information, status, and documentation endpoints."""
    return RootResponse(
        title="Telco Customer Churn Prediction API",
        description="ML service providing real-time customer churn predictions.",
        version="1.0.0",
        status="running",
        docs_url="/docs",
        health_url="/health",
        predict_url="/predict"
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    tags=["Monitoring"]
)
def health() -> HealthResponse:
    """Return the health status of the service and model loading state."""
    model = get_model()
    metadata = get_metadata()
    model_loaded = model is not None
    model_ver = metadata.get("model_version", "1.0.0") if model_loaded else "unavailable"

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_version=model_ver,
        scikit_learn_version=sklearn.__version__
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict Customer Churn",
    tags=["Inference"]
)
def predict(customer: CustomerInput) -> PredictionResponse:
    """
    Predict churn for a single customer based on contract, service, and demographic data.

    Returns:
        - **prediction**: "Yes" (customer likely to churn) or "No" (customer likely to stay)
        - **churn_probability**: Floating point probability [0.0 - 1.0]
        - **model_version**: Model artifact version used for prediction
    """
    if get_model() is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifact is not loaded or unavailable."
        )

    try:
        data = customer.model_dump()
        pred_label, churn_proba, version = predict_churn(data)
        return PredictionResponse(
            prediction=pred_label,
            churn_probability=churn_proba,
            model_version=version
        )
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err)
        )
    except Exception as exc:
        logger.error("Inference failure: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal prediction error occurred."
        )
