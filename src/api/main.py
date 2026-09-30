"""FastAPI scoring service."""
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from src.api.model_loader import get_model_bundle
from src.api.predictor import Predictor
from src.api.schemas import ApplicantInput, PredictionResponse, HealthResponse


# Global predictor — initialized at startup
predictor: Predictor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts at startup, release at shutdown."""
    global predictor
    bundle = get_model_bundle()
    predictor = Predictor(bundle)
    print("Model loaded. Ready to serve.")
    yield
    predictor = None


app = FastAPI(
    title="Credit Risk Decision Engine",
    description="Score loan applications and return decisions with reason codes.",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health", response_model=HealthResponse)
def health():
    """Liveness check."""
    return HealthResponse(
        status="ok",
        model_loaded=predictor is not None
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: ApplicantInput):
    """Score a single applicant."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        result = predictor.predict(payload.features)
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))