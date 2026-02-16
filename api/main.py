"""CricOracle 2026 — FastAPI Application.

Run with: uvicorn api.main:app --reload --port 8000
Swagger UI: http://localhost:8000/docs
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import predict, squad

app = FastAPI(
    title="CricOracle 2026 API",
    description=(
        "T20 World Cup 2026 Prediction Platform. "
        "Match outcome predictions, score forecasts, and squad optimisation."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS (allow frontend to call API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(predict.router, prefix="/predict", tags=["Predictions"])
app.include_router(squad.router, prefix="/squad", tags=["Squad"])


@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint."""
    return {
        "status": "online",
        "project": "CricOracle 2026",
        "version": "0.1.0",
        "endpoints": ["/predict/match", "/predict/score", "/squad/optimise", "/docs"],
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "models_loaded": False,  # TODO: Check if models are loaded
        "data_available": False,  # TODO: Check if feature data exists
    }
