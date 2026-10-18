"""FastAPI app: loads the trained model and serves fare predictions."""
import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"
STATIC_DIR = Path(__file__).resolve().parent / "static"

# Load once when the server starts (not on every request).
MODEL = joblib.load(MODEL_DIR / "model.joblib")
with open(MODEL_DIR / "columns.json", encoding="utf-8") as f:
    META = json.load(f)
FEATURE_COLUMNS = META["feature_columns"]

app = FastAPI(
    title="Boston Cab Fare Estimator",
    description="Estimate Uber/Lyft fares in Boston (model trained on Nov-Dec 2018 data).",
    version="1.0.0",
)


class PredictRequest(BaseModel):
    source: str
    destination: str
    ride_type: str = Field(description="For example UberX, Lyft, Black")
    distance: float = Field(gt=0, description="Trip distance in miles")
    surge_multiplier: float = Field(default=1.0, description="1.0 means no surge")

    model_config = {
        "json_schema_extra": {
            "example": {
                "source": "Back Bay",
                "destination": "Fenway",
                "ride_type": "UberX",
                "distance": 2.0,
                "surge_multiplier": 1.0,
            }
        }
    }


class PredictResponse(BaseModel):
    predicted_price_usd: float
    cab_type: str
    ride_type: str


def check_allowed(value, allowed, label):
    """Reject values the model has never seen during training."""
    if value not in allowed:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown {label} '{value}'. Allowed: {sorted(allowed)}",
        )

@app.get("/", include_in_schema=False)
def home():
    """Serve the web page."""
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/health")
def health():
    return {"status": "ok", "model_metrics": META["metrics"]}


@app.get("/options")
def options():
    """Everything the form needs to fill its dropdowns."""
    return {
        "sources": META["sources"],
        "destinations": META["destinations"],
        "ride_types": META["ride_types"],
        "distance_range": META["distance_range"],
        "surge_values": META["surge_values"],
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    check_allowed(req.source, META["sources"], "source")
    check_allowed(req.destination, META["destinations"], "destination")
    check_allowed(req.ride_type, META["ride_types"], "ride_type")
    check_allowed(req.surge_multiplier, META["surge_values"], "surge_multiplier")

    low, high = META["distance_range"]
    if not low <= req.distance <= high:
        raise HTTPException(
            status_code=422,
            detail=f"distance must be between {low} and {high} miles (the range the model was trained on).",
        )
    if req.source == req.destination:
        raise HTTPException(status_code=422, detail="source and destination must be different.")

    # Build one row with the exact same columns, in the exact same order, as training.
    row = dict.fromkeys(FEATURE_COLUMNS, 0)
    row["distance"] = req.distance
    row["surge_multiplier"] = req.surge_multiplier
    row[f"source_{req.source}"] = 1
    row[f"destination_{req.destination}"] = 1
    row[f"name_{req.ride_type}"] = 1
    X = pd.DataFrame([row], columns=FEATURE_COLUMNS)

    price = float(MODEL.predict(X)[0])
    return PredictResponse(
        predicted_price_usd=round(price, 2),
        cab_type=META["ride_types"][req.ride_type],
        ride_type=req.ride_type,
    )