"""Train the cab fare model and save everything the API needs into model/."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "cab_rides.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

# 1) Load data
rides = pd.read_csv(DATA_PATH)
print("Rows loaded:", len(rides))

# Some rides have no price (we cannot learn from them), so drop them.
rides = rides.dropna(subset=["price"]).reset_index(drop=True)
print("Rows with a price:", len(rides))

# 2) Choose features 
# Inputs the user can give: source, destination, ride type, distance, surge.
CATEGORICAL = ["source", "destination", "name"]
NUMERIC = ["distance", "surge_multiplier"]
TARGET = "price"

X = rides[CATEGORICAL + NUMERIC]
y = rides[TARGET]

# Text columns -> 0/1 columns (one-hot encoding), because the model needs numbers.
X = pd.get_dummies(X, columns=CATEGORICAL, dtype=int)
feature_columns = X.columns.tolist()

# 3) Train / test split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.7, random_state=1
)

# 4) Train 
model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

# 5) Evaluate on data the model has never seen 
pred = model.predict(X_test)
rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
r2 = float(r2_score(y_test, pred))
print(f"Test RMSE: {rmse:.3f}   R2: {r2:.4f}")

# 6) Save what the API needs 
joblib.dump(model, MODEL_DIR / "model.joblib")

ride_types = (
    rides.groupby("name")["cab_type"].first().sort_index().to_dict()
)  # e.g. {"Lux": "Lyft", "UberX": "Uber", ...}

meta = {
    "feature_columns": feature_columns,
    "sources": sorted(rides["source"].unique().tolist()),
    "destinations": sorted(rides["destination"].unique().tolist()),
    "ride_types": ride_types,
    "distance_range": [float(rides["distance"].min()), float(rides["distance"].max())],
    "surge_values": sorted(rides["surge_multiplier"].unique().tolist()),
    "metrics": {"rmse": round(rmse, 3), "r2": round(r2, 4)},
}
with open(MODEL_DIR / "columns.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print("Saved: model/model.joblib and model/columns.json")