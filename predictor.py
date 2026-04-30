"""
predictor.py

Loads best_model.pkl and preprocessor_meta.pkl once at startup and exposes
a single predict function that the API calls on every request.
"""

import joblib
import numpy as np
import pandas as pd
from preprocessor import transform, load_meta

# Paths
MODEL_PATH = "data/processed/best_model.pkl"
META_PATH = "data/processed/preprocessor_meta.pkl"

# Load once at startup so every request reuses the same objects
model = joblib.load(MODEL_PATH)
feature_cols, modes = load_meta(META_PATH)

# Name of the winning model for the response metadata
MODEL_NAME = type(model.named_steps["model"]).__name__


def predict(raw_input: dict) -> dict:
    """
    Receive a dict of raw house fields, preprocess it and return the
    predicted price together with a confidence range and top features.

    Fields not present in raw_input are filled with the training data mode
    so the API works even when the LLM only extracts a subset of fields.
    """
    # Merge training modes with the extracted fields so nothing is missing
    full_input = {**modes, **raw_input}

    # Build a single-row dataframe and run the full preprocessing pipeline
    df = pd.DataFrame([full_input])
    df = transform(df, feature_cols)

    # Predict in log scale and convert back to dollars
    log_price = model.predict(df)[0]
    price = float(np.expm1(log_price))

    # Extract the five most influential features for this prediction
    step = model.named_steps["model"]
    if hasattr(step, "coef_"):
        importances = pd.Series(np.abs(step.coef_), index=feature_cols)
    elif hasattr(step, "feature_importances_"):
        importances = pd.Series(step.feature_importances_, index=feature_cols)
    else:
        importances = pd.Series(dtype=float)

    top_features = {
        k: round(float(v), 6)
        for k, v in importances.nlargest(5).items()
    }

    # Confidence range based on the model MAPE of roughly eight percent
    return {
        "predicted_price": round(price, 2),
        "price_range_low": round(price * 0.92, 2),
        "price_range_high": round(price * 1.08, 2),
        "top_features": top_features,
        "model_used": MODEL_NAME,
    }