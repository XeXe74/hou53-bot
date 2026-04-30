"""
api.py

FastAPI backend for HOU53-bot. Receives a natural language description of a
house, extracts structured features using a local or remote LLM, runs the
regression model and returns both the predicted price and an explanation.
"""

import os
import json
import logging
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from predictor import predict, MODEL_NAME

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LLM settings read from environment so the same image works locally and in prod
LLM_URL = os.getenv("LLM_URL", "http://localhost:11434/api/generate")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:3b")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "60"))

# System prompt that instructs the LLM to extract house features as JSON
EXTRACTION_PROMPT = """
You are a real estate data assistant. Your only task is to extract house
features from the user description and return them as a valid JSON object.

Only include features that are EXPLICITLY mentioned in the description.
Do NOT infer, assume, or invent any values. If a feature is not mentioned, omit it entirely.
Use the exact field names listed below. Do not add any explanation or text outside the JSON object.

Key fields to extract (all optional):
- OverallQual (int 1-10): overall material and finish quality
- OverallCond (int 1-10): overall condition
- GrLivArea (int): above-ground living area in square feet
- TotalBsmtSF (int): total basement area in square feet
- GarageCars (int): garage capacity in cars
- GarageArea (int): garage area in square feet
- YearBuilt (int): year the house was built
- YearRemodAdd (int): year of last remodel
- FullBath (int): full bathrooms above grade
- HalfBath (int): half bathrooms above grade
- BedroomAbvGr (int): bedrooms above grade
- TotRmsAbvGrd (int): total rooms above grade
- Fireplaces (int): number of fireplaces
- LotArea (int): lot size in square feet
- Neighborhood (str): neighborhood name in Ames Iowa
- MSZoning (str): zoning classification (RL, RM, FV, RH)
- HouseStyle (str): style such as 1Story 2Story 1.5Fin
- ExterQual (str): exterior quality (Ex Gd TA Fa Po)
- KitchenQual (str): kitchen quality (Ex Gd TA Fa Po)
- BsmtQual (str): basement height quality (Ex Gd TA Fa Po None)
- GarageType (str): garage type (Attchd Detchd BuiltIn None)
- CentralAir (str): central air conditioning (Y or N)
- Foundation (str): foundation type (PConc CBlock Slab Stone Wood BrkTil)
- SaleType (str): type of sale (WD New COD)
- YrSold (int): year sold
- MoSold (int): month sold (1-12)

User description:
{description}

Return only the JSON object with the extracted features.
""".strip()


class PredictRequest(BaseModel):
    description: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description="Natural language description of the house to value.",
        examples=["A 3 bedroom house built in 1995 with a 2-car garage and central air."],
    )


class FeatureExplanation(BaseModel):
    feature: str
    importance: float
    description: str


class PredictResponse(BaseModel):
    predicted_price: float
    price_range_low: float
    price_range_high: float
    currency: str = "USD"
    model_used: str
    extracted_features: dict
    top_features: list[FeatureExplanation]
    summary: str


# Human-readable labels for the most common features shown in explanations
FEATURE_LABELS = {
    "OverallQual": "Overall quality rating",
    "GrLivArea": "Above-ground living area",
    "TotalSF": "Total square footage",
    "Qual_x_LiveArea": "Quality times living area",
    "Qual_x_TotalSF": "Quality times total area",
    "GarageArea": "Garage area",
    "TotalBsmtSF": "Basement area",
    "YearBuilt": "Year built",
    "HouseAge": "Age of the house",
    "Neighborhood_NridgHt": "Northridge Heights neighborhood",
    "Neighborhood_StoneBr": "Stone Brook neighborhood",
    "Neighborhood_NoRidge": "Northridge neighborhood",
}


app = FastAPI(
    title="HOU53-bot API",
    description="House price prediction from natural language descriptions.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


async def extract_features_with_llm(description: str) -> dict:
    """
    Call the configured LLM to extract structured house features from the
    natural language description. Returns an empty dict if the LLM is
    unavailable so the predictor can still run using dataset modes as defaults.
    """
    prompt = EXTRACTION_PROMPT.format(description=description)
    payload = {"model": LLM_MODEL, "prompt": prompt, "stream": False}

    try:
        async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
            response = await client.post(LLM_URL, json=payload)
            response.raise_for_status()
            raw = response.json().get("response", "")
            # Extract only the JSON block from the LLM response
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start == -1 or end == 0:
                logger.warning("LLM response contained no JSON block.")
                return {}
            return json.loads(raw[start:end])
    except httpx.ConnectError:
        logger.warning("LLM not reachable at %s. Using dataset defaults.", LLM_URL)
        return {}
    except (json.JSONDecodeError, KeyError) as exc:
        logger.warning("Could not parse LLM JSON response: %s", exc)
        return {}


def build_explanation(top_features: dict) -> list[FeatureExplanation]:
    """Convert the raw importance dict into human-readable explanations."""
    result = []
    for feature, importance in top_features.items():
        label = FEATURE_LABELS.get(feature, feature.replace("_", " ").title())
        result.append(FeatureExplanation(
            feature=feature,
            importance=importance,
            description=label,
        ))
    return result


def build_summary(price: float, extracted: dict) -> str:
    """Generate a one-sentence plain-language summary of the prediction."""
    qual = extracted.get("OverallQual")
    area = extracted.get("GrLivArea")
    year = extracted.get("YearBuilt")
    parts = []
    if qual:
        parts.append(f"quality rating of {qual}/10")
    if area:
        parts.append(f"living area of {area:,} sq ft")
    if year:
        parts.append(f"built in {year}")
    detail = (", ".join(parts) + " ") if parts else ""
    return (
        f"Based on the {detail}and similar properties in the dataset, "
        f"the estimated market value is ${price:,.0f}."
    )


@app.get("/health", summary="Health check")
async def health():
    """Returns 200 when the API and model are loaded and ready."""
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/predict", response_model=PredictResponse, summary="Predict house price")
async def predict_price(request: PredictRequest):
    """
    Receive a natural language description of a house, extract structured
    features with the LLM, run the regression model, and return the predicted
    price together with a feature importance explanation.
    """
    logger.info("Received description: %.80s...", request.description)

    # Step one: extract structured features from the natural language input
    extracted = await extract_features_with_llm(request.description)
    extracted = {k: v for k, v in extracted.items() if v not in (0, None, "", "None")}
    logger.info("Extracted features: %s", extracted)

    # Step two: run inference (missing fields filled with dataset modes)
    try:
        result = predict(extracted)
    except Exception as exc:
        logger.exception("Prediction failed.")
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc

    # Step three: build the human-readable explanation and summary
    explanation = build_explanation(result["top_features"])
    summary = build_summary(result["predicted_price"], extracted)

    return PredictResponse(
        predicted_price=result["predicted_price"],
        price_range_low=result["price_range_low"],
        price_range_high=result["price_range_high"],
        model_used=result["model_used"],
        extracted_features=extracted,
        top_features=explanation,
        summary=summary,
    )