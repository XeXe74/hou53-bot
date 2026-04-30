"""
preprocessor.py

Replicates every step from preprocessing.ipynb so the API can transform
raw house data into the feature vector that best_model.pkl expects.

Run once after training to generate preprocessor_meta.pkl:
    python -m app.preprocessor
"""

import os
import joblib
import numpy as np
import pandas as pd
from scipy.stats import mstats

# Paths
RAW_DATA = os.path.join("data", "raw", "house_prices.csv")
PROCESSED_CSV = os.path.join("data", "processed", "house_prices_preprocessed.csv")
META_PATH = os.path.join("data", "processed", "preprocessor_meta.pkl")

# Columns where NaN means the feature simply does not exist
CAT_NONE_COLS = {
    "Fence": "no fence",
    "FireplaceQu": "no fireplace",
    "GarageType": "no garage",
    "GarageFinish": "no garage",
    "GarageQual": "no garage",
    "GarageCond": "no garage",
    "BsmtQual": "no basement",
    "BsmtCond": "no basement",
    "BsmtExposure": "no basement",
    "BsmtFinType1": "no basement",
    "BsmtFinType2": "no basement",
    "MasVnrType": "no masonry veneer",
    "Alley": "no alley access",
    "PoolQC": "no pool",
    "MiscFeature": "no misc feature",
}

# Numeric columns where NaN means zero
NUM_ZERO_COLS = [
    "GarageYrBlt", "GarageArea", "GarageCars",
    "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF",
    "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath", "MasVnrArea",
]

# Columns encoded with the standard quality scale Ex to Po
ORDINAL_QUALITY_COLS = [
    "ExterQual", "ExterCond", "BsmtQual", "BsmtCond",
    "HeatingQC", "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond",
]

# Quality scale shared by all ordinal quality columns
QUALITY_MAP = {
    "Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1,
    "None": 0, "no garage": 0, "no basement": 0, "no fireplace": 0,
}

# Basement finish type scale
BSMTFIN_MAP = {
    "GLQ": 6, "ALQ": 5, "BLQ": 4, "Rec": 3,
    "LwQ": 2, "Unf": 1, "None": 0, "no basement": 0,
}

# Nominal columns that get one-hot encoded
NOMINAL_COLS = [
    "MSZoning", "Street", "Alley", "LotShape", "LandContour",
    "LotConfig", "Neighborhood", "Condition1", "Condition2",
    "BldgType", "HouseStyle", "RoofStyle", "RoofMatl",
    "Exterior1st", "Exterior2nd", "MasVnrType", "Foundation",
    "Heating", "Electrical", "GarageType", "MiscFeature",
    "SaleType", "SaleCondition", "Fence", "PoolQC",
]

# Skewed numeric columns that get log1p transformation
LOG_COLS = [
    "MasVnrArea", "OpenPorchSF", "LotArea", "LotFrontage",
    "WoodDeckSF", "GrLivArea", "BsmtUnfSF", "2ndFlrSF",
    "BsmtFinSF1", "EnclosedPorch",
]

# Columns that get winsorized to reduce the effect of extreme outliers
WINSORIZE_COLS = ["LotArea", "TotalBsmtSF", "GarageArea", "1stFlrSF"]


def transform(df_raw: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Transform a raw dataframe into the feature matrix expected by best_model.pkl.
    feature_cols is the ordered list of columns saved during fit_and_save.
    Missing dummy columns are filled with zero so new data always aligns.
    """
    df = df_raw.copy()

    # Fill NaN in categorical columns that indicate absence of a feature
    for col, val in CAT_NONE_COLS.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)

    # Fill NaN in numeric columns where absence means zero
    for col in NUM_ZERO_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Fill LotFrontage with the median of the same neighborhood
    if "LotFrontage" in df.columns and "Neighborhood" in df.columns:
        df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"] \
                              .transform(lambda x: x.fillna(x.median()))
        df["LotFrontage"] = df["LotFrontage"].fillna(df["LotFrontage"].median())

    # Fill the single Electrical NaN with the most common value
    if "Electrical" in df.columns:
        df["Electrical"] = df["Electrical"].fillna("SBrkr")

    # Encode ordinal quality columns with the shared quality scale
    for col in ORDINAL_QUALITY_COLS:
        if col in df.columns:
            df[col] = df[col].map(QUALITY_MAP).fillna(0).astype(int)

    # Encode basement exposure from best to none
    if "BsmtExposure" in df.columns:
        df["BsmtExposure"] = df["BsmtExposure"].map(
            {"Gd": 4, "Av": 3, "Mn": 2, "No": 1, "None": 0, "no basement": 0}
        ).fillna(0).astype(int)

    # Encode basement finish type for both type columns
    if "BsmtFinType1" in df.columns:
        df["BsmtFinType1"] = df["BsmtFinType1"].map(BSMTFIN_MAP).fillna(0).astype(int)
    if "BsmtFinType2" in df.columns:
        df["BsmtFinType2"] = df["BsmtFinType2"].map(BSMTFIN_MAP).fillna(0).astype(int)

    # Encode garage finish quality
    if "GarageFinish" in df.columns:
        df["GarageFinish"] = df["GarageFinish"].map(
            {"Fin": 3, "RFn": 2, "Unf": 1, "None": 0, "no garage": 0}
        ).fillna(0).astype(int)

    # Encode how functional the house is from typical to salvage only
    if "Functional" in df.columns:
        df["Functional"] = df["Functional"].map(
            {"Typ": 7, "Min1": 6, "Min2": 5, "Mod": 4, "Maj1": 3, "Maj2": 2, "Sev": 1, "Sal": 0}
        ).fillna(7).astype(int)

    # Encode terrain slope from gentle to severe
    if "LandSlope" in df.columns:
        df["LandSlope"] = df["LandSlope"].map(
            {"Gtl": 2, "Mod": 1, "Sev": 0}
        ).fillna(2).astype(int)

    # Encode whether the driveway is paved
    if "PavedDrive" in df.columns:
        df["PavedDrive"] = df["PavedDrive"].map(
            {"Y": 2, "P": 1, "N": 0}
        ).fillna(2).astype(int)

    # Encode central air as binary
    if "CentralAir" in df.columns:
        df["CentralAir"] = df["CentralAir"].map(
            {"Y": 1, "N": 0}
        ).fillna(1).astype(int)

    # One-hot encode MSSubClass treated as a category not a number
    if "MSSubClass" in df.columns:
        df["MSSubClass"] = df["MSSubClass"].astype(str)
        df = pd.get_dummies(df, columns=["MSSubClass"], drop_first=True)

    # Drop Utilities because it has almost no variance in the dataset
    df.drop(columns=["Utilities"], inplace=True, errors="ignore")

    # One-hot encode all nominal categorical columns
    existing_nominal = [c for c in NOMINAL_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=existing_nominal, drop_first=True)

    # Winsorize extreme values using the same percentile limits as training
    for col in WINSORIZE_COLS:
        if col in df.columns:
            df[col] = mstats.winsorize(df[col], limits=[0.01, 0.01])

    # Combined quality score of overall quality plus overall condition
    df["TotalQual"] = df.get("OverallQual", 5) + df.get("OverallCond", 5)

    # Total square footage across basement and both floors
    df["TotalSF"] = df.get("TotalBsmtSF", 0) + df.get("1stFlrSF", 0) + df.get("2ndFlrSF", 0)

    # Age of the house and of the last remodel at the time of sale
    df["HouseAge"] = df.get("YrSold", 2010) - df.get("YearBuilt", 2000)
    df["RemodAge"] = df.get("YrSold", 2010) - df.get("YearRemodAdd", 2000)

    # Binary flag for whether the house was ever remodeled
    df["WasRemodeled"] = (df.get("YearBuilt", 0) != df.get("YearRemodAdd", 0)).astype(int)

    # Total bathrooms counting half baths as half
    df["TotalBaths"] = (
        df.get("FullBath", 0) + df.get("BsmtFullBath", 0)
        + 0.5 * df.get("HalfBath", 0) + df.get("BsmtHalfBath", 0)
    )

    # Binary flags for the presence of key features
    df["HasGarage"] = (df.get("GarageArea", 0) > 0).astype(int)
    df["HasBasement"] = (df.get("TotalBsmtSF", 0) > 0).astype(int)
    df["HasFireplace"] = (df.get("Fireplaces", 0) > 0).astype(int)
    df["HasPool"] = (df.get("PoolArea", 0) > 0).astype(int)
    df["HasPorch"] = (
        (df.get("OpenPorchSF", 0) + df.get("EnclosedPorch", 0)
         + df.get("ScreenPorch", 0) + df.get("WoodDeckSF", 0)) > 0
    ).astype(int)

    # Quality divided by living area to capture value per square foot
    df["QualPerSF"] = df.get("OverallQual", 5) / (df.get("GrLivArea", 1) + 1)

    # Interaction terms between quality and size features (names match the CSV)
    df["Qual_x_LiveArea"] = df.get("OverallQual", 5) * df.get("GrLivArea", 1000)
    df["Qual_x_TotalSF"] = df.get("OverallQual", 5) * df["TotalSF"]
    df["Qual_x_GarageArea"] = df.get("OverallQual", 5) * df.get("GarageArea", 0)

    # Ratio of finished basement to total basement area
    df["BsmtFinRatio"] = df.get("BsmtFinSF1", 0) / (df.get("TotalBsmtSF", 0) + 1)

    # How old the garage is relative to sale year
    df["GarageAge"] = df.get("YrSold", 2010) - df.get("GarageYrBlt", df.get("YearBuilt", 2000))

    # Flag for garages built after the house was constructed
    df["GarageNewerThanHouse"] = (df.get("GarageYrBlt", 0) > df.get("YearBuilt", 0)).astype(int)

    # Total porch area across all porch types
    df["TotalPorchSF"] = (
        df.get("OpenPorchSF", 0) + df.get("EnclosedPorch", 0)
        + df.get("ScreenPorch", 0) + df.get("WoodDeckSF", 0)
        + df.get("3SsnPorch", 0)
    )

    # Log transform skewed features to reduce the effect of extreme values
    for col in LOG_COLS:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))

    # Drop target and identifier columns if present
    df.drop(columns=["Id", "SalePrice"], inplace=True, errors="ignore")

    # Align columns to training order and fill any unseen dummies with zero
    if feature_cols:
        df = df.reindex(columns=feature_cols, fill_value=0)

    return df


def fit_and_save(raw_csv_path: str = RAW_DATA, processed_csv_path: str = PROCESSED_CSV, meta_path: str = META_PATH):
    """
    Run once after training to persist the column order and raw column modes.
    Feature columns are read directly from the preprocessed CSV so they always
    match exactly what the model saw during training.
    The modes are used by the API to fill fields the LLM did not extract.
    """
    # Read feature columns directly from the preprocessed CSV (source of truth)
    df_processed = pd.read_csv(processed_csv_path)
    feature_cols = [c for c in df_processed.columns if c != "SalePrice"]

    # Compute modes on raw data to use as defaults for missing LLM fields
    df_raw = pd.read_csv(raw_csv_path, na_values=["?", "NA", ""])
    modes = df_raw.mode().iloc[0].to_dict()

    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    joblib.dump({"feature_cols": feature_cols, "modes": modes}, meta_path)

    print(f"Saved preprocessor meta to {meta_path}")
    print(f"Features saved: {len(feature_cols)}")
    print(f"Raw column modes saved: {len(modes)}")

    return feature_cols, modes


def load_meta(meta_path: str = META_PATH) -> tuple[list, dict]:
    """Load the feature column order and raw column modes from disk."""
    meta = joblib.load(meta_path)
    return meta["feature_cols"], meta["modes"]


if __name__ == "__main__":
    fit_and_save()