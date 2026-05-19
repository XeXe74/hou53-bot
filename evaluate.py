"""
evaluate.py

Loads the trained model from disk and reports evaluation metrics on the
held-out test set using the same split as regression.py (random_state=42).
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
)

DATA       = "data/processed/house_prices_preprocessed.csv"
MODEL_FILE = "data/processed/best_model.pkl"
TEST_SPLIT = 0.2
RANDOM_STATE = 42

df = pd.read_csv(DATA)
X = df.drop(columns=["SalePrice"])
y = df["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SPLIT, random_state=RANDOM_STATE
)

model = joblib.load(MODEL_FILE)
model_name = type(model.named_steps["model"]).__name__

y_pred_real = np.expm1(model.predict(X_test))
y_test_real = np.expm1(y_test)

r2   = r2_score(y_test_real, y_pred_real)
mae  = mean_absolute_error(y_test_real, y_pred_real)
rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
mape = mean_absolute_percentage_error(y_test_real, y_pred_real) * 100

y_train_real = np.expm1(y_train)
y_train_pred_real = np.expm1(model.predict(X_train))
train_r2   = r2_score(y_train_real, y_train_pred_real)
train_rmse = np.sqrt(mean_squared_error(y_train_real, y_train_pred_real))

print(f"\n{'='*40}")
print(f"  Model : {model_name}")
print(f"{'='*40}")
print(f"  {'':20s} {'Train':>10} {'Test':>10}")
print(f"  {'R²':20s} {train_r2:>10.4f} {r2:>10.4f}")
print(f"  {'RMSE':20s} ${train_rmse:>9,.2f} ${rmse:>9,.2f}")
print(f"{'─'*40}")
print(f"  MAE  : ${mae:,.2f}")
print(f"  MAPE : {mape:.2f}%")
print(f"{'─'*40}")
gap_r2   = train_r2 - r2
gap_rmse = rmse - train_rmse
print(f"  R² gap   (train - test) : {gap_r2:.4f}")
print(f"  RMSE gap (test - train) : ${gap_rmse:,.2f}")
if gap_r2 > 0.1:
    print("  ⚠ Possible overfitting")
elif gap_r2 < 0:
    print("  ⚠ Possible underfitting")
else:
    print("  ✓ Good generalization")
print(f"{'='*40}\n")
