import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# ── Config ────────────────────────────────────────────────────────────────────
DATA         = "data/processed/house_prices_preprocessed.csv"
MODEL_FILE   = "data/processed/best_model.pkl"
TEST_SPLIT   = 0.2
RANDOM_STATE = 42
CV_FOLDS     = 5

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA)

X = df.drop(columns=["SalePrice"])
y = df["SalePrice"]  # log1p scale

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SPLIT, random_state=RANDOM_STATE
)
print(f"\nTrain: {X_train.shape} | Test: {X_test.shape}")

# ── Models ────────────────────────────────────────────────────────────────────
models = [
    ("Decision Tree",     DecisionTreeRegressor(random_state=RANDOM_STATE)),
    ("Ridge",             Ridge()),
    ("Lasso",             Lasso(max_iter=20000)),
    ("SVR",               SVR()),
    ("Random Forest",     RandomForestRegressor(random_state=RANDOM_STATE)),
    ("Gradient Boosting", GradientBoostingRegressor(random_state=RANDOM_STATE)),
]

# ── Hyperparameter grids ──────────────────────────────────────────────────────
hyperparam_grid = {
    "Decision Tree": {
        "model__max_depth":         [None, 5, 10, 20],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf":  [1, 2, 4],
        "model__max_features":      [None, "sqrt", "log2"],
    },
    "Ridge": {
        "model__alpha":         [0.1, 1, 10, 50, 100],
        "model__fit_intercept": [True, False],
    },
    "Lasso": {
        "model__alpha":         [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1],
        "model__fit_intercept": [True, False],
    },
    "SVR": {
        "model__C":       [0.1, 1, 10],
        "model__kernel":  ["rbf", "poly"],
        "model__gamma":   ["scale", "auto"],
        "model__epsilon": [0.05, 0.1, 0.2],
    },
    "Random Forest": {
        "model__n_estimators":      [100, 200],
        "model__max_depth":         [None, 10, 20],
        "model__min_samples_split": [2, 5],
        "model__max_features":      ["sqrt", "log2"],
    },
    "Gradient Boosting": {
        "model__n_estimators":  [100, 200],
        "model__learning_rate": [0.05, 0.1, 0.2],
        "model__max_depth":     [3, 5, 10],
    },
}

# ── Training + GridSearch ─────────────────────────────────────────────────────
results = []

for name, estimator in models:
    print(f"\n{'='*50}\nTraining: {name}")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  estimator),
    ])

    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(
        pipeline,
        hyperparam_grid[name],
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train, y_train)

    best = grid.best_estimator_
    y_pred_real = np.expm1(best.predict(X_test))
    y_test_real = np.expm1(y_test)

    rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
    mae  = mean_absolute_error(y_test_real, y_pred_real)
    mse  = mean_squared_error(y_test_real, y_pred_real)
    mape = mean_absolute_percentage_error(y_test_real, y_pred_real) * 100
    r2   = r2_score(y_test_real, y_pred_real)

    print(f"  Best params  : {grid.best_params_}")
    print(f"  CV RMSE (log): {-grid.best_score_:.4f}")
    print(f"  Test R²      : {r2:.4f}")
    print(f"  Test MAE     : ${mae:,.2f}")
    print(f"  Test MSE     : ${mse:,.2f}")
    print(f"  Test RMSE    : ${rmse:,.2f}")
    print(f"  Test MAPE    : {mape:.2f}%")

    results.append({
        "name":  name,
        "model": best,
        "r2":    r2,
        "mae":   mae,
        "mse":   mse,
        "rmse":  rmse,
        "mape":  mape,
    })

# ── Best model ────────────────────────────────────────────────────────────────
best_result = min(results, key=lambda x: x["rmse"])
best_model  = best_result["model"]

print(f"\n✅ Best model: {best_result['name']}")
print(f"   RMSE: ${best_result['rmse']:,.2f} | R²: {best_result['r2']:.4f}")

joblib.dump(best_model, MODEL_FILE)
print(f"   Saved to {MODEL_FILE}")

# ── Final evaluation ──────────────────────────────────────────────────────────
y_pred_log  = best_model.predict(X_test)
y_pred_real = np.expm1(y_pred_log)
y_test_real = np.expm1(y_test)

y_train_pred_real = np.expm1(best_model.predict(X_train))
y_train_real      = np.expm1(y_train)

r2   = r2_score(y_test_real, y_pred_real)
mae  = mean_absolute_error(y_test_real, y_pred_real)
mse  = mean_squared_error(y_test_real, y_pred_real)
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(y_test_real, y_pred_real) * 100

print(f"\n{'='*40}")
print(f"  Best model : {best_result['name']}")
print(f"{'='*40}")
print(f"  R²  : {r2:.4f}")
print(f"  MAE : ${mae:,.2f}")
print(f"  MSE : ${mse:,.2f}")
print(f"  RMSE: ${rmse:,.2f}")
print(f"  MAPE: {mape:.2f}%")

# ── Overfitting check ─────────────────────────────────────────────────────────
train_r2   = r2_score(y_train_real, y_train_pred_real)
train_rmse = np.sqrt(mean_squared_error(y_train_real, y_train_pred_real))

print(f"\n{'─'*40}")
print(f"  Overfitting check")
print(f"{'─'*40}")
print(f"  {'':20s} {'Train':>10} {'Test':>10}")
print(f"  {'R²':20s} {train_r2:>10.4f} {r2:>10.4f}")
print(f"  {'RMSE':20s} ${train_rmse:>9,.2f} ${rmse:>9,.2f}")

gap_r2   = train_r2 - r2
gap_rmse = rmse - train_rmse

print(f"\n  R² gap   (train - test) : {gap_r2:.4f}")
print(f"  RMSE gap (test - train) : ${gap_rmse:,.2f}")

if gap_r2 > 0.1:
    print("Possible overfitting — R² gap > 0.1")
elif gap_r2 < 0:
    print("Possible underfitting — test R² > train R²")
else:
    print("Good generalization")

# ── Feature importance (tree-based models only) ───────────────────────────────
for r in results:
    final_model = r["model"].named_steps["model"]
    if hasattr(final_model, "feature_importances_"):
        fi = pd.Series(final_model.feature_importances_, index=X.columns)
        fi.nlargest(15).plot(kind="barh", title=f"Top 15 features — {r['name']}")
        plt.tight_layout()
        plt.show()

# ── Predicted vs Actual ───────────────────────────────────────────────────────
plt.figure(figsize=(8, 6))
plt.scatter(y_test_real, y_pred_real, alpha=0.4, edgecolors="none")
plt.plot([y_test_real.min(), y_test_real.max()],
         [y_test_real.min(), y_test_real.max()],
         "r--", linewidth=1.5, label="Perfect prediction")
plt.xlabel("Actual SalePrice ($)")
plt.ylabel("Predicted SalePrice ($)")
plt.title(f"Predicted vs Actual — {best_result['name']}")
plt.legend()
plt.tight_layout()
plt.savefig("data/processed/predicted_vs_actual.png", dpi=150, bbox_inches="tight")
plt.show()

# ── Residuals ─────────────────────────────────────────────────────────────────
residuals = y_test_real - y_pred_real

plt.figure(figsize=(8, 4))
plt.scatter(y_pred_real, residuals, alpha=0.4, edgecolors="none")
plt.axhline(0, color="red", linestyle="--", linewidth=1.5)
plt.xlabel("Predicted SalePrice ($)")
plt.ylabel("Residual ($)")
plt.title(f"Residuals — {best_result['name']}")
plt.tight_layout()
plt.show()