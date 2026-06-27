"""
Baseline LightGBM Model for CI/CD Duration Prediction
Simple, fast, and interpretable — use this to benchmark against the Transformer.
"""

import random
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor

SEED = 42
CSV_PATH = "complete_executions.csv"
TARGET_COL = "duration_seconds"

random.seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────────
# 1. LOAD & CLEAN
# ─────────────────────────────────────────────
print("Loading & Cleaning...")
df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=[TARGET_COL])
q_low, q_high = df[TARGET_COL].quantile(0.01), df[TARGET_COL].quantile(0.99)
df = df[(df[TARGET_COL] >= q_low) & (df[TARGET_COL] <= q_high)].reset_index(drop=True)
print(f"  Dataset size: {len(df):,} rows")

# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("Feature Engineering...")

df["repo_workflow_id"] = (
    df["repository_name"].astype(str) + "::" + df["workflow_path"].astype(str)
)

y_raw = df[TARGET_COL].values
y     = np.log1p(y_raw)   # log-transform target
X     = df.drop(columns=[TARGET_COL]).copy()

# Text: commit message keywords
X["metadata.head_commit.message"] = X["metadata.head_commit.message"].fillna("").astype(str)
msg = X["metadata.head_commit.message"].str.lower()
X["msg_len"]      = X["metadata.head_commit.message"].str.len()
X["msg_is_fix"]   = msg.str.contains(r"\bfix\b",     regex=True).astype(int)
X["msg_is_merge"] = msg.str.contains(r"\bmerge\b",   regex=True).astype(int)
X["msg_is_feat"]  = msg.str.contains(r"\bfeat\b",    regex=True).astype(int)
X["msg_is_ci"]    = msg.str.contains(r"\bci\b",      regex=True).astype(int)
X["msg_is_test"]  = msg.str.contains(r"\btest\b",    regex=True).astype(int)
X["msg_is_chore"] = msg.str.contains(r"\bchore\b",   regex=True).astype(int)

# Temporal cyclical encoding
for col in ["start_hour", "start_dayofweek", "start_month",
            "run_attempt", "time_to_import", "commit_to_run_seconds"]:
    if col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

X["hour_sin"]  = np.sin(2 * np.pi * X["start_hour"]      / 24)
X["hour_cos"]  = np.cos(2 * np.pi * X["start_hour"]      / 24)
X["day_sin"]   = np.sin(2 * np.pi * X["start_dayofweek"] / 7)
X["day_cos"]   = np.cos(2 * np.pi * X["start_dayofweek"] / 7)
X["month_sin"] = np.sin(2 * np.pi * X["start_month"]     / 12)
X["month_cos"] = np.cos(2 * np.pi * X["start_month"]     / 12)

# Bot flag
X["is_bot"] = (
    X["metadata.actor.type"].fillna("").str.lower() == "bot"
).astype(int)

# Frequency encoding (how often does this workflow/repo appear?)
for col in ["repo_workflow_id", "repository_name"]:
    freq = X[col].value_counts()
    X[f"{col}_freq"] = X[col].map(freq).fillna(0).astype(int)

# Label-encode categoricals — LightGBM handles these natively
cat_cols = [
    "repository_name", "metadata.event", "metadata.head_branch",
    "metadata.actor.type", "repo_workflow_id", "workflow_path"
]
for col in cat_cols:
    if col in X.columns:
        X[col] = X[col].fillna("missing").astype(str)
        X[col] = LabelEncoder().fit_transform(X[col])

# Drop raw text columns (already extracted what we need)
X = X.drop(columns=["metadata.head_commit.message"], errors="ignore")

# Label-encode ANY remaining object/string columns LightGBM can't handle
for col in X.select_dtypes(include=["object", "string"]).columns:
    X[col] = LabelEncoder().fit_transform(X[col].fillna("missing").astype(str))

# Fill any remaining NaNs
X = X.fillna(0)

print(f"  Features: {X.shape[1]}")

# ─────────────────────────────────────────────
# 3. SPLIT
# ─────────────────────────────────────────────
print("Splitting data (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)
_, _, y_train_raw, y_test_raw = train_test_split(
    X, y_raw, test_size=0.2, random_state=SEED
)

# ─────────────────────────────────────────────
# 4. TRAIN
# ─────────────────────────────────────────────
print("\nTraining LightGBM...")

model = LGBMRegressor(
    objective="huber",        # robust to outliers
    alpha=0.9,
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=8,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=20,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=SEED,
    verbose=-1,
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[
        __import__("lightgbm").early_stopping(stopping_rounds=50, verbose=False),
        __import__("lightgbm").log_evaluation(period=100),
    ],
)

print(f"  Best iteration: {model.best_iteration_}")

# ─────────────────────────────────────────────
# 5. EVALUATE
# ─────────────────────────────────────────────
y_pred = np.expm1(np.clip(model.predict(X_test), 0, None))

rmse = np.sqrt(mean_squared_error(y_test_raw, y_pred))
mae  = mean_absolute_error(y_test_raw, y_pred)
r2   = r2_score(y_test_raw, y_pred)

print("\n" + "★" * 6)
print("RESULTS — LightGBM Baseline")
print("★" * 6)
print(f"RMSE : {rmse:.4f} sec")
print(f"MAE  : {mae:.4f} sec")
print(f"R²   : {r2:.4f}")
print("★" * 6)

# ─────────────────────────────────────────────
# 6. FEATURE IMPORTANCE (top 15)
# ─────────────────────────────────────────────
importance = pd.DataFrame({
    "feature":    X_train.columns,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False).head(15)

print("\nTop 15 Most Important Features:")
print(importance.to_string(index=False))
