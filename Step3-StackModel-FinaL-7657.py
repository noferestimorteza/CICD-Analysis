import gc
import random
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import StackingRegressor
from category_encoders import LeaveOneOutEncoder, CatBoostEncoder
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor


SEED = 42
CSV_PATH = r"complete_executions.csv"
TARGET_COL = "duration_seconds"
MIN_RUNS = 3

random.seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────────
# 1. LOAD, CLEAN
# ─────────────────────────────────────────────
print("\nLoading & Cleaning...")
df = pd.read_csv(CSV_PATH)

df = df.dropna(subset=[TARGET_COL])
q_low, q_high = df[TARGET_COL].quantile(0.01), df[TARGET_COL].quantile(0.99)
df = df[(df[TARGET_COL] >= q_low) & (df[TARGET_COL] <= q_high)].reset_index(drop=True)

df["repo_workflow_id"] = df["repository_name"].astype(str) + "::" + df["workflow_path"].astype(str)

# Sort chronologically FIRST -- required for both the historical features
# and the MIN_RUNS filter below to be causally valid (no using the future
# to predict/filter the past).
time_col = next((c for c in df.columns if "run_started_at" in c.lower()), None)
if time_col:
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.sort_values(time_col).reset_index(drop=True)
    print(f"  Sorted by: {time_col}")
else:
    print("  WARNING: no run_started_at-like column found -- historical features "
          "and the MIN_RUNS filter cannot be made causally valid without one.")

# ── Remove workflows with < MIN_RUNS total runs ─────────────────────────
# Per-workflow holdout (see split section below) means every included
# workflow contributes to BOTH train and test by construction, so there's
# no global cutoff needed here -- just require enough total history that,
# after holding out its most recent run(s), real history remains.
run_counts = df.groupby("repo_workflow_id").size()
eligible_workflows = run_counts[run_counts >= MIN_RUNS].index

before_rows, before_workflows = len(df), df["repo_workflow_id"].nunique()
df = df[df["repo_workflow_id"].isin(eligible_workflows)].reset_index(drop=True)
after_rows, after_workflows = len(df), df["repo_workflow_id"].nunique()

print(f"Workflow run-count filter (MIN_RUNS={MIN_RUNS} total runs):")
print(f"  Rows:      {before_rows:,} -> {after_rows:,} ({before_rows - after_rows:,} removed)")
print(f"  Workflows: {before_workflows:,} -> {after_workflows:,} ({before_workflows - after_workflows:,} removed)")

# ─────────────────────────────────────────────
# 2. HISTORICAL / LAG FEATURES
# ─────────────────────────────────────────────
print("Building historical features...")
grp = df.groupby("repo_workflow_id")[TARGET_COL]
df["hist_mean_5"] = grp.transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
df["hist_ewm_5"]  = grp.transform(lambda x: x.shift(1).ewm(span=5).mean())
df["hist_last"]   = grp.transform(lambda x: x.shift(1))
df["hist_std_5"]  = grp.transform(lambda x: x.shift(1).rolling(5, min_periods=1).std().fillna(0))
df["is_first_run"] = (df.groupby("repo_workflow_id").cumcount() == 0).astype(int)

global_mean = df[TARGET_COL].mean()
for col in ["hist_mean_5", "hist_ewm_5", "hist_last", "hist_std_5"]:
    df[col] = df[col].fillna(global_mean)

if time_col:
    df["time_since_last_run_sec"] = df.groupby("repo_workflow_id")[time_col].diff().dt.total_seconds()
    df["time_since_last_run_sec"] = df["time_since_last_run_sec"].fillna(df["time_since_last_run_sec"].median())

if "metadata.actor.login" in df.columns:
    actor_grp = df.groupby("metadata.actor.login")[TARGET_COL]
    df["actor_hist_mean_5"] = actor_grp.transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    df["actor_hist_last_1"] = actor_grp.transform(lambda x: x.shift(1))
    for col in ["actor_hist_mean_5", "actor_hist_last_1"]:
        df[col] = df[col].fillna(global_mean)

print("  Added: hist_mean_5, hist_ewm_5, hist_last, hist_std_5, is_first_run, "
      "time_since_last_run_sec, actor_hist_mean_5, actor_hist_last_1")

# ─────────────────────────────────────────────
# 3. FEATURE PREP
# ─────────────────────────────────────────────
y = df[TARGET_COL].values
X = df.drop(columns=[TARGET_COL]).copy()

# Text & Booleans
text_cols = ["metadata.head_commit.message", "repo_workflow_id", "metadata.event"]
for col in text_cols:
    X[col] = X[col].fillna("").astype(str)

X["master_text"] = X["metadata.head_commit.message"] + " " + X["metadata.event"]
X["msg_len"] = X["metadata.head_commit.message"].apply(len)
X["is_bot"] = X["metadata.actor.type"].apply(lambda x: 1 if str(x).lower() == 'bot' else 0)

# Numerics & Time
num_cols = ["start_hour", "start_dayofweek", "start_month", "run_attempt", "time_to_import", "commit_to_run_seconds"]
for col in num_cols:
    if col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

if "start_hour" in X.columns:
    X["hour_sin"] = np.sin(2 * np.pi * X["start_hour"] / 24)
    X["hour_cos"] = np.cos(2 * np.pi * X["start_hour"] / 24)

num_cols_all = X.select_dtypes(include=[np.number]).columns
X[num_cols_all] = X[num_cols_all].fillna(0)
cat_cols_all = X.select_dtypes(exclude=[np.number]).columns
X[cat_cols_all] = X[cat_cols_all].fillna("missing")

assert X.isna().sum().sum() == 0, "NaNs remaining!"

# ─────────────────────────────────────────────
# 4. SPLIT (per-workflow holdout, not a global cutoff) & TARGET ENCODING
# ─────────────────────────────────────────────
print("Per-workflow holdout split & Target Encoding...")

N_HOLDOUT_PER_WORKFLOW = 1  # hold out each workflow's N most-recent runs as test

if time_col:
    # df is already sorted ascending by time_col (from the sort at the top
    # of the script), and no row-reordering operation has touched it since.
    # cumcount(ascending=False) numbers each group's rows 0, 1, 2, ... going
    # BACKWARD from its last occurrence -- so rank 0 is that workflow's most
    # recent run, rank 1 its second-most-recent, etc. This guarantees every
    # included workflow contributes to BOTH train and test (its own earlier
    # runs vs. its own most recent run), sidestepping the global-cutoff
    # cohort-shift problem entirely.
    rank_desc = df.groupby("repo_workflow_id").cumcount(ascending=False)
    test_mask = rank_desc < N_HOLDOUT_PER_WORKFLOW
else:
    # No timestamp -- can't define "most recent," fall back to a plain
    # positional split (least meaningful option, but keeps the script
    # runnable).
    split_idx = int(len(df) * 0.8)
    test_mask = pd.Series([False] * split_idx + [True] * (len(df) - split_idx), index=df.index)

train_mask = ~test_mask
X_train, X_test = X[train_mask].copy(), X[~train_mask].copy()
y_train, y_test = y[train_mask.values], y[~train_mask.values]
print(f"  Train: {len(X_train):,} rows | Test: {len(X_test):,} rows "
      f"(test = each workflow's most recent {N_HOLDOUT_PER_WORKFLOW} run(s))")

if len(X_test) == 0 or len(X_train) == 0:
    raise RuntimeError(
        "Split produced an empty train or test set -- check MIN_RUNS and "
        "N_HOLDOUT_PER_WORKFLOW against the row/workflow counts printed above."
    )

loo_encoder = LeaveOneOutEncoder(sigma=2.0, random_state=SEED)
X_train["baseline_mean"] = loo_encoder.fit_transform(X_train["repo_workflow_id"], y_train)
X_test["baseline_mean"] = loo_encoder.transform(X_test["repo_workflow_id"])

global_mean_enc = y_train.mean()
X_train["baseline_mean"] = X_train["baseline_mean"].fillna(global_mean_enc)
X_test["baseline_mean"] = X_test["baseline_mean"].fillna(global_mean_enc)

# ─────────────────────────────────────────────
# 5. DENSE FEATURE EXTRACTION
# ─────────────────────────────────────────────
print("NLP + Categoricals...")

# Text SVD
tfidf = TfidfVectorizer(max_features=15000, ngram_range=(1, 2), min_df=3, sublinear_tf=True)
X_train_tfidf = tfidf.fit_transform(X_train["master_text"])
X_test_tfidf = tfidf.transform(X_test["master_text"])

svd = TruncatedSVD(n_components=100, random_state=SEED)
X_train_text_dense = svd.fit_transform(X_train_tfidf)
X_test_text_dense = svd.transform(X_test_tfidf)

# Cat Encoding
cat_cols = ["repository_name", "metadata.event", "metadata.head_branch", "repo_workflow_id"]
cb_enc = CatBoostEncoder(cols=[c for c in cat_cols if c in X_train.columns])
X_train_struct = cb_enc.fit_transform(X_train.drop(columns=["master_text", "metadata.head_commit.message"]), y_train)
X_test_struct = cb_enc.transform(X_test.drop(columns=["master_text", "metadata.head_commit.message"]))

# Scaler
scaler = StandardScaler()
X_train_num_scaled = scaler.fit_transform(X_train_struct.select_dtypes(include=np.number))
X_test_num_scaled = scaler.transform(X_test_struct.select_dtypes(include=np.number))

# Final Matrix
X_train_final = np.hstack([X_train_num_scaled, X_train_text_dense])
X_test_final = np.hstack([X_test_num_scaled, X_test_text_dense])

# ─────────────────────────────────────────────
# 6. SEGMENT ROUTING (MIXTURE OF EXPERTS)
# ─────────────────────────────────────────────
print("Routing Data to Segmented Experts...")

THRESH_FAST = 90
THRESH_SLOW = 400

def get_segment_masks(baseline_array):
    m_fast = baseline_array <= THRESH_FAST
    m_med = (baseline_array > THRESH_FAST) & (baseline_array <= THRESH_SLOW)
    m_slow = baseline_array > THRESH_SLOW
    return m_fast, m_med, m_slow

train_m_fast, train_m_med, train_m_slow = get_segment_masks(X_train["baseline_mean"].values)
test_m_fast, test_m_med, test_m_slow = get_segment_masks(X_test["baseline_mean"].values)

print(f"   -> FAST   (<={THRESH_FAST}s)  : {train_m_fast.sum()} samples")
print(f"   -> MEDIUM ({THRESH_FAST}-{THRESH_SLOW}s): {train_m_med.sum()} samples")
print(f"   -> SLOW   (>{THRESH_SLOW}s) : {train_m_slow.sum()} samples")

# ─────────────────────────────────────────────
# 7. TRAINING SEGMENTED EXPERTS
# ─────────────────────────────────────────────
print("\nTraining Segmented Stacking Models...")

def build_expert(segment_name):
    learning_rate = 0.03 if segment_name == "FAST" else 0.04

    estimators = [
        ('lgb_rmse', LGBMRegressor(
            objective="regression", n_estimators=1000, learning_rate=learning_rate,
            max_depth=10, num_leaves=64, subsample=0.8, random_state=SEED, verbose=-1
        )),
        ('xgb_poisson', XGBRegressor(
            objective="count:poisson", n_estimators=800, learning_rate=learning_rate,
            max_depth=8, subsample=0.8, tree_method="hist", random_state=SEED
        ))
    ]
    return StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=2.0, positive=True),
        cv=KFold(n_splits=5, shuffle=True, random_state=SEED),
        n_jobs=-1,
        passthrough=True
    )

models = {
    "FAST": build_expert("FAST"),
    "MEDIUM": build_expert("MEDIUM"),
    "SLOW": build_expert("SLOW")
}

if train_m_fast.sum() > 0:
    models["FAST"].fit(X_train_final[train_m_fast], y_train[train_m_fast])

if train_m_med.sum() > 0:
    models["MEDIUM"].fit(X_train_final[train_m_med], y_train[train_m_med])

if train_m_slow.sum() > 0:
    models["SLOW"].fit(X_train_final[train_m_slow], y_train[train_m_slow])

# ─────────────────────────────────────────────
# 8. INFERENCE & ASSEMBLY
# ─────────────────────────────────────────────
print("\nAssembling Predictions and Evaluating...")

y_pred = np.zeros(len(y_test))

if test_m_fast.sum() > 0:
    y_pred[test_m_fast] = models["FAST"].predict(X_test_final[test_m_fast])

if test_m_med.sum() > 0:
    y_pred[test_m_med] = models["MEDIUM"].predict(X_test_final[test_m_med])

if test_m_slow.sum() > 0:
    y_pred[test_m_slow] = models["SLOW"].predict(X_test_final[test_m_slow])

y_pred = np.clip(y_pred, 0, None)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n" + "★" * 6)
print("RESULTS")
print("★" * 6)
print(f"RMSE : {rmse:.4f} sec")
print(f"MAE  : {mae:.4f} sec")
print(f"R²   : {r2:.4f}")
print("★" * 6)

gc.collect()