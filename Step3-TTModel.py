"""
Deep Learning Pipeline for CI/CD Duration Prediction
Architecture: Tabular Transformer + Residual MLP (FT-Transformer style)
"""

import random
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from category_encoders import LeaveOneOutEncoder
warnings.filterwarnings("ignore")

SEED = 42
CSV_PATH = "complete_executions.csv"
TARGET_COL = "duration_seconds"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ─────────────────────────────────────────────
# 1. LOAD & CLEAN
# ─────────────────────────────────────────────
print("\nLoading & Cleaning...")
df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=[TARGET_COL])
q_low, q_high = df[TARGET_COL].quantile(0.01), df[TARGET_COL].quantile(0.99)
df = df[(df[TARGET_COL] >= q_low) & (df[TARGET_COL] <= q_high)].reset_index(drop=True)

# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("Feature Engineering...")

df["repo_workflow_id"] = df["repository_name"].astype(str) + "::" + df["workflow_path"].astype(str)

y_raw = df[TARGET_COL].values
y = np.log1p(y_raw)  # log-transform target
X = df.drop(columns=[TARGET_COL]).copy()

# Text
for col in ["metadata.head_commit.message", "metadata.event", "repo_workflow_id"]:
    X[col] = X[col].fillna("").astype(str)

X["master_text"] = X["metadata.head_commit.message"] + " " + X["metadata.event"]
X["msg_len"] = X["metadata.head_commit.message"].str.len()
X["is_bot"] = (X["metadata.actor.type"].str.lower() == "bot").astype(int)

# Keyword flags
msg = X["metadata.head_commit.message"].str.lower()
for kw in ["fix", "merge", "release", "chore", "test", "revert", "feat", "ci"]:
    X[f"msg_{kw}"] = msg.str.contains(rf"\b{kw}\b", regex=True).astype(int)

# Temporal cyclical
num_cols = ["start_hour", "start_dayofweek", "start_month",
            "run_attempt", "time_to_import", "commit_to_run_seconds"]
for col in num_cols:
    if col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

X["hour_sin"]  = np.sin(2 * np.pi * X["start_hour"] / 24)
X["hour_cos"]  = np.cos(2 * np.pi * X["start_hour"] / 24)
X["day_sin"]   = np.sin(2 * np.pi * X["start_dayofweek"] / 7)
X["day_cos"]   = np.cos(2 * np.pi * X["start_dayofweek"] / 7)
X["month_sin"] = np.sin(2 * np.pi * X["start_month"] / 12)
X["month_cos"] = np.cos(2 * np.pi * X["start_month"] / 12)

# Frequency encoding
for col in ["repo_workflow_id", "repository_name"]:
    freq = X[col].value_counts()
    X[f"{col}_freq"] = X[col].map(freq).fillna(0).astype(int)

# Identify categorical columns for embedding
CAT_COLS = ["metadata.event", "metadata.head_branch", "metadata.actor.type",
            "repository_name", "repo_workflow_id"]
CAT_COLS = [c for c in CAT_COLS if c in X.columns]

# Label-encode categoricals for embedding layer
label_encoders = {}
for col in CAT_COLS:
    X[col] = X[col].fillna("missing").astype(str)
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Fill remaining
X[X.select_dtypes(include=[np.number]).columns] = \
    X.select_dtypes(include=[np.number]).fillna(0)
X[X.select_dtypes(exclude=[np.number]).columns] = \
    X.select_dtypes(exclude=[np.number]).fillna("missing")

# ─────────────────────────────────────────────
# 3. SPLIT
# ─────────────────────────────────────────────
print("Splitting...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
_, _, y_train_raw, y_test_raw = train_test_split(X, y_raw, test_size=0.2, random_state=SEED)

# ─────────────────────────────────────────────
# 4. LOO TARGET ENCODING + TF-IDF
# ─────────────────────────────────────────────
print("Target Encoding + NLP...")

loo = LeaveOneOutEncoder(sigma=1.5, random_state=SEED)
X_train["baseline_mean"] = loo.fit_transform(
    X_train["repo_workflow_id"].astype(str), y_train
).fillna(y_train.mean())
X_test["baseline_mean"] = loo.transform(
    X_test["repo_workflow_id"].astype(str)
).fillna(y_train.mean())

# TF-IDF + SVD on commit messages
tfidf = TfidfVectorizer(max_features=15000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
svd   = TruncatedSVD(n_components=64, random_state=SEED)
X_train_text = svd.fit_transform(tfidf.fit_transform(X_train["master_text"]))
X_test_text  = svd.transform(tfidf.transform(X_test["master_text"]))

# Numeric features (all non-cat, non-text columns)
exclude = set(CAT_COLS) | {"master_text", "metadata.head_commit.message"}
num_feat_cols = [c for c in X_train.select_dtypes(include=np.number).columns
                 if c not in exclude]

scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train[num_feat_cols].values)
X_test_num  = scaler.transform(X_test[num_feat_cols].values)

# Combine numeric + text into final continuous matrix
X_train_cont = np.hstack([X_train_num, X_train_text]).astype(np.float32)
X_test_cont  = np.hstack([X_test_num,  X_test_text]).astype(np.float32)

# Categorical integer arrays for embedding
X_train_cat = X_train[CAT_COLS].values.astype(np.int64)
X_test_cat  = X_test[CAT_COLS].values.astype(np.int64)

cat_vocab_sizes = [X[col].nunique() + 1 for col in CAT_COLS]

# ─────────────────────────────────────────────
# 5. PYTORCH DATASET
# ─────────────────────────────────────────────
class CICDDataset(Dataset):
    def __init__(self, cont, cat, targets):
        self.cont    = torch.tensor(cont, dtype=torch.float32)
        self.cat     = torch.tensor(cat,  dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.cont[idx], self.cat[idx], self.targets[idx]

train_ds = CICDDataset(X_train_cont, X_train_cat, y_train.astype(np.float32))
test_ds  = CICDDataset(X_test_cont,  X_test_cat,  y_test.astype(np.float32))

train_loader = DataLoader(train_ds, batch_size=512, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=512, shuffle=False, num_workers=0)

# ─────────────────────────────────────────────
# 6. MODEL: FT-Transformer Style
#    Continuous features → linear projection → Transformer
#    Categorical features → embedding → Transformer
#    [CLS] token output → Residual MLP → prediction
# ─────────────────────────────────────────────
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.net(x))


class CICDTransformer(nn.Module):
    def __init__(self, cont_dim, cat_vocab_sizes, embed_dim=64,
                 n_heads=4, n_layers=3, mlp_dim=256, dropout=0.1):
        super().__init__()

        # Continuous feature projection: each feature → embed_dim token
        self.cont_proj = nn.Linear(cont_dim, cont_dim * embed_dim)
        self.cont_dim  = cont_dim
        self.embed_dim = embed_dim

        # Categorical embeddings
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(vocab, embed_dim) for vocab in cat_vocab_sizes
        ])

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout, batch_first=True,
            norm_first=True,   # Pre-LN for stable training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Residual MLP head
        total_tokens = cont_dim + len(cat_vocab_sizes) + 1  # +1 for CLS
        self.head = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            ResidualBlock(mlp_dim, dropout),
            ResidualBlock(mlp_dim, dropout),
            nn.Linear(mlp_dim, 1),
        )

    def forward(self, cont, cat):
        B = cont.size(0)

        # Project continuous features → (B, cont_dim, embed_dim)
        cont_tokens = self.cont_proj(cont).view(B, self.cont_dim, self.embed_dim)

        # Embed categorical features → (B, n_cat, embed_dim)
        cat_tokens = torch.stack(
            [emb(cat[:, i]) for i, emb in enumerate(self.cat_embeddings)], dim=1
        )

        # CLS token → (B, 1, embed_dim)
        cls = self.cls_token.expand(B, -1, -1)

        # Concatenate all tokens: [CLS | cont | cat]
        tokens = torch.cat([cls, cont_tokens, cat_tokens], dim=1)

        # Transformer
        out = self.transformer(tokens)

        # Take CLS output for prediction
        cls_out = out[:, 0, :]
        return self.head(cls_out).squeeze(-1)


cont_dim = X_train_cont.shape[1]
model = CICDTransformer(
    cont_dim=cont_dim,
    cat_vocab_sizes=cat_vocab_sizes,
    embed_dim=64,
    n_heads=4,
    n_layers=3,
    mlp_dim=256,
    dropout=0.15,
).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel parameters: {total_params:,}")

# ─────────────────────────────────────────────
# 7. TRAINING
# ─────────────────────────────────────────────
print("\nTraining...")

optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# Cosine annealing with warm restarts
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2, eta_min=1e-5
)

# Huber loss: robust to outliers
criterion = nn.HuberLoss(delta=1.0)

EPOCHS = 60
best_val_loss = float("inf")
best_state = None
patience = 10
no_improve = 0

for epoch in range(1, EPOCHS + 1):
    # ── Train ──
    model.train()
    train_loss = 0.0
    for cont_b, cat_b, y_b in train_loader:
        cont_b, cat_b, y_b = cont_b.to(DEVICE), cat_b.to(DEVICE), y_b.to(DEVICE)
        optimizer.zero_grad()
        pred = model(cont_b, cat_b)
        loss = criterion(pred, y_b)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # gradient clipping
        optimizer.step()
        train_loss += loss.item() * len(y_b)

    train_loss /= len(train_ds)
    scheduler.step()

    # ── Validate ──
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for cont_b, cat_b, y_b in test_loader:
            cont_b, cat_b, y_b = cont_b.to(DEVICE), cat_b.to(DEVICE), y_b.to(DEVICE)
            pred = model(cont_b, cat_b)
            val_loss += criterion(pred, y_b).item() * len(y_b)
    val_loss /= len(test_ds)

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        no_improve = 0
    else:
        no_improve += 1

    if epoch % 10 == 0 or epoch == 1:
        print(f"  Epoch {epoch:3d}/{EPOCHS} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

    if no_improve >= patience:
        print(f"  Early stopping at epoch {epoch}")
        break

# ─────────────────────────────────────────────
# 8. EVALUATION
# ─────────────────────────────────────────────
print("\nEvaluating best model...")
model.load_state_dict(best_state)
model.eval()

preds_log = []
with torch.no_grad():
    for cont_b, cat_b, _ in test_loader:
        cont_b, cat_b = cont_b.to(DEVICE), cat_b.to(DEVICE)
        preds_log.append(model(cont_b, cat_b).cpu().numpy())

y_pred_log = np.concatenate(preds_log)
y_pred = np.expm1(np.clip(y_pred_log, 0, None))

rmse = np.sqrt(mean_squared_error(y_test_raw, y_pred))
mae  = mean_absolute_error(y_test_raw, y_pred)
r2   = r2_score(y_test_raw, y_pred)

print("\n" + "★" * 6)
print("RESULTS — FT-Transformer")
print("★" * 6)
print(f"RMSE : {rmse:.4f} sec")
print(f"MAE  : {mae:.4f} sec")
print(f"R²   : {r2:.4f}")
print("★" * 6)

# Save model
torch.save({
    "model_state": best_state,
    "cont_dim": cont_dim,
    "cat_vocab_sizes": cat_vocab_sizes,
    "scaler": scaler,
    "tfidf": tfidf,
    "svd": svd,
    "loo": loo,
}, "cicd_transformer.pt")
print("\nModel saved to cicd_transformer.pt")