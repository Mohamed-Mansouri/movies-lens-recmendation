"""
train.py — Multi-label genre classification.
Given a user_id, predict which genres they are interested in.
Run once: python classification/train.py
"""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, hamming_loss, classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from xgboost import XGBClassifier

DATA_DIR   = Path(__file__).parent.parent / "dataset"
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("Loading data...")
ratings = pd.read_csv(DATA_DIR / "ratings.csv")
movies  = pd.read_csv(DATA_DIR / "movies.csv")

# ---------------------------------------------------------------------------
# 2. Build user-genre interaction table
#    Explode each rating into one row per genre
# ---------------------------------------------------------------------------
movies["genre_list"] = movies["genres"].apply(
    lambda g: [x for x in g.split("|") if x != "(no genres listed)"]
)
# Keep only genres that appear enough to be meaningful
all_genres = sorted({g for lst in movies["genre_list"] for g in lst})

# Join ratings with genre lists, then explode
rated = ratings.merge(movies[["movieId", "genre_list"]], on="movieId", how="left")
rated = rated.explode("genre_list").rename(columns={"genre_list": "genre"})
rated = rated.dropna(subset=["genre"])

# Per-user, per-genre: average rating and count
user_genre = (
    rated.groupby(["userId", "genre"])["rating"]
    .agg(avg="mean", cnt="count")
    .reset_index()
)

# ---------------------------------------------------------------------------
# 3. Build target matrix Y  (one row per user)
#
#    Interest = engagement (volume) OR quality above personal mean.
#
#    Signal 1 — ENGAGEMENT: user watches ≥10% of their total movies in this
#               genre → they keep coming back regardless of rating given.
#               (50 action @ 3★ beats 1 comedy @ 5★)
#
#    Signal 2 — QUALITY: user rates this genre above their own personal mean
#               AND has rated ≥2 movies in it (avoids single-movie noise).
#
#    Label = 1 if EITHER signal is true.
# ---------------------------------------------------------------------------
ENGAGEMENT_SHARE = 0.10   # ≥10% of total ratings in this genre
MIN_RATINGS      = 2      # minimum ratings for quality signal to count

user_ids = sorted(ratings["userId"].unique())

# Pivot counts and averages (users × genres)
genre_avg = user_genre.pivot(index="userId", columns="genre", values="avg").reindex(
    index=user_ids, columns=all_genres
).fillna(0)

genre_cnt = user_genre.pivot(index="userId", columns="genre", values="cnt").reindex(
    index=user_ids, columns=all_genres
).fillna(0)

# Signal 1: relative watch share per genre
user_total   = genre_cnt.sum(axis=1).replace(0, 1)          # total rated movies per user
genre_share  = genre_cnt.div(user_total, axis=0)             # fraction per genre
engaged      = genre_share >= ENGAGEMENT_SHARE               # watches ≥10% in this genre

# Signal 2: rates this genre above their personal mean
user_mean    = ratings.groupby("userId")["rating"].mean()
above_mean   = genre_avg.sub(user_mean, axis=0) > 0          # genre avg > personal avg
quality      = above_mean & (genre_cnt >= MIN_RATINGS)        # with enough data

Y = (engaged | quality).astype(int)

print(f"  Genres  : {len(all_genres)}")
print(f"  Users   : {len(user_ids)}")
print(f"  Label density: {Y.values.mean():.2%}  (avg labels per user: {Y.sum(axis=1).mean():.1f})")

# ---------------------------------------------------------------------------
# 4. Feature engineering  (one row per user)
# ---------------------------------------------------------------------------
# Basic user stats
user_stats = ratings.groupby("userId")["rating"].agg(
    user_mean="mean",
    user_count="count",
    user_std="std",
).fillna(0)

# SVD latent features from user-item matrix
matrix      = ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0)
matrix_vals = matrix.values.astype(float)
user_index  = {uid: i for i, uid in enumerate(matrix.index)}

counts    = (matrix_vals != 0).sum(1)
row_means = np.where(counts > 0, matrix_vals.sum(1) / np.maximum(counts, 1), 0)
matrix_centred = matrix_vals.copy()
for i in range(matrix_centred.shape[0]):
    mask = matrix_centred[i] != 0
    matrix_centred[i, mask] -= row_means[i]

svd     = TruncatedSVD(n_components=50, random_state=42)
U_sigma = svd.fit_transform(matrix_centred)
svd_cols = [f"svd_{i}" for i in range(U_sigma.shape[1])]
svd_df   = pd.DataFrame(U_sigma, index=matrix.index, columns=svd_cols)

# Combine all user features
X_df = user_stats.join(svd_df, how="left").reindex(user_ids).fillna(0)
feat_cols = list(X_df.columns)

X = X_df.values
print(f"  Features: {X.shape[1]}  (user stats: 3, SVD latent: 50)")

# ---------------------------------------------------------------------------
# 5. Train / test split at USER level
# ---------------------------------------------------------------------------
X_train, X_test, Y_train, Y_test, uid_train, uid_test = train_test_split(
    X, Y.values, user_ids, test_size=0.2, random_state=42
)

# ---------------------------------------------------------------------------
# 6. Train models
# ---------------------------------------------------------------------------
def evaluate(name, Y_true, Y_pred):
    f1_micro = f1_score(Y_true, Y_pred, average="micro", zero_division=0)
    f1_macro = f1_score(Y_true, Y_pred, average="macro", zero_division=0)
    hl       = hamming_loss(Y_true, Y_pred)
    print(f"  {name:25s}  F1-micro={f1_micro:.4f}  F1-macro={f1_macro:.4f}  Hamming={hl:.4f}")

# ---- Logistic Regression ----
print("Training Logistic Regression...")
lr_base = LogisticRegression(C=1.0, max_iter=500, random_state=42)
lr      = MultiOutputClassifier(lr_base, n_jobs=-1)
lr.fit(X_train, Y_train)
evaluate("Logistic Regression", Y_test, lr.predict(X_test))
joblib.dump(lr, MODELS_DIR / "lr.pkl")

# ---- Random Forest ----
print("Training Random Forest...")
rf_base = RandomForestClassifier(n_estimators=200, max_depth=12,
                                  min_samples_leaf=4, random_state=42, n_jobs=-1)
rf      = MultiOutputClassifier(rf_base, n_jobs=1)
rf.fit(X_train, Y_train)
evaluate("Random Forest", Y_test, rf.predict(X_test))
joblib.dump(rf, MODELS_DIR / "rf.pkl")

# ---- XGBoost ----
print("Training XGBoost...")
xgb_base = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8,
                          use_label_encoder=False, eval_metric="logloss",
                          random_state=42, n_jobs=-1, verbosity=0)
xgb = MultiOutputClassifier(xgb_base, n_jobs=1)
xgb.fit(X_train, Y_train)
evaluate("XGBoost", Y_test, xgb.predict(X_test))
joblib.dump(xgb, MODELS_DIR / "xgb.pkl")

# ---------------------------------------------------------------------------
# 7. Save metadata
# ---------------------------------------------------------------------------
joblib.dump({
    "all_genres":   all_genres,
    "feat_cols":    feat_cols,
    "user_index":   user_index,
    "U_sigma":      U_sigma,
    "svd_cols":     svd_cols,
    "user_stats":   user_stats.to_dict(),   # dict of dicts for O(1) lookup
    "global_mean":  float(ratings["rating"].mean()),
    "user_ids":     user_ids,
}, MODELS_DIR / "meta.pkl")
print("  Metadata saved.")

print("\nDone. All artifacts in classification/models/")
