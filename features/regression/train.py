"""
train.py — Train Ridge, Random Forest, and XGBoost regression models.
Predicts the exact rating a user would give a specific movie.
Run once: python regression/train.py
"""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MultiLabelBinarizer
from xgboost import XGBRegressor

DATA_DIR   = Path(__file__).parent.parent.parent / "dataset"
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("Loading data...")
ratings = pd.read_csv(DATA_DIR / "ratings.csv")
movies  = pd.read_csv(DATA_DIR / "movies.csv")

train_df, test_df = train_test_split(ratings, test_size=0.2, random_state=42)
train_df = train_df.reset_index(drop=True)
test_df  = test_df.reset_index(drop=True)

# ---------------------------------------------------------------------------
# 2. Shared statistics
# ---------------------------------------------------------------------------
global_mean = float(train_df["rating"].mean())

user_stats  = train_df.groupby("userId")["rating"].agg(user_mean="mean",  user_count="count")
movie_stats = train_df.groupby("movieId")["rating"].agg(movie_mean="mean", movie_count="count")

# ---------------------------------------------------------------------------
# 3. Genre one-hot encoding
# ---------------------------------------------------------------------------
movies["genre_list"] = movies["genres"].apply(
    lambda g: g.split("|") if g != "(no genres listed)" else []
)
mlb = MultiLabelBinarizer()
genre_matrix = mlb.fit_transform(movies["genre_list"])
genre_df     = pd.DataFrame(genre_matrix, columns=[f"genre_{g}" for g in mlb.classes_])
genre_df["movieId"] = movies["movieId"].values
genre_cols   = [c for c in genre_df.columns if c != "movieId"]

# ---------------------------------------------------------------------------
# 4. SVD user latent features
# ---------------------------------------------------------------------------
print("Computing SVD latent features...")
matrix      = train_df.pivot(index="userId", columns="movieId", values="rating").fillna(0)
matrix_vals = matrix.values.astype(float)
user_index  = {uid: i for i, uid in enumerate(matrix.index)}
movie_index = {mid: i for i, mid in enumerate(matrix.columns)}

counts    = (matrix_vals != 0).sum(1)
row_means = np.where(counts > 0, matrix_vals.sum(1) / np.maximum(counts, 1), global_mean)

matrix_centred = matrix_vals.copy()
for i in range(matrix_centred.shape[0]):
    mask = matrix_centred[i] != 0
    matrix_centred[i, mask] -= row_means[i]

svd     = TruncatedSVD(n_components=50, random_state=42)
U_sigma = svd.fit_transform(matrix_centred)   # (n_users, 50)
svd_cols = [f"svd_{i}" for i in range(U_sigma.shape[1])]
svd_user_df = pd.DataFrame(U_sigma, index=matrix.index, columns=svd_cols).reset_index()

# ---------------------------------------------------------------------------
# 5. Feature builder
# ---------------------------------------------------------------------------
feat_cols = ["user_mean", "user_count", "movie_mean", "movie_count"] + svd_cols + genre_cols

def build_features(df):
    d = df[["userId", "movieId"]].copy()
    d = d.merge(user_stats,  on="userId",  how="left")
    d = d.merge(movie_stats, on="movieId", how="left")
    d = d.merge(svd_user_df, on="userId",  how="left")
    d = d.merge(genre_df,    on="movieId", how="left")
    d["user_mean"]   = d["user_mean"].fillna(global_mean)
    d["user_count"]  = d["user_count"].fillna(0)
    d["movie_mean"]  = d["movie_mean"].fillna(global_mean)
    d["movie_count"] = d["movie_count"].fillna(0)
    d[svd_cols]   = d[svd_cols].fillna(0)
    d[genre_cols] = d[genre_cols].fillna(0)
    return d[feat_cols].values

print("Building feature matrices...")
X_train = build_features(train_df)
y_train = train_df["rating"].values
X_test  = build_features(test_df)
y_test  = test_df["rating"].values

print(f"  Features: {X_train.shape[1]}  |  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ---------------------------------------------------------------------------
# 6. Train models
# ---------------------------------------------------------------------------
def evaluate(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    print(f"  {name:20s}  RMSE={rmse:.4f}  MAE={mae:.4f}")

# ---- Ridge ----
print("Training Ridge...")
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
evaluate("Ridge", y_test, np.clip(ridge.predict(X_test), 0.5, 5.0))
joblib.dump(ridge, MODELS_DIR / "ridge.pkl")

# ---- Random Forest ----
print("Training Random Forest...")
rf = RandomForestRegressor(n_estimators=200, max_depth=12,
                            min_samples_leaf=4, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
evaluate("Random Forest", y_test, np.clip(rf.predict(X_test), 0.5, 5.0))
joblib.dump(rf, MODELS_DIR / "rf.pkl")

# ---- XGBoost ----
print("Training XGBoost...")
xgb = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                   subsample=0.8, colsample_bytree=0.8,
                   random_state=42, n_jobs=-1, verbosity=0)
xgb.fit(X_train, y_train)
evaluate("XGBoost", y_test, np.clip(xgb.predict(X_test), 0.5, 5.0))
joblib.dump(xgb, MODELS_DIR / "xgb.pkl")

# ---------------------------------------------------------------------------
# 7. Save shared metadata (dict lookups for fast prediction)
# ---------------------------------------------------------------------------
joblib.dump({
    "global_mean":       global_mean,
    "user_mean_dict":    user_stats["user_mean"].to_dict(),
    "user_count_dict":   user_stats["user_count"].to_dict(),
    "movie_mean_dict":   movie_stats["movie_mean"].to_dict(),
    "movie_count_dict":  movie_stats["movie_count"].to_dict(),
    "user_index":        user_index,
    "U_sigma":           U_sigma,
    "svd_cols":          svd_cols,
    "genre_df":          genre_df,
    "genre_cols":        genre_cols,
    "feat_cols":         feat_cols,
    "movies":            movies[["movieId", "title", "genres"]],
    "ratings":           ratings[["userId", "movieId", "rating"]],
}, MODELS_DIR / "meta.pkl")
print("  Metadata saved.")

print("\nDone. All artifacts in regression/models/")
