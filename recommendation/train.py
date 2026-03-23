"""
train.py — Train all 4 recommendation models and save compact artifacts to models/
Run once: python train.py
"""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from xgboost import XGBRegressor

DATA_DIR   = Path(__file__).parent.parent / "dataset"
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("Loading data...")
ratings = pd.read_csv(DATA_DIR / "ratings.csv")
movies  = pd.read_csv(DATA_DIR / "movies.csv")

train_df, _ = train_test_split(ratings, test_size=0.2, random_state=42)
train_df = train_df.reset_index(drop=True)

# ---------------------------------------------------------------------------
# 2. User-item matrix + shared stats
# ---------------------------------------------------------------------------
matrix      = train_df.pivot(index="userId", columns="movieId", values="rating").fillna(0)
matrix_vals = matrix.values.astype(float)
global_mean = float(train_df["rating"].mean())
user_means  = train_df.groupby("userId")["rating"].mean().to_dict()
user_index  = {uid: i for i, uid in enumerate(matrix.index)}
movie_index = {mid: i for i, mid in enumerate(matrix.columns)}

counts    = (matrix_vals != 0).sum(1)
row_means = np.where(counts > 0, matrix_vals.sum(1) / np.maximum(counts, 1), global_mean)

# ---------------------------------------------------------------------------
# 3. SVD — save compact components, NOT the full reconstructed matrix
# ---------------------------------------------------------------------------
print("Training SVD...")
matrix_centred = matrix_vals.copy()
for i in range(matrix_centred.shape[0]):
    mask = matrix_centred[i] != 0
    matrix_centred[i, mask] -= row_means[i]

svd     = TruncatedSVD(n_components=100, random_state=42)
U_sigma = svd.fit_transform(matrix_centred)   # (n_users, 100)
Vt      = svd.components_                     # (100, n_movies)

# Store components only — ~8 MB instead of ~47 MB for the full matrix
joblib.dump({"U_sigma": U_sigma, "Vt": Vt, "row_means": row_means}, MODELS_DIR / "svd.pkl")
print("  SVD saved.")

# ---------------------------------------------------------------------------
# 4. KNN — save distances + sparse matrix_vals
# ---------------------------------------------------------------------------
print("Training KNN...")
knn = NearestNeighbors(n_neighbors=41, metric="cosine", algorithm="brute")
knn.fit(matrix_vals)
knn_distances, knn_indices = knn.kneighbors(matrix_vals)

joblib.dump({
    "matrix_vals":    matrix_vals,
    "row_means":      row_means,
    "knn_distances":  knn_distances,
    "knn_indices":    knn_indices,
}, MODELS_DIR / "knn.pkl")
print("  KNN saved.")

# ---------------------------------------------------------------------------
# 6. XGBoost — precompute dict lookups for fast prediction
# ---------------------------------------------------------------------------
print("Training XGBoost...")
user_stats  = train_df.groupby("userId")["rating"].agg(user_mean="mean",  user_count="count")
movie_stats = train_df.groupby("movieId")["rating"].agg(movie_mean="mean", movie_count="count")

# Build feature matrix for training
svd_features = U_sigma[np.array([user_index[uid] for uid in train_df["userId"]])]
X_train = np.column_stack([
    train_df["userId"].map(user_stats["user_mean"]).fillna(global_mean),
    train_df["userId"].map(user_stats["user_count"]).fillna(0),
    train_df["movieId"].map(movie_stats["movie_mean"]).fillna(global_mean),
    train_df["movieId"].map(movie_stats["movie_count"]).fillna(0),
    svd_features,
])
y_train = train_df["rating"].values

xgb = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                   subsample=0.8, colsample_bytree=0.8,
                   random_state=42, n_jobs=-1, verbosity=0)
xgb.fit(X_train, y_train)

# Save dicts (O(1) lookup at prediction time — no DataFrame merges)
joblib.dump({
    "model":            xgb,
    "user_mean_dict":   user_stats["user_mean"].to_dict(),
    "user_count_dict":  user_stats["user_count"].to_dict(),
    "movie_mean_dict":  movie_stats["movie_mean"].to_dict(),
    "movie_count_dict": movie_stats["movie_count"].to_dict(),
    "U_sigma":          U_sigma,   # for user latent features
}, MODELS_DIR / "xgb.pkl")
print("  XGBoost saved.")

# ---------------------------------------------------------------------------
# 7. Shared metadata
# ---------------------------------------------------------------------------
joblib.dump({
    "movies":         movies,
    "train_df":       train_df,
    "user_means":     user_means,
    "global_mean":    global_mean,
    "user_index":     user_index,
    "movie_index":    movie_index,
    "matrix_columns": list(matrix.columns),
}, MODELS_DIR / "meta.pkl")
print("  Metadata saved.")

print("\nDone. All artifacts in models/")
