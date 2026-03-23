"""
train.py — Cluster users by taste profile using K-Means and DBSCAN.
Run once: python clustering/train.py
"""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

DATA_DIR   = Path(__file__).parent.parent.parent / "dataset"
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("Loading data...")
ratings = pd.read_csv(DATA_DIR / "ratings.csv")
movies  = pd.read_csv(DATA_DIR / "movies.csv")

# ---------------------------------------------------------------------------
# 2. Build per-user features
# ---------------------------------------------------------------------------
movies["genre_list"] = movies["genres"].apply(
    lambda g: [x for x in g.split("|") if x != "(no genres listed)"]
)
all_genres = sorted({g for lst in movies["genre_list"] for g in lst})

# Explode ratings by genre
rated     = ratings.merge(movies[["movieId", "genre_list"]], on="movieId")
rated_exp = rated.explode("genre_list").rename(columns={"genre_list": "genre"})

# Per-user genre watch share
user_genre_cnt = (
    rated_exp.groupby(["userId", "genre"])["rating"]
    .count()
    .unstack(fill_value=0)
    .reindex(columns=all_genres, fill_value=0)
)
user_totals    = user_genre_cnt.sum(axis=1).replace(0, 1)
genre_share_df = user_genre_cnt.div(user_totals, axis=0)   # share per genre

# Per-user basic stats
user_stats = ratings.groupby("userId")["rating"].agg(
    user_mean="mean", user_count="count", user_std="std"
).fillna(0)

# SVD latent features
matrix      = ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0)
matrix_vals = matrix.values.astype(float)
user_index  = {uid: i for i, uid in enumerate(matrix.index)}

counts    = (matrix_vals != 0).sum(1)
row_means = np.where(counts > 0, matrix_vals.sum(1) / np.maximum(counts, 1), 0)
mc = matrix_vals.copy()
for i in range(mc.shape[0]):
    mask = mc[i] != 0
    mc[i, mask] -= row_means[i]

svd     = TruncatedSVD(n_components=20, random_state=42)
U_sigma = svd.fit_transform(mc)
svd_cols = [f"svd_{i}" for i in range(U_sigma.shape[1])]
svd_df   = pd.DataFrame(U_sigma, index=matrix.index, columns=svd_cols)

# Combine all features
user_ids = sorted(ratings["userId"].unique())
X_df = (
    user_stats
    .join(genre_share_df, how="left")
    .join(svd_df, how="left")
    .reindex(user_ids)
    .fillna(0)
)
feat_cols = list(X_df.columns)
print(f"  Users: {len(user_ids)}  Features: {len(feat_cols)}")
print(f"  (user stats: 3, genre shares: {len(all_genres)}, SVD latent: 20)")

# ---------------------------------------------------------------------------
# 3. Scale features
# ---------------------------------------------------------------------------
scaler  = StandardScaler()
X_scaled = scaler.fit_transform(X_df.values)

# ---------------------------------------------------------------------------
# 4. K-Means — find best K using silhouette score
# ---------------------------------------------------------------------------
print("\nFitting K-Means (K=2..10)...")
best_k, best_score, best_km = 2, -1, None
results_k = []

for k in range(2, 11):
    km    = KMeans(n_clusters=k, random_state=42, n_init=10)
    labs  = km.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labs)
    results_k.append((k, score))
    print(f"  K={k}  silhouette={score:.4f}")
    if score > best_score:
        best_score, best_k, best_km = score, k, km

print(f"  Best K={best_k}  silhouette={best_score:.4f}")
kmeans_labels = best_km.labels_

# ---------------------------------------------------------------------------
# 5. DBSCAN — tune eps with nearest-neighbour distances
# ---------------------------------------------------------------------------
print("\nFitting DBSCAN...")
from sklearn.neighbors import NearestNeighbors

nn   = NearestNeighbors(n_neighbors=5)
nn.fit(X_scaled)
dists, _ = nn.kneighbors(X_scaled)
eps_auto = float(np.percentile(dists[:, -1], 90))   # 90th pct of 5-NN distances

dbscan        = DBSCAN(eps=eps_auto, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)

n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise    = (dbscan_labels == -1).sum()
print(f"  eps={eps_auto:.3f}  clusters={n_clusters}  noise={n_noise}")

# ---------------------------------------------------------------------------
# 6. Build cluster profiles (dominant genres + avg stats)
# ---------------------------------------------------------------------------
def build_profiles(labels, X_df, all_genres, algo_name):
    df = X_df.copy()
    df["cluster"] = labels
    profiles = {}
    for cid in sorted(df["cluster"].unique()):
        members = df[df["cluster"] == cid]
        genre_means = members[all_genres].mean().sort_values(ascending=False)
        top_genres  = genre_means[genre_means > 0.08].index.tolist()[:5]
        profiles[int(cid)] = {
            "size":       int(len(members)),
            "top_genres": top_genres,
            "user_mean":  round(float(members["user_mean"].mean()), 2),
            "user_count": round(float(members["user_count"].mean()), 1),
        }
    return profiles

kmeans_profiles = build_profiles(kmeans_labels, X_df, all_genres, "kmeans")
dbscan_profiles = build_profiles(dbscan_labels, X_df, all_genres, "dbscan")

print(f"\nK-Means clusters: {len(kmeans_profiles)}")
for cid, p in kmeans_profiles.items():
    print(f"  Cluster {cid}: {p['size']} users  top={p['top_genres'][:3]}")

print(f"\nDBSCAN clusters: {len(dbscan_profiles)}")
for cid, p in dbscan_profiles.items():
    label = "Noise" if cid == -1 else f"Cluster {cid}"
    print(f"  {label}: {p['size']} users  top={p['top_genres'][:3]}")

# ---------------------------------------------------------------------------
# 7. Save
# ---------------------------------------------------------------------------
user_kmeans = dict(zip(user_ids, kmeans_labels.tolist()))
user_dbscan = dict(zip(user_ids, dbscan_labels.tolist()))

joblib.dump({
    # Models
    "kmeans":          best_km,
    "dbscan":          dbscan,
    "scaler":          scaler,
    # Assignments
    "user_kmeans":     user_kmeans,    # uid -> cluster_id
    "user_dbscan":     user_dbscan,
    # Profiles
    "kmeans_profiles": kmeans_profiles,
    "dbscan_profiles": dbscan_profiles,
    # Feature building
    "user_index":      user_index,
    "U_sigma":         U_sigma,
    "svd_cols":        svd_cols,
    "user_stats":      user_stats.to_dict(),
    "genre_share_df":  genre_share_df,
    "all_genres":      all_genres,
    "feat_cols":       feat_cols,
    "global_mean":     float(ratings["rating"].mean()),
    "user_ids":        user_ids,
}, MODELS_DIR / "clustering.pkl")

print("\nDone. All artifacts in clustering/models/")
