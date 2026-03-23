# Clustering Feature

Group users by taste profile using unsupervised learning — K-Means and DBSCAN.

> **Why clustering?** Unlike classification (predicting labels) or regression (predicting values), clustering finds natural groupings in the data with no labels required. Users in the same cluster have similar watching habits and genre preferences.

---

## Dataset

Source: `../dataset/`

| File | Used for |
|---|---|
| `ratings.csv` | User rating history — features and behaviour |
| `movies.csv` | Genre extraction per movie |

---

## Feature Engineering

One row per user — **42 features** total, all standardised with `StandardScaler`.

### User statistics (3)

| Feature | Description |
|---|---|
| `user_mean` | Overall average rating this user gives |
| `user_count` | Total number of movies they have rated |
| `user_std` | Standard deviation of ratings — how discriminating they are |

### Genre watch-share (19)

For each genre, the fraction of the user's total ratings that fall in that genre:

```
genre_share[user, genre] = count(ratings in genre) / total_ratings[user]
```

A user who has rated 60% Drama and 30% Comedy has values `Drama=0.60, Comedy=0.30`.
This captures what users *choose to watch*, independent of the rating they give.

### SVD latent features (20)

`TruncatedSVD(n_components=20)` applied to the mean-centred user-item matrix.
Captures collaborative taste patterns — which "taste groups" a user resembles — without leaking genre labels.

---

## Algorithms

### 1. K-Means

Partitions users into K non-overlapping clusters by minimising within-cluster variance (inertia).

**How it works:**
```
1. Initialise K centroids randomly (k-means++ seeding)
2. Assign each user to nearest centroid (Euclidean in scaled space)
3. Recompute centroids as cluster means
4. Repeat until convergence
```

**K selection — silhouette score:**
```
silhouette(i) = (b_i - a_i) / max(a_i, b_i)
  a_i = mean distance to other users in same cluster
  b_i = mean distance to nearest other cluster
range: -1 (wrong cluster) to +1 (dense, well-separated)
```
K is tried from 2 to 10; the K with the highest mean silhouette score is selected.

**Hyperparameters:**

| Parameter | Value |
|---|---|
| K range searched | 2 – 10 |
| Selection criterion | Silhouette score |
| `n_init` | 10 |
| `random_state` | 42 |

**Saved**: `models/clustering.pkl` (key: `kmeans`)

---

### 2. DBSCAN

Density-Based Spatial Clustering of Applications with Noise.
Groups dense regions of the feature space; sparse (isolated) users are labelled noise (−1).

**How it works:**
```
For each user p:
  Find all users within radius eps (eps-neighbourhood)
  If |neighbourhood| >= min_samples  → p is a core point
  Core points and their reachable neighbours form a cluster
  Users not reachable from any core point  → noise (-1)
```

**eps auto-tuning:**
```
1. Compute 5-nearest-neighbour distances for all users
2. eps = 90th percentile of those distances
```
This avoids manual tuning — it sets eps just above the typical density of the dataset.

**Hyperparameters:**

| Parameter | Value |
|---|---|
| `eps` | Auto (90th pct of 5-NN distances) |
| `min_samples` | 5 |

**Key differences from K-Means:**
- No K to specify upfront
- Discovers clusters of arbitrary shape
- Noise users get label −1 (not forced into a cluster)
- Cannot predict cluster for genuinely new users (treated as noise)

**Saved**: `models/clustering.pkl` (key: `dbscan`)

---

## Cluster Profiles

After clustering, each cluster is summarised:

| Field | Description |
|---|---|
| `size` | Number of users in this cluster |
| `top_genres` | Genres with mean watch-share > 8%, sorted descending, top 5 |
| `user_mean` | Average rating given across all cluster members |
| `user_count` | Average number of movies rated per user |

---

## Saved Artifacts (`models/`)

| File | Contents |
|---|---|
| `clustering.pkl` | KMeans model, DBSCAN model, StandardScaler, user→cluster assignments (kmeans + dbscan), cluster profiles, feature building data |

Single file for simplicity — everything needed is in `clustering.pkl`.

---

## API

| Endpoint | Description |
|---|---|
| `GET /` | HTML UI |
| `GET /users` | List of valid user IDs |
| `GET /cluster?user_id=42&model=kmeans` | Get cluster assignment and profile |

**Response:**
```json
{
  "user_id": 42,
  "model": "kmeans",
  "cluster_id": 2,
  "profile": {
    "size": 187,
    "top_genres": ["Drama", "Comedy", "Thriller"],
    "user_mean": 3.72,
    "user_count": 148.3
  },
  "all_clusters": [...]
}
```

Cluster ID −1 (DBSCAN only) means the user was classified as noise.

---

## Usage

```bash
# Step 1 — train once
python clustering/train.py

# Step 2 — start API
uvicorn clustering.main:app --reload --port 8003
```

Open `http://127.0.0.1:8003`. Enter a User ID, pick K-Means or DBSCAN, and see which cluster the user belongs to alongside all cluster profiles.
