# Movie Recommendation Feature

Collaborative filtering system built on the **MovieLens 10M** dataset.
Given a `user_id`, each model predicts ratings for all unseen movies and returns the top-N highest.

---

## Dataset

Source: `../dataset/` (ml-1m)

| File | Contents |
|---|---|
| `ratings.csv` | `userId, movieId, rating, timestamp` — 10,000,054 ratings |
| `movies.csv` | `movieId, title, genres` — 10,681 movies |

- 69,878 users · ratings on a 1.0 – 5.0 full-star scale
- **Train / Test split**: 80 / 20 (`random_state=42`)

---

## Data Preparation (shared across all models)

1. **Load** `ratings.csv` and split into train / test sets.

2. **Build user-item matrix**
   Pivot the training ratings into a `(69,878 users × 10,681 movies)` matrix.
   Missing entries (no rating) are filled with `0`.

   ```
         movie_1  movie_2  movie_3 ...
   user_1    4.0      0.0      3.5
   user_2    0.0      5.0      0.0
   ...
   ```

3. **Compute per-user means**
   For each user, compute their average rating over *observed* entries only (ignoring the 0-fills).
   Used as a fallback when a user or movie is unknown at prediction time.

4. **Compute row means (`row_means`)**
   Same as above but stored as a numpy array aligned to matrix row order — used by SVD mean-centering and KNN prediction.

---

## Models

### 1. SVD — Truncated Singular Value Decomposition

**Idea**: Decompose the user-item matrix into latent factors that capture hidden patterns (e.g. "action fans", "drama lovers"). Reconstruct the full matrix to get predicted ratings.

**Data preparation specific to SVD**:
- **Mean-center** the matrix before decomposition: for each user row, subtract their `row_mean` from every *observed* (non-zero) entry. This removes individual rating bias so the decomposition focuses on relative preferences.

**How it works**:
```
matrix_centred  →  TruncatedSVD  →  U_sigma (6040×100)  ·  Vt (100×3952)
                                     ↓
                   predicted = (U_sigma · Vt) + row_means   (add bias back)
```

**Input features**: the full mean-centred user-item matrix `(6040 × 3952)`

**Saved artifacts**:
- `U_sigma` — user latent matrix `(6040 × 100)`
- `Vt` — item latent matrix `(100 × 3952)`
- `row_means` — per-user mean rating `(6040,)`

**Prediction**: reconstruct `U_sigma @ Vt + row_means` once on load, then index directly with `rec[user_idx, movie_idx]`.

**Hyperparameters**:
| Parameter | Value |
|---|---|
| `n_components` | 100 latent factors |
| `random_state` | 42 |

---

### 2. KNN — User-based K-Nearest Neighbours

**Idea**: Find the K most similar users to the target user based on their rating history, then predict a rating using a weighted average of what those neighbours rated.

**No extra data preparation** beyond the shared user-item matrix.

**How it works**:
```
1. Measure cosine similarity between all user row vectors.
2. For a (user, movie) pair — find the 40 most similar users who rated that movie.
3. Predict using mean-centred weighted average:

   pred = user_mean + Σ(sim_k × (rating_k − mean_k)) / Σ|sim_k|
```

Mean-centering the prediction corrects for the fact that some users rate everything high or low.

**Input features**: the raw user-item matrix `(6040 × 3952)` (0 where no rating)

**Saved artifacts**:
- `matrix_vals` — the user-item matrix as numpy array
- `row_means` — per-user mean rating
- `knn_distances` — pre-computed cosine distances to 41 neighbours per user `(6040 × 41)`
- `knn_indices` — corresponding neighbour row indices `(6040 × 41)`

**Prediction**: use cached distances — no recomputation needed per request.

**Hyperparameters**:
| Parameter | Value |
|---|---|
| `n_neighbors` | 40 (+ 1 self) |
| `metric` | cosine |
| `algorithm` | brute |

---

### 3. XGBoost — Gradient Boosted Regression

**Idea**: Frame rating prediction as a standard regression problem. Engineer features that describe the user and the movie, then train XGBoost to predict the rating value. Unlike matrix factorization, it can incorporate any tabular signal.

**Data preparation specific to XGBoost**:
- Compute per-user and per-movie statistics from the training set.
- Reuse the SVD `U_sigma` matrix as user latent features — gives XGBoost 100 dimensions of learned user taste.

**Input features** (104 total per training sample):

| Feature | Description |
|---|---|
| `user_mean` | Average rating this user gives across all their ratings |
| `user_count` | Total number of ratings this user has made |
| `movie_mean` | Average rating this movie receives from all users |
| `movie_count` | Total number of ratings this movie has received |
| `svd_0` … `svd_99` | 100 SVD latent dimensions for this user (from `U_sigma`) |

**How it works**:
```
For each (userId, movieId) training pair:
  → build a 104-dim feature vector
  → XGBoost learns a regression tree ensemble to minimise MSE on rating
```

**Saved artifacts**:
- `model` — trained `XGBRegressor`
- `user_mean_dict` / `user_count_dict` — O(1) user stat lookup
- `movie_mean_dict` / `movie_count_dict` — O(1) movie stat lookup
- `U_sigma` — SVD user latent matrix for feature construction

**Prediction**: dict lookups + numpy row slice → single `xgb.predict()` call per (user, movie).

**Hyperparameters**:
| Parameter | Value |
|---|---|
| `n_estimators` | 300 trees |
| `max_depth` | 6 |
| `learning_rate` | 0.05 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |

---

## Saved Artifacts (`models/`)

| File | Contents | Approximate size |
|---|---|---|
| `svd.pkl` | `U_sigma`, `Vt`, `row_means` | ~8 MB |
| `knn.pkl` | `matrix_vals`, `row_means`, `knn_distances`, `knn_indices` | ~45 MB |
| `xgb.pkl` | XGBoost model + stat dicts + `U_sigma` | ~5 MB |
| `meta.pkl` | `movies`, `train_df`, index dicts, column list | ~5 MB |

> Artifacts are stored as compact components — **not** as reconstructed matrices — to keep file sizes small and load times fast (~2–3 s vs ~30 s).

---

## API

| Endpoint | Description |
|---|---|
| `GET /` | Serves the HTML UI |
| `GET /users` | Returns list of valid user IDs |
| `GET /recommend?user_id=42&model=svd&n=10` | Returns top-N recommendations |

**Model values**: `svd` · `knn` · `xgboost`

---

## Usage

```bash
# Step 1 — train once (~2 min)
python recommendation/train.py

# Step 2 — start API (loads in ~3 s)
uvicorn recommendation.main:app --reload
```

Then open `http://127.0.0.1:8000`.
