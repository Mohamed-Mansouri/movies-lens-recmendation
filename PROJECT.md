# MovieLens Recommendation System — Project Overview

End-to-end machine learning project built on the **MovieLens Latest Small** dataset (`ml-latest-small`).
Four independent ML features, each with its own training script, FastAPI server, and HTML UI.

---

## Dataset

Source: `dataset/` (ml-latest-small)

| File | Contents |
|---|---|
| `ratings.csv` | 100,836 ratings · 610 users · 9,742 movies · scale 0.5–5.0 |
| `movies.csv` | Movie titles and pipe-separated genre strings |

---

## Project Structure

```
movies lens recmendation/
├── dataset/
│   ├── ratings.csv
│   └── movies.csv
│
├── features/
│   ├── recommendation/        # Feature 1 — Collaborative Filtering
│   │   ├── train.py           # Train SVD, KNN, XGBoost → models/
│   │   ├── main.py            # FastAPI · GET /recommend
│   │   ├── static/index.html  # HTML UI
│   │   ├── README.md
│   │   └── models/            # svd.pkl  knn.pkl  xgb.pkl  meta.pkl
│   │
│   ├── regression/            # Feature 2 — Rating Prediction
│   │   ├── train.py           # Train Ridge, RandomForest, XGBoost → models/
│   │   ├── main.py            # FastAPI · GET /predict
│   │   ├── static/index.html
│   │   ├── README.md
│   │   └── models/            # ridge.pkl  rf.pkl  xgb.pkl  meta.pkl
│   │
│   ├── classification/        # Feature 3 — Genre Interest Classification
│   │   ├── train.py           # Train LR, RandomForest, XGBoost (multi-label) → models/
│   │   ├── main.py            # FastAPI · GET /classify
│   │   ├── static/index.html
│   │   ├── README.md
│   │   └── models/            # lr.pkl  rf.pkl  xgb.pkl  meta.pkl
│   │
│   └── clustering/            # Feature 4 — User Taste Clustering
│       ├── train.py           # Train K-Means + DBSCAN → models/
│       ├── main.py            # FastAPI · GET /cluster
│       ├── static/index.html
│       ├── README.md
│       └── models/            # clustering.pkl
│
├── app/                   # Unified entry point
│   ├── main.py            # Mounts all 4 sub-apps + dashboard
│   └── static/index.html  # Central dashboard UI
│
└── PROJECT.md             # ← this file
```

---

## Feature 1 — Recommendation (`features/recommendation/`)

**Question answered**: *"Given a user, what movies should they watch next?"*

**Port**: `8000`

### How it works
Each model scores all movies the user has not yet seen, then returns the top N.

### Models

| Model | Approach | Features |
|---|---|---|
| **SVD** | Truncated SVD on the mean-centred user-item matrix; reconstruct to get full predicted-rating matrix | User-item matrix (610×9742) → 100 latent dims |
| **KNN** | User-based collaborative filtering; finds 40 most similar users by cosine similarity; predicts via mean-centred weighted average | User-item matrix as row vectors |
| **XGBoost** | Regression per (user, movie) pair; uses engineered tabular features | `user_mean`, `user_count`, `movie_mean`, `movie_count`, 100 SVD latent dims |

### Key design decisions
- **Compact artifact storage**: Only `U_sigma`, `Vt`, `row_means` are saved (not the reconstructed matrix). Reconstruction happens once on load. Load time: ~3 s vs. ~30 s.
- **Integer indexing**: All predictions use `arr[user_idx, movie_idx]` — no pandas `.loc[]` overhead.
- **Hot reload**: Models reload automatically if `.pkl` files change (no server restart needed after retraining).

### API
```
GET /recommend?user_id=42&model=svd&n=10
GET /users
```

---

## Feature 2 — Regression (`features/regression/`)

**Question answered**: *"What rating would user X give movie Y?"*

**Port**: `8001`

### How it works
Predict a continuous rating value (0.5–5.0) for a specific (user, movie) pair.

### Features (74 per sample)

| Group | Count | Description |
|---|---|---|
| User stats | 2 | `user_mean`, `user_count` |
| Movie stats | 2 | `movie_mean`, `movie_count` |
| SVD latent | 50 | User taste vector from `TruncatedSVD(n_components=50)` on mean-centred matrix |
| Genre one-hot | 20 | Binary flags per genre via `MultiLabelBinarizer` |

### Models

| Model | Approach |
|---|---|
| **Ridge** | Linear regression with L2 regularisation (`alpha=1.0`). Baseline — fast, interpretable. |
| **Random Forest** | 200 trees, bootstrap bagging, max_depth=12. Handles non-linear feature interactions. |
| **XGBoost** | 300 sequential boosting trees, learning_rate=0.05. Highest accuracy. |

### API
```
GET /predict?user_id=42&movie_id=318&model=xgboost
GET /users
GET /movies
```
Response includes `predicted_rating`, `actual_rating` (null if not in dataset), `title`, `genres`.

---

## Feature 3 — Classification (`features/classification/`)

**Question answered**: *"Which genres is this user interested in?"*

**Port**: `8002`

### How it works
Multi-label classification — each genre is a binary label (interested / not interested).
The UI displays only the **top genre** (highest-confidence positive prediction).

### Label construction (Y matrix)

A user is labelled as interested in a genre if **either** signal is true:

| Signal | Rule | Reasoning |
|---|---|---|
| **Engagement** | `genre_count / total_ratings >= 10%` | Watching lots of a genre = revealed preference, regardless of star rating given. A user who rates 41% Drama at 0.75★ is a Drama fan who had bad luck with specific films. |
| **Quality** | `genre_avg > user_overall_mean` AND `genre_count >= 2` | Genres the user rates above their own personal average = hidden preference even if rarely watched. |

```
Y[user, genre] = engagement OR quality
```

### Features (53 per user)

| Group | Count | Description |
|---|---|---|
| User stats | 3 | `user_mean`, `user_count`, `user_std` |
| SVD latent | 50 | From `TruncatedSVD(n_components=50)` on mean-centred matrix |

> Genre averages are deliberately excluded from X — they would be direct data leakage into Y.

### Models

| Model | Approach |
|---|---|
| **Logistic Regression** | `MultiOutputClassifier` wrapping one LR per genre. Linear decision boundary per genre. |
| **Random Forest** | `MultiOutputClassifier` wrapping one RF per genre (200 trees, max_depth=12). |
| **XGBoost** | `MultiOutputClassifier` wrapping one XGBClassifier per genre (200 trees, logloss). |

### Known vs unknown users
- **Known user**: actual Y label is used directly (bypasses model to avoid prediction errors for users with enough history).
- **Unknown user**: model prediction used as fallback.

### Evaluation metrics
- **F1-micro** — global F1 across all genre-user pairs (favours common genres)
- **F1-macro** — average F1 per genre (treats all genres equally)
- **Hamming Loss** — fraction of label-user pairs predicted incorrectly

### API
```
GET /classify?user_id=42&model=xgboost
GET /users
```

---

## Feature 4 — Clustering (`features/clustering/`)

**Question answered**: *"Which group of similar users does this user belong to?"*

**Port**: `8003`

### How it works
Unsupervised grouping — no labels required. Users are grouped by taste profile.
The UI shows the user's cluster and all clusters with their dominant genres.

### Features (42 per user, all standardised with `StandardScaler`)

| Group | Count | Description |
|---|---|---|
| User stats | 3 | `user_mean`, `user_count`, `user_std` |
| Genre watch-share | 19 | `count(genre) / total_ratings` — what users choose to watch |
| SVD latent | 20 | From `TruncatedSVD(n_components=20)` on mean-centred matrix |

### Algorithms

#### K-Means
Partitions users into K non-overlapping clusters by minimising inertia.
- K searched over 2–10; best K chosen by highest **silhouette score**
- Silhouette score: how similar a user is to their own cluster vs. the nearest other cluster (range −1 to +1)
- Every user is assigned exactly one cluster

#### DBSCAN
Density-based clustering — groups dense regions, marks sparse users as noise (−1).
- `eps` auto-tuned: 90th percentile of 5-nearest-neighbour distances (avoids manual tuning)
- `min_samples = 5`
- Does not require K; discovers arbitrary-shaped clusters
- Noise label (−1) = user's taste is too distinct to fit any dense group

### Cluster profiles
Each cluster is described by: top genres (watch-share > 8%), average rating, average movie count.

### API
```
GET /cluster?user_id=42&model=kmeans
GET /users
```
Response includes `cluster_id`, profile, and a list of all clusters.

---

## Shared Patterns Across All Features

### Hot model reload
All four APIs reload models without restarting the server:
```python
def _pkl_mtime():
    pkls = list(MODELS_DIR.glob("*.pkl"))
    return max((p.stat().st_mtime for p in pkls), default=0) if pkls else 0

def _ensure_fresh():
    if _pkl_mtime() > state["loaded_at"]:
        _load()
```
Every request calls `_ensure_fresh()`. If `train.py` has been run since the last load, models are reloaded on the next request.

### SVD for latent features
All four features use `TruncatedSVD` on a **mean-centred** user-item matrix to produce latent user features:
- Recommendation: 100 components (primary model)
- Regression: 50 components (part of 74-feature vector)
- Classification: 50 components (part of 53-feature vector)
- Clustering: 20 components (part of 42-feature vector)

Mean-centering removes individual rating bias before decomposition so SVD focuses on relative preference patterns.

### Compact storage
Artifacts are stored as **model components** (e.g. `U_sigma`, `Vt`) — not as reconstructed matrices or DataFrames. Reconstruction is done once on load. This keeps file sizes small (8 MB vs. 47 MB) and load times fast (~2–3 s).

### HTML UI
Every feature has a dark-themed HTML page with:
- Input controls (user ID dropdown/input, model selector)
- Results display specific to the feature
- **Collapsible right-side panel** explaining the full pipeline, features, and model mechanics

---

## Quick Start

```bash
# Install dependencies
pip install fastapi uvicorn scikit-learn xgboost pandas numpy joblib

# Step 1 — Train all features (run once each)
python features/recommendation/train.py
python features/regression/train.py
python features/classification/train.py
python features/clustering/train.py

# Step 2 — Start the unified app (all features on one server)
uvicorn app.main:app --reload --port 8000
```

Open `http://127.0.0.1:8000` — dashboard with all 4 features accessible via tabs.

| Path | Feature |
|---|---|
| `/` | Dashboard |
| `/recommendation/` | Recommendation UI |
| `/regression/` | Regression UI |
| `/classification/` | Classification UI |
| `/clustering/` | Clustering UI |

**Run features individually** (optional):
```bash
uvicorn features.recommendation.main:app --reload --port 8000
uvicorn features.regression.main:app     --reload --port 8001
uvicorn features.classification.main:app --reload --port 8002
uvicorn features.clustering.main:app     --reload --port 8003
```
