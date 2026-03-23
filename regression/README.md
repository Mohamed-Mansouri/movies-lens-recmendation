# Rating Regression Feature

Given a specific `user_id` and `movie_id`, predict the exact rating (0.5 – 5.0) that user would give that movie.

> **Difference from Recommendation**: Recommendation scores *all unseen movies* and returns a ranked list. Regression answers a single question: *"What rating would user X give movie Y?"*

---

## Dataset

Source: `../dataset/`

| File | Used for |
|---|---|
| `ratings.csv` | Training targets (`rating`) and user/movie statistics |
| `movies.csv` | Genre one-hot encoding |

- **Train / Test split**: 80 / 20 (`random_state=42`)

---

## Feature Engineering (shared across all models)

74 features are built per (user, movie) training sample:

### User features (2)
Computed from training ratings only.

| Feature | Description |
|---|---|
| `user_mean` | Average rating given by this user across all their ratings |
| `user_count` | Total number of movies this user has rated |

### Movie features (2)
Computed from training ratings only.

| Feature | Description |
|---|---|
| `movie_mean` | Average rating received by this movie from all users |
| `movie_count` | Total number of users who have rated this movie |

### SVD latent features (50)
A `TruncatedSVD(n_components=50)` is applied to the mean-centred user-item matrix. This produces a 50-dimensional vector per user that encodes learned taste preferences — richer signal than simple averages.

| Feature | Description |
|---|---|
| `svd_0` … `svd_49` | 50 latent user dimensions from SVD decomposition |

### Genre features (20)
Each movie's pipe-separated genre string is split and one-hot encoded using `MultiLabelBinarizer`. This gives the model content-side signal independent of user history.

| Feature | Description |
|---|---|
| `genre_Action`, `genre_Drama`, … | Binary flag: 1 if movie belongs to this genre |

---

## Models

### 1. Ridge Regression

**Idea**: Fit a single linear equation through the feature space. The simplest and fastest model — good as a baseline.

**How it works**:
```
rating = w₁·user_mean + w₂·user_count + ... + w₇₄·genre_Western + b
```

**Regularisation**: L2 penalty (`alpha=1.0`) added to the loss:
```
Loss = MSE + alpha × Σ(wᵢ²)
```
Prevents any single weight from dominating. Especially useful when SVD features are correlated.

**Hyperparameters**:
| Parameter | Value |
|---|---|
| `alpha` | 1.0 |

**Saved**: `models/ridge.pkl`

---

### 2. Random Forest Regressor

**Idea**: Build many independent decision trees on random subsets of data and features, then average their predictions. Handles non-linear relationships that Ridge cannot.

**How it works**:
```
1. Sample with replacement from training data (bootstrap)
2. Build a decision tree using a random subset of features at each split
3. Repeat 200 times → 200 independent trees
4. Prediction = mean of all 200 tree outputs
```

Averaging cancels out individual tree errors — this is **variance reduction** (bagging).

**Hyperparameters**:
| Parameter | Value | Reason |
|---|---|---|
| `n_estimators` | 200 trees | More trees = more stable average |
| `max_depth` | 12 | Limits tree complexity to prevent overfitting |
| `min_samples_leaf` | 4 | Each leaf must cover at least 4 samples |

**Saved**: `models/rf.pkl`

---

### 3. XGBoost Regressor

**Idea**: Build trees *sequentially*, where each new tree corrects the errors made by all previous trees. More accurate than Random Forest but slower to train.

**How it works**:
```
Tree 1: predicts raw ratings → residuals computed
Tree 2: predicts the residuals from Tree 1 → new residuals
Tree 3: predicts residuals from Trees 1+2 → ...
...
Final prediction = sum of contributions from all 300 trees
```

**Regularisation**:
- `subsample=0.8` — each tree sees only 80% of training rows (like dropout)
- `colsample_bytree=0.8` — each tree uses only 80% of features
- `learning_rate=0.05` — shrinks each tree's contribution, requiring more trees but preventing overfitting

**Hyperparameters**:
| Parameter | Value |
|---|---|
| `n_estimators` | 300 trees |
| `max_depth` | 6 |
| `learning_rate` | 0.05 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |

**Saved**: `models/xgb.pkl`

---

## Model Comparison

| Model | Strengths | Weaknesses |
|---|---|---|
| Ridge | Fast, interpretable, stable | Cannot learn non-linear patterns |
| Random Forest | Handles non-linearity, robust to outliers | Slower, larger model file |
| XGBoost | Best accuracy, learns complex interactions | Slowest to train |

---

## Saved Artifacts (`models/`)

| File | Contents |
|---|---|
| `ridge.pkl` | Trained `Ridge` model |
| `rf.pkl` | Trained `RandomForestRegressor` |
| `xgb.pkl` | Trained `XGBRegressor` |
| `meta.pkl` | User/movie stat dicts, SVD matrix, genre df, feature column list, movies df |

---

## API

| Endpoint | Description |
|---|---|
| `GET /` | HTML UI |
| `GET /users` | List of valid user IDs |
| `GET /movies` | List of all movies with titles and genres |
| `GET /predict?user_id=42&movie_id=318&model=xgboost` | Predict rating |

**Response includes**:
- `predicted_rating` — model output clipped to [0.5, 5.0]
- `actual_rating` — real rating from the dataset if it exists, otherwise `null`
- `title`, `genres` — movie metadata

---

## Usage

```bash
# Step 1 — train once
python regression/train.py

# Step 2 — start API
uvicorn regression.main:app --reload --port 8001
```

Open `http://127.0.0.1:8001`. Enter a User ID and Movie ID, pick a model, and see the predicted rating alongside the actual rating (if it exists in the dataset).
