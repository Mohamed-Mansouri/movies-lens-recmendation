# Genre Classification Feature

Given a `user_id`, predict which movie genres that user is interested in.

> **Why multi-label?** A user can like multiple genres simultaneously (e.g. Action AND Comedy AND Drama). This is not a single-class problem — every genre gets its own binary prediction.

> **Difference from Regression**: Regression predicts a continuous value (a rating). Classification predicts discrete labels (interested / not interested per genre).

---

## Dataset

Source: `../dataset/`

| File | Used for |
|---|---|
| `ratings.csv` | User rating history — used to build labels and features |
| `movies.csv` | Genre extraction per movie |

---

## Label Construction (Y)

The target is a binary matrix of shape `(n_users × n_genres)`.

**A user is labelled as interested in a genre if EITHER signal is true:**

### Signal 1 — Engagement (volume)
```
genre_count[user, genre] / total_ratings[user] >= 0.10
```
If a user watches ≥10% of all their movies in a given genre, they are clearly interested — regardless of the rating they give. A user who watches 50 Action movies at 3★ is more interested in Action than someone who watched 1 Comedy at 5★.

### Signal 2 — Quality (above personal mean)
```
genre_avg[user, genre] > user_overall_mean  AND  genre_count[user, genre] >= 2
```
If a user consistently rates a genre **above their own personal average**, that's a preference signal — even if they don't watch it often. The `>= 2` minimum avoids noise from a single lucky rating.

```
Label = engagement OR quality
```

This two-signal approach captures both heavy watchers (volume) and selective raters (quality).

---

## Feature Engineering (X)

One row per user — 53 features total:

### User statistics (3)
Computed from the full ratings dataset.

| Feature | Description |
|---|---|
| `user_mean` | Overall average rating this user gives |
| `user_count` | Total number of movies they have rated |
| `user_std` | Standard deviation of their ratings — measures how discriminating they are |

### SVD latent features (50)
A `TruncatedSVD(n_components=50)` is applied to the mean-centred user-item matrix. Each user gets a 50-dimensional vector that encodes their latent taste profile — which "taste clusters" they belong to — without explicitly encoding per-genre preferences.

| Feature | Description |
|---|---|
| `svd_0` … `svd_49` | 50 latent dimensions from decomposition of the user-item matrix |

> **Why SVD features and not genre averages?** Using the user's own genre averages as input features to predict genre labels would be direct data leakage. SVD captures global collaborative patterns instead.

---

## Models

All three models use the **One-vs-Rest** strategy via `MultiOutputClassifier` — one independent binary classifier is trained per genre (19 classifiers total).

### 1. Logistic Regression

**Idea**: Fit a linear decision boundary per genre. The simplest and fastest model.

**How it works per genre**:
```
P(interested in genre G) = sigmoid(w · features + b)
label = 1 if P >= 0.5
```

L2 regularisation (`C=1.0`) prevents overfitting.

**Hyperparameters**:
| Parameter | Value |
|---|---|
| `C` | 1.0 (inverse regularisation strength) |
| `max_iter` | 500 |

**Saved**: `models/lr.pkl`

---

### 2. Random Forest Classifier

**Idea**: 200 decision trees per genre, each trained on a random bootstrap sample. Majority vote determines the label.

**How it works**:
```
1. Bootstrap sample training users
2. Build tree using random feature subsets at each split
3. Repeat 200 times
4. Label = majority vote of all 200 trees
5. P(interested) = fraction of trees voting "yes"
```

Handles non-linear interactions between features (e.g. "users with high user_std AND specific SVD dims prefer Thriller").

**Hyperparameters**:
| Parameter | Value |
|---|---|
| `n_estimators` | 200 |
| `max_depth` | 12 |
| `min_samples_leaf` | 4 |

**Saved**: `models/rf.pkl`

---

### 3. XGBoost Classifier

**Idea**: Sequential boosting — each new tree corrects the errors of all previous trees. Uses binary cross-entropy loss (`logloss`) since the target is binary per genre.

**How it works**:
```
Tree 1: predict genre labels → compute logloss residuals
Tree 2: predict residuals → new residuals
...
Final P(interested) = sigmoid(sum of all 200 tree contributions)
```

Row and column subsampling regularise the model similarly to dropout.

**Hyperparameters**:
| Parameter | Value |
|---|---|
| `n_estimators` | 200 |
| `max_depth` | 5 |
| `learning_rate` | 0.05 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `eval_metric` | logloss |

**Saved**: `models/xgb.pkl`

---

## Evaluation Metrics

Since this is multi-label classification, standard accuracy is not enough.

| Metric | Description |
|---|---|
| **F1-micro** | F1 computed globally across all genre-user pairs — favours common genres |
| **F1-macro** | Average F1 per genre — treats all genres equally regardless of frequency |
| **Hamming Loss** | Fraction of label-user pairs predicted incorrectly — lower is better |

---

## Saved Artifacts (`models/`)

| File | Contents |
|---|---|
| `lr.pkl` | 19 trained `LogisticRegression` estimators (one per genre) |
| `rf.pkl` | 19 trained `RandomForestClassifier` estimators |
| `xgb.pkl` | 19 trained `XGBClassifier` estimators |
| `meta.pkl` | Genre list, feature columns, user index, `U_sigma`, user stat dicts |

---

## API

| Endpoint | Description |
|---|---|
| `GET /` | HTML UI |
| `GET /users` | List of valid user IDs |
| `GET /classify?user_id=42&model=xgboost` | Predict genre interests |

**Response per genre**:
```json
{
  "genre": "Action",
  "interested": true,
  "confidence": 0.821
}
```

Results sorted: interested genres first, then by confidence descending.

---

## Usage

```bash
# Step 1 — train once
python classification/train.py

# Step 2 — start API
uvicorn classification.main:app --reload --port 8002
```

Open `http://127.0.0.1:8002`. Enter a User ID, pick a model, and see the predicted genre profile with confidence bars.
