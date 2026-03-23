"""
main.py — FastAPI app. Loads pre-trained artifacts and serves the UI.
Run: uvicorn main:app --reload
Requires: python train.py to have been run first.
"""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR   = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
HTML_FILE  = BASE_DIR / "static" / "index.html"

state: dict = {}


def load_models():
    if not MODELS_DIR.exists() or not any(MODELS_DIR.iterdir()):
        raise RuntimeError("No trained models found. Run 'python train.py' first.")

    print("Loading artifacts...")

    # -- Shared metadata --
    meta        = joblib.load(MODELS_DIR / "meta.pkl")
    movies      = meta["movies"]
    train_df    = meta["train_df"]
    user_means  = meta["user_means"]          # dict  uid -> mean
    global_mean = meta["global_mean"]         # float
    user_index  = meta["user_index"]          # dict  uid -> row int
    movie_index = meta["movie_index"]         # dict  mid -> col int
    all_movies  = meta["matrix_columns"]      # list of movieIds

    zero_svd = np.zeros(100)  # fallback for unknown users

    # ---- SVD ----
    svd_data  = joblib.load(MODELS_DIR / "svd.pkl")
    U_sigma   = svd_data["U_sigma"]                    # (n_users, 100)
    Vt        = svd_data["Vt"]                         # (100, n_movies)
    row_means = svd_data["row_means"]                  # (n_users,)
    # Reconstruct full matrix once as numpy — fast int indexing at prediction time
    svd_rec   = (U_sigma @ Vt) + row_means.reshape(-1, 1)   # (n_users, n_movies)

    def predict_svd(uid, mid):
        if uid in user_index and mid in movie_index:
            return float(np.clip(svd_rec[user_index[uid], movie_index[mid]], 0.5, 5.0))
        return float(user_means.get(uid, global_mean))

    # ---- KNN ----
    knn_data      = joblib.load(MODELS_DIR / "knn.pkl")
    matrix_vals   = knn_data["matrix_vals"]
    knn_row_means = knn_data["row_means"]
    knn_distances = knn_data["knn_distances"]
    knn_indices   = knn_data["knn_indices"]

    def predict_knn(uid, mid):
        if uid not in user_index or mid not in movie_index:
            return float(user_means.get(uid, global_mean))
        u_idx     = user_index[uid]
        m_idx     = movie_index[mid]
        n_idxs    = knn_indices[u_idx, 1:]
        n_sims    = 1.0 - knn_distances[u_idx, 1:]
        n_ratings = matrix_vals[n_idxs, m_idx]
        mask      = n_ratings > 0
        if not mask.any():
            return float(knn_row_means[u_idx])
        sims   = n_sims[mask]
        rtings = n_ratings[mask]
        n_mu   = knn_row_means[n_idxs[mask]]
        denom  = np.abs(sims).sum()
        if denom == 0:
            return float(knn_row_means[u_idx])
        return float(np.clip(knn_row_means[u_idx] + (sims * (rtings - n_mu)).sum() / denom, 0.5, 5.0))

    # ---- XGBoost — O(1) dict lookups, no DataFrame merges ----
    xgb_data         = joblib.load(MODELS_DIR / "xgb.pkl")
    xgb              = xgb_data["model"]
    user_mean_dict   = xgb_data["user_mean_dict"]
    user_count_dict  = xgb_data["user_count_dict"]
    movie_mean_dict  = xgb_data["movie_mean_dict"]
    movie_count_dict = xgb_data["movie_count_dict"]
    xgb_U_sigma      = xgb_data["U_sigma"]

    def predict_xgb(uid, mid):
        u_vec = xgb_U_sigma[user_index[uid]] if uid in user_index else zero_svd
        feat  = np.array([[
            user_mean_dict.get(uid,  global_mean),
            user_count_dict.get(uid, 0),
            movie_mean_dict.get(mid,  global_mean),
            movie_count_dict.get(mid, 0),
            *u_vec,
        ]])
        return float(np.clip(xgb.predict(feat)[0], 0.5, 5.0))

    state.update(
        movies=movies,
        train_df=train_df,
        all_movies=all_movies,
        valid_users=sorted(train_df["userId"].unique().tolist()),
        predictors={
            "svd":     predict_svd,
            "knn":     predict_knn,
            "xgboost": predict_xgb,
        },
    )
    print("Ready.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield

app = FastAPI(title="Movie Recommender", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_FILE.read_text(encoding="utf-8")


@app.get("/users")
def get_users():
    return {"users": state["valid_users"]}


@app.get("/recommend")
def recommend(
    user_id: int = Query(..., description="User ID"),
    model:   str = Query("svd", description="svd | nmf | knn | xgboost"),
    n:       int = Query(10,    description="Number of recommendations"),
):
    predictors = state["predictors"]
    if model not in predictors:
        raise HTTPException(400, f"Unknown model '{model}'. Choose from: {list(predictors)}")

    predict       = predictors[model]
    already_rated = set(state["train_df"][state["train_df"]["userId"] == user_id]["movieId"])
    candidates    = [m for m in state["all_movies"] if m not in already_rated]

    scores = [(mid, predict(user_id, mid)) for mid in candidates]
    scores.sort(key=lambda x: x[1], reverse=True)

    recs = pd.DataFrame(scores[:n], columns=["movieId", "predicted_rating"])
    recs = recs.merge(state["movies"][["movieId", "title", "genres"]], on="movieId", how="left")
    recs["predicted_rating"] = recs["predicted_rating"].round(2)

    return {
        "user_id": user_id,
        "model":   model,
        "recommendations": recs.to_dict(orient="records"),
    }
