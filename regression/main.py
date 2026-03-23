"""
main.py — FastAPI app for rating regression.
Run: uvicorn regression.main:app --reload
Requires: python regression/train.py to have been run first.
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
        raise RuntimeError("No trained models found. Run 'python regression/train.py' first.")

    print("Loading regression models...")

    meta = joblib.load(MODELS_DIR / "meta.pkl")
    global_mean      = meta["global_mean"]
    user_mean_dict   = meta["user_mean_dict"]
    user_count_dict  = meta["user_count_dict"]
    movie_mean_dict  = meta["movie_mean_dict"]
    movie_count_dict = meta["movie_count_dict"]
    user_index       = meta["user_index"]
    U_sigma          = meta["U_sigma"]
    svd_cols         = meta["svd_cols"]
    genre_df         = meta["genre_df"].set_index("movieId")
    genre_cols       = meta["genre_cols"]
    movies           = meta["movies"]
    ratings          = meta["ratings"]

    zero_svd   = np.zeros(len(svd_cols))
    zero_genre = np.zeros(len(genre_cols))

    def build_feature_vector(user_id, movie_id):
        u_svd   = U_sigma[user_index[user_id]] if user_id in user_index else zero_svd
        g_vec   = genre_df.loc[movie_id, genre_cols].values.astype(float) \
                  if movie_id in genre_df.index else zero_genre
        return np.array([[
            user_mean_dict.get(user_id,   global_mean),
            user_count_dict.get(user_id,  0),
            movie_mean_dict.get(movie_id,  global_mean),
            movie_count_dict.get(movie_id, 0),
            *u_svd,
            *g_vec,
        ]])

    ridge = joblib.load(MODELS_DIR / "ridge.pkl")
    rf    = joblib.load(MODELS_DIR / "rf.pkl")
    xgb   = joblib.load(MODELS_DIR / "xgb.pkl")

    def predict(model_name, user_id, movie_id):
        feat = build_feature_vector(user_id, movie_id)
        if model_name == "ridge":
            raw = ridge.predict(feat)[0]
        elif model_name == "random_forest":
            raw = rf.predict(feat)[0]
        else:
            raw = xgb.predict(feat)[0]
        return float(np.clip(raw, 0.5, 5.0))

    state.update(
        predict=predict,
        movies=movies,
        ratings=ratings,
        valid_users=sorted(ratings["userId"].unique().tolist()),
        valid_movies=sorted(ratings["movieId"].unique().tolist()),
    )
    print("Ready.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield

app = FastAPI(title="Movie Rating Regression", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_FILE.read_text(encoding="utf-8")


@app.get("/users")
def get_users():
    return {"users": state["valid_users"]}


@app.get("/movies")
def get_movies():
    return state["movies"].to_dict(orient="records")


@app.get("/predict")
def predict_rating(
    user_id:  int = Query(..., description="User ID"),
    movie_id: int = Query(..., description="Movie ID"),
    model:    str = Query("xgboost", description="ridge | random_forest | xgboost"),
):
    valid_models = ["ridge", "random_forest", "xgboost"]
    if model not in valid_models:
        raise HTTPException(400, f"Unknown model '{model}'. Choose from: {valid_models}")

    predicted = state["predict"](model, user_id, movie_id)

    # Look up actual rating if it exists in the dataset
    row = state["ratings"][
        (state["ratings"]["userId"] == user_id) &
        (state["ratings"]["movieId"] == movie_id)
    ]
    actual = float(row["rating"].values[0]) if not row.empty else None

    # Movie info
    movie_row = state["movies"][state["movies"]["movieId"] == movie_id]
    title  = movie_row["title"].values[0]  if not movie_row.empty else f"Movie {movie_id}"
    genres = movie_row["genres"].values[0] if not movie_row.empty else ""

    return {
        "user_id":          user_id,
        "movie_id":         movie_id,
        "title":            title,
        "genres":           genres,
        "model":            model,
        "predicted_rating": round(predicted, 2),
        "actual_rating":    actual,
    }
