"""
main.py — FastAPI app for genre classification.
Run: uvicorn classification.main:app --reload --port 8002
Requires: python classification/train.py to have been run first.
"""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from contextlib import asynccontextmanager

import numpy as np
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
        raise RuntimeError("No trained models found. Run 'python classification/train.py' first.")

    print("Loading classification models...")

    meta        = joblib.load(MODELS_DIR / "meta.pkl")
    all_genres  = meta["all_genres"]
    user_index  = meta["user_index"]
    U_sigma     = meta["U_sigma"]
    svd_cols    = meta["svd_cols"]
    user_stats  = meta["user_stats"]        # {"user_mean": {uid: v}, ...}
    global_mean = meta["global_mean"]
    user_ids    = meta["user_ids"]

    lr  = joblib.load(MODELS_DIR / "lr.pkl")
    rf  = joblib.load(MODELS_DIR / "rf.pkl")
    xgb = joblib.load(MODELS_DIR / "xgb.pkl")

    models = {"logistic_regression": lr, "random_forest": rf, "xgboost": xgb}

    def build_feature_vector(user_id):
        u_mean  = user_stats["user_mean"].get(user_id,  global_mean)
        u_count = user_stats["user_count"].get(user_id, 0)
        u_std   = user_stats["user_std"].get(user_id,   0)
        u_svd   = U_sigma[user_index[user_id]] if user_id in user_index else np.zeros(len(svd_cols))
        return np.array([[u_mean, u_count, u_std, *u_svd]])

    def predict(model_name, user_id):
        model = models[model_name]
        feat  = build_feature_vector(user_id)

        # Binary predictions
        pred_labels = model.predict(feat)[0]

        # Probabilities from each per-genre estimator
        proba = []
        for est in model.estimators_:
            p = est.predict_proba(feat)[0]
            # predict_proba returns [P(0), P(1)] — take P(1)
            proba.append(float(p[1]) if len(p) > 1 else float(p[0]))

        results = []
        for genre, label, prob in zip(all_genres, pred_labels, proba):
            results.append({
                "genre":       genre,
                "interested":  bool(label),
                "confidence":  round(prob, 3),
            })

        # Sort: interested first, then by confidence desc
        results.sort(key=lambda x: (not x["interested"], -x["confidence"]))
        return results

    state.update(
        predict=predict,
        valid_users=sorted(user_ids),
        all_genres=all_genres,
    )
    print("Ready.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield

app = FastAPI(title="Genre Classifier", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_FILE.read_text(encoding="utf-8")


@app.get("/users")
def get_users():
    return {"users": state["valid_users"]}


@app.get("/classify")
def classify(
    user_id: int = Query(..., description="User ID"),
    model:   str = Query("xgboost", description="logistic_regression | random_forest | xgboost"),
):
    valid_models = ["logistic_regression", "random_forest", "xgboost"]
    if model not in valid_models:
        raise HTTPException(400, f"Unknown model '{model}'. Choose from: {valid_models}")

    results = state["predict"](model, user_id)

    return {
        "user_id": user_id,
        "model":   model,
        "genres":  results,
    }
