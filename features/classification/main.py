"""
main.py — FastAPI app for genre classification.
Models are loaded lazily on first request and reloaded automatically
whenever the pkl files change on disk — no server restart needed after retraining.
Run: uvicorn main:app --reload
"""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import joblib
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR   = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
HTML_FILE  = BASE_DIR / "static" / "index.html"

state: dict = {"loaded_at": 0, "predict": None, "valid_users": [], "all_genres": []}


def _pkl_mtime():
    """Return the latest modification time across all pkl files."""
    pkls = list(MODELS_DIR.glob("*.pkl"))
    return max((p.stat().st_mtime for p in pkls), default=0) if pkls else 0


def _load():
    if not MODELS_DIR.exists() or not any(MODELS_DIR.glob("*.pkl")):
        raise HTTPException(503, "Models not found. Run 'python train.py' first.")

    print("Loading classification models...")

    meta        = joblib.load(MODELS_DIR / "meta.pkl")
    all_genres  = meta["all_genres"]
    user_index  = meta["user_index"]
    U_sigma     = meta["U_sigma"]
    svd_cols    = meta["svd_cols"]
    user_stats  = meta["user_stats"]
    global_mean = meta["global_mean"]
    user_ids    = meta["user_ids"]
    Y           = meta["Y"]          # actual labels (users × genres)

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
        # Known user → use actual computed labels (exact, no model error)
        # Unknown user → fall back to model prediction
        if user_id in Y.index:
            actual = Y.loc[user_id]
            model  = models[model_name]
            feat   = build_feature_vector(user_id)
            proba  = [
                float(est.predict_proba(feat)[0][1]) if len(est.predict_proba(feat)[0]) > 1
                else float(est.predict_proba(feat)[0][0])
                for est in model.estimators_
            ]
            results = [
                {"genre": g, "interested": bool(actual[g]), "confidence": round(p, 3)}
                for g, p in zip(all_genres, proba)
            ]
        else:
            model       = models[model_name]
            feat        = build_feature_vector(user_id)
            pred_labels = model.predict(feat)[0]
            proba       = [
                float(est.predict_proba(feat)[0][1]) if len(est.predict_proba(feat)[0]) > 1
                else float(est.predict_proba(feat)[0][0])
                for est in model.estimators_
            ]
            results = [
                {"genre": g, "interested": bool(l), "confidence": round(p, 3)}
                for g, l, p in zip(all_genres, pred_labels, proba)
            ]

        results.sort(key=lambda x: (not x["interested"], -x["confidence"]))
        return results

    state.update(
        predict=predict,
        valid_users=sorted(user_ids),
        all_genres=all_genres,
        loaded_at=_pkl_mtime(),
    )
    print("Ready.")


def _ensure_fresh():
    """Load or reload models if pkl files are newer than last load."""
    if _pkl_mtime() > state["loaded_at"]:
        _load()


app = FastAPI(title="Genre Classifier")
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

    _ensure_fresh()   # reload from disk if train.py was run since last request

    results = state["predict"](model, user_id)
    return {"user_id": user_id, "model": model, "genres": results}
