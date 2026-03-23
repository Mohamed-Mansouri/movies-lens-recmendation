"""
main.py — FastAPI app for user clustering.
Models are loaded lazily and reloaded automatically when pkl files change.
Run: uvicorn clustering.main:app --reload --port 8003
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

state: dict = {"loaded_at": 0, "assign": None, "valid_users": [], "profiles": {}}


def _pkl_mtime():
    pkls = list(MODELS_DIR.glob("*.pkl"))
    return max((p.stat().st_mtime for p in pkls), default=0) if pkls else 0


def _load():
    pkl = MODELS_DIR / "clustering.pkl"
    if not pkl.exists():
        raise HTTPException(503, "Models not found. Run 'python clustering/train.py' first.")

    print("Loading clustering models...")
    data = joblib.load(pkl)

    kmeans          = data["kmeans"]
    dbscan          = data["dbscan"]
    scaler          = data["scaler"]
    user_kmeans     = data["user_kmeans"]      # uid -> cluster_id
    user_dbscan     = data["user_dbscan"]
    kmeans_profiles = data["kmeans_profiles"]
    dbscan_profiles = data["dbscan_profiles"]
    feat_cols       = data["feat_cols"]

    # Reconstruct feature matrix for unknown user assignment
    import pandas as pd
    user_stats     = pd.DataFrame(data["user_stats"])
    genre_share_df = data["genre_share_df"]
    svd_cols       = data["svd_cols"]
    all_genres     = data["all_genres"]
    user_ids       = data["user_ids"]

    U_sigma    = data["U_sigma"]
    user_index = data["user_index"]
    global_mean = data["global_mean"]

    svd_df = pd.DataFrame(
        U_sigma,
        index=list(user_index.keys()),
        columns=svd_cols,
    )

    def assign(algo, user_id):
        # Known user → stored assignment
        if algo == "kmeans":
            cluster_id = user_kmeans.get(user_id)
            profiles   = kmeans_profiles
        else:
            cluster_id = user_dbscan.get(user_id)
            profiles   = dbscan_profiles

        if cluster_id is None:
            # Unknown user: build feature vector and predict
            u_mean  = float(user_stats["user_mean"].get(user_id, global_mean))
            u_count = float(user_stats["user_count"].get(user_id, 0))
            u_std   = float(user_stats["user_std"].get(user_id, 0))
            u_svd   = U_sigma[user_index[user_id]] if user_id in user_index else np.zeros(len(svd_cols))
            genre_row = genre_share_df.loc[user_id].values if user_id in genre_share_df.index else np.zeros(len(all_genres))
            feat = np.array([[u_mean, u_count, u_std, *genre_row, *u_svd]])
            feat_scaled = scaler.transform(feat)
            if algo == "kmeans":
                cluster_id = int(kmeans.predict(feat_scaled)[0])
            else:
                cluster_id = -1   # DBSCAN can't predict new points — treat as noise

        profile = profiles.get(int(cluster_id), {})
        all_profiles_list = [
            {"cluster_id": cid, **p}
            for cid, p in profiles.items()
        ]
        all_profiles_list.sort(key=lambda x: x["cluster_id"])

        return {
            "cluster_id":    int(cluster_id),
            "profile":       profile,
            "all_clusters":  all_profiles_list,
        }

    state.update(
        assign=assign,
        valid_users=sorted(user_ids),
        profiles={"kmeans": kmeans_profiles, "dbscan": dbscan_profiles},
        loaded_at=_pkl_mtime(),
    )
    print("Ready.")


def _ensure_fresh():
    if _pkl_mtime() > state["loaded_at"]:
        _load()


app = FastAPI(title="User Clustering")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_FILE.read_text(encoding="utf-8")


@app.get("/users")
def get_users():
    return {"users": state["valid_users"]}


@app.get("/cluster")
def cluster(
    user_id: int = Query(..., description="User ID"),
    model:   str = Query("kmeans", description="kmeans | dbscan"),
):
    valid_models = ["kmeans", "dbscan"]
    if model not in valid_models:
        raise HTTPException(400, f"Unknown model '{model}'. Choose from: {valid_models}")

    _ensure_fresh()

    result = state["assign"](model, user_id)
    return {"user_id": user_id, "model": model, **result}
