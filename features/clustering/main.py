"""
main.py — FastAPI app for user clustering.
Models are loaded lazily and reloaded automatically when pkl files change.
Run: uvicorn features.clustering.main:app --reload --port 8003
"""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from math import log

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR   = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR   = BASE_DIR.parent.parent / "dataset"
HTML_FILE  = BASE_DIR / "static" / "index.html"

state: dict = {"loaded_at": 0}


def _pkl_mtime():
    pkls = list(MODELS_DIR.glob("*.pkl"))
    return max((p.stat().st_mtime for p in pkls), default=0) if pkls else 0


def _load():
    pkl = MODELS_DIR / "clustering.pkl"
    if not pkl.exists():
        raise HTTPException(503, "Models not found. Run 'python features/clustering/train.py' first.")

    print("Loading clustering models...")
    data = joblib.load(pkl)

    kmeans          = data["kmeans"]
    dbscan          = data["dbscan"]
    scaler          = data["scaler"]
    user_kmeans     = data["user_kmeans"]
    user_dbscan     = data["user_dbscan"]
    kmeans_profiles = data["kmeans_profiles"]
    dbscan_profiles = data["dbscan_profiles"]
    all_genres      = data["all_genres"]
    user_ids        = data["user_ids"]
    genre_share_df  = data["genre_share_df"]
    U_sigma         = data["U_sigma"]
    user_index      = data["user_index"]
    svd_cols        = data["svd_cols"]
    global_mean     = data["global_mean"]
    user_stats      = pd.DataFrame(data["user_stats"])

    # Load ratings + movies for recommendations
    ratings = pd.read_csv(DATA_DIR / "ratings.csv")
    movies  = pd.read_csv(DATA_DIR / "movies.csv")[["movieId", "title", "genres"]]

    # Build cluster-member lookup: algo -> cluster_id -> set of userIds
    def _members(user_map):
        members = {}
        for uid, cid in user_map.items():
            members.setdefault(cid, set()).add(uid)
        return members

    kmeans_members = _members(user_kmeans)
    dbscan_members = _members(user_dbscan)

    # Pre-compute cluster genre averages for radar chart
    def _cluster_genre_avg(user_map, cid):
        uids = [u for u, c in user_map.items() if c == cid]
        subset = genre_share_df.loc[[u for u in uids if u in genre_share_df.index]]
        if subset.empty:
            return {}
        return subset.mean().to_dict()

    def _user_genre_profile(user_id):
        if user_id in genre_share_df.index:
            return genre_share_df.loc[user_id].to_dict()
        return {g: 0.0 for g in all_genres}

    def assign(algo, user_id):
        if algo == "kmeans":
            cluster_id = user_kmeans.get(user_id)
            profiles   = kmeans_profiles
        else:
            cluster_id = user_dbscan.get(user_id)
            profiles   = dbscan_profiles

        if cluster_id is None:
            u_mean    = float(user_stats["user_mean"].get(user_id, global_mean))
            u_count   = float(user_stats["user_count"].get(user_id, 0))
            u_std     = float(user_stats["user_std"].get(user_id, 0))
            u_svd     = U_sigma[user_index[user_id]] if user_id in user_index else np.zeros(len(svd_cols))
            genre_row = genre_share_df.loc[user_id].values if user_id in genre_share_df.index else np.zeros(len(all_genres))
            feat      = np.array([[u_mean, u_count, u_std, *genre_row, *u_svd]])
            feat_sc   = scaler.transform(feat)
            cluster_id = int(kmeans.predict(feat_sc)[0]) if algo == "kmeans" else -1

        cid     = int(cluster_id)
        profile = profiles.get(cid, {})

        # Genre profiles for radar
        user_gp    = _user_genre_profile(user_id)
        cluster_gp = _cluster_genre_avg(
            user_kmeans if algo == "kmeans" else user_dbscan, cid
        )

        # Pick top 8 genres by cluster average (most descriptive axes)
        top_axes = sorted(cluster_gp, key=cluster_gp.get, reverse=True)[:8]
        radar = {
            "axes":    top_axes,
            "user":    [round(user_gp.get(g, 0.0), 3)    for g in top_axes],
            "cluster": [round(cluster_gp.get(g, 0.0), 3) for g in top_axes],
        }

        all_profiles_list = sorted(
            [{"cluster_id": c, **p} for c, p in profiles.items()],
            key=lambda x: x["cluster_id"],
        )

        return {
            "cluster_id":   cid,
            "profile":      profile,
            "all_clusters": all_profiles_list,
            "radar":        radar,
        }

    def recommend(algo, user_id, n=10):
        if algo == "kmeans":
            cluster_id = user_kmeans.get(user_id, -1)
            members    = kmeans_members.get(cluster_id, set())
        else:
            cluster_id = user_dbscan.get(user_id, -1)
            members    = dbscan_members.get(cluster_id, set())

        if cluster_id == -1 or not members:
            return []

        # Movies this user has already rated
        seen = set(ratings[ratings["userId"] == user_id]["movieId"])

        # Ratings from cluster members, excluding movies the user already saw
        cluster_ratings = ratings[
            ratings["userId"].isin(members) & ~ratings["movieId"].isin(seen)
        ]

        if cluster_ratings.empty:
            return []

        # Score = avg_rating weighted by log(count+1) — confidence-adjusted
        agg = (
            cluster_ratings.groupby("movieId")["rating"]
            .agg(avg="mean", cnt="count")
            .reset_index()
        )
        agg = agg[agg["cnt"] >= 2]   # at least 2 cluster members rated it
        agg["score"] = agg["avg"] * agg["cnt"].apply(lambda c: log(c + 1))
        agg = agg.sort_values("score", ascending=False).head(n)

        result = agg.merge(movies, on="movieId", how="left")
        result["avg"] = result["avg"].round(2)
        return result[["movieId", "title", "genres", "avg", "cnt"]].to_dict(orient="records")

    state.update(
        assign=assign,
        recommend=recommend,
        valid_users=sorted(user_ids),
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
    return {"users": state.get("valid_users", [])}


@app.get("/cluster")
def cluster(
    user_id: int = Query(..., description="User ID"),
    model:   str = Query("kmeans", description="kmeans | dbscan"),
):
    if model not in ("kmeans", "dbscan"):
        raise HTTPException(400, "model must be 'kmeans' or 'dbscan'")
    _ensure_fresh()
    result = state["assign"](model, user_id)
    return {"user_id": user_id, "model": model, **result}


@app.get("/cluster_recommendations")
def cluster_recommendations(
    user_id: int = Query(..., description="User ID"),
    model:   str = Query("kmeans", description="kmeans | dbscan"),
    n:       int = Query(8, description="Number of recommendations"),
):
    if model not in ("kmeans", "dbscan"):
        raise HTTPException(400, "model must be 'kmeans' or 'dbscan'")
    _ensure_fresh()
    recs = state["recommend"](model, user_id, n)
    return {"user_id": user_id, "model": model, "recommendations": recs}
