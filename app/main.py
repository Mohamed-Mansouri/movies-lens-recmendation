"""
app/main.py — Unified FastAPI app.
Mounts all 4 feature sub-apps under prefixed routes and serves
a central dashboard at /.

Run from project root:
    uvicorn app.main:app --reload --port 8000
"""
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from features.recommendation.main import app as rec_app
from features.regression.main     import app as reg_app
from features.classification.main import app as cls_app
from features.clustering.main     import app as clu_app

BASE_DIR  = Path(__file__).parent
HTML_FILE = BASE_DIR / "static" / "index.html"

app = FastAPI(title="MovieLens — Unified API")

# ── Feature sub-apps ───────────────────────────────────────────────────────────
app.mount("/recommendation", rec_app)
app.mount("/regression",     reg_app)
app.mount("/classification", cls_app)
app.mount("/clustering",     clu_app)

# ── Dashboard static assets ────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return HTML_FILE.read_text(encoding="utf-8")
