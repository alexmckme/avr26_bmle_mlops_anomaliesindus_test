"""API FastAPI — Anomalies Indus (Phase 1).

Endpoints :
    GET  /            -> infos de l'API
    POST /training    -> entraîne un modèle PaDiM pour une catégorie (depuis MinIO)
    POST /predict     -> score d'anomalie d'une image avec le modèle d'une catégorie

Tout le travail (lecture MinIO, fit PaDiM, évaluation, prédiction) est délégué
à `core.*` : l'API est une fine couche HTTP par-dessus le même code que les
scripts (anti train/serve skew).

Lancement (dev) :
    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional, Union

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

import core.padim as padim
from core.config import load_settings
from core.storage import get_client

settings = load_settings()
client = get_client(settings)                 # client MinIO partagé
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"

app = FastAPI(title="Anomalies Indus API", version="0.1.0")


class TrainingRequest(BaseModel):
    category: str
    data_version: Optional[Union[int, str]] = None   # 0..4 ou 'full' (grille)
    fraction: Optional[float] = Field(default=None, gt=0.0, le=1.0)
    img_size: int = Field(default=padim.IMG_SIZE)
    eval: bool = False                               # AUC + seuil sur le split test


def _category_exists(category: str) -> bool:
    prefix = f"raw/{category}/train/good/"
    return bool(padim.list_minio_entries(client, settings.minio_bucket, prefix))


def _model_path_for(category: str) -> Path:
    """Modèle à utiliser pour /predict : `full` si dispo, sinon la version la plus récente."""
    try:
        return padim.resolve_model_path(category, MODEL_DIR)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/")
def root() -> dict:
    return {"app": "anomalies-indus", "version": app.version,
            "endpoints": ["/training", "/predict"]}


@app.post("/training")
def training(req: TrainingRequest) -> dict:
    category = req.category.strip()
    if not _category_exists(category):
        raise HTTPException(
            status_code=404,
            detail=f"Catégorie '{category}' introuvable dans le bucket "
                   f"(s3://{settings.minio_bucket}/raw/{category}/train/good/).",
        )

    t0 = time.time()
    model, meta = padim.fit_from_minio(
        client, settings.minio_bucket, category,
        img_size=req.img_size,
        data_version=req.data_version,
        fraction=req.fraction,
    )

    if req.eval:
        meta.update(padim.evaluate_on_test(client, settings.minio_bucket, model,
                                           category, req.img_size))

    # Nom d'artefact porteur de la version (le « full » garde le nom simple)
    if meta.get("full", False):
        out_name = f"{category}.npz"
    elif meta.get("data_version") is not None:
        out_name = f"{category}.v{meta['data_version']}.npz"
    else:
        out_name = f"{category}.f{int(round(meta['fraction'] * 100))}.npz"
    artifact = padim.save_model(model, MODEL_DIR / out_name)

    meta.update({
        "artifact": artifact,
        "elapsed_s": round(time.time() - t0, 1),
    })

    # Suivi MLflow (même helper que les scripts)
    from core import tracking

    if tracking.setup_mlflow():
        run_id = tracking.log_training(meta, artifact, run_name=category)
        if run_id:
            meta["mlflow_run_id"] = run_id
    return meta


@app.post("/predict")
async def predict(category: str = Form(...), file: UploadFile = File(...)) -> dict:
    category = category.strip()
    model_path = _model_path_for(category)
    model = padim.load_model(model_path)

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Fichier image vide.")
    try:
        score, _ = padim.predict_image(data, model)
    except Exception as exc:  # noqa: BLE001 — image invalide
        raise HTTPException(status_code=422, detail=f"Image illisible : {exc}") from exc

    result = {
        "category": category,
        "model": model_path.name,
        "score": round(float(score), 2),
        "threshold": model.threshold,
        "anomaly": bool(score > model.threshold) if model.threshold is not None else None,
    }
    return result
