"""API FastAPI — Anomalies Industrielles.

Endpoints :
    GET  /            -> infos de l'API
    GET  /metrics     -> métriques Prometheus (profil `monitoring`)
    POST /training    -> entraîne un modèle PaDiM pour une catégorie (depuis MinIO)
    POST /predict     -> score d'anomalie d'une image avec le modèle d'une catégorie

Tout le travail (lecture MinIO, fit PaDiM, évaluation, prédiction) est délégué
à `core.*` : l'API est une fine couche HTTP par-dessus le même code que les
scripts (anti train/serve skew).

Lancement (dev) :
    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import base64
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Union

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

import core.metrics as metrics
import core.padim as padim
import core.versioning as versioning
from core.config import load_settings
from core.storage import get_client

settings = load_settings()
client = get_client(settings)                 # client MinIO partagé
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Au démarrage : expose l'AUC du champion courant en métrique.

    Sans ça, la jauge `anomalies_champion_auc` ne serait alimentée qu'à la
    première promotion et le tableau de bord Grafana resterait vide.
    """
    from core import tracking

    if tracking.setup_mlflow():
        n = metrics.refresh_champion_gauges()
        print(f"[metrics] AUC du champion exposée pour {n} catégorie(s)")
    yield


app = FastAPI(title="Anomalies Industrielles API", version="0.1.0", lifespan=lifespan)

# `/metrics` : séries temporelles que Prometheus vient chercher (modèle « pull »).
# L'instrumentation couvre le HTTP (http_requests_total, http_request_duration_…).
try:
    from prometheus_fastapi_instrumentator import Instrumentator

    Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)
except ImportError:  # monitoring non installé : l'API reste pleinement fonctionnelle
    print("[metrics] prometheus-fastapi-instrumentator absent : /metrics désactivé")


class TrainingRequest(BaseModel):
    category: str
    data_version: Optional[Union[int, str]] = None   # index 0..9 ou 'full' (voir /data-versions)
    fraction: Optional[float] = Field(default=None, gt=0.0, le=1.0)
    img_size: int = Field(default=padim.IMG_SIZE)
    eval: bool = False                               # AUC + seuil sur le split test
    register: bool = True                            # version au Registry (artefact ~260 Mo)
    promote: bool = True                             # promouvoir en 'champion' si meilleur


def _category_exists(category: str) -> bool:
    prefix = f"raw/{category}/train/good/"
    return bool(padim.list_minio_entries(client, settings.minio_bucket, prefix))


def _model_path_for(category: str) -> tuple[Path, str]:
    """Modèle à servir : champion du Registry si dispo, sinon fichiers locaux."""
    from core import tracking

    try:
        return tracking.resolve_inference_model(category, MODEL_DIR)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/")
def root() -> dict:
    return {"app": "anomalies-industrielles", "version": app.version,
            "endpoints": ["/training", "/predict", "/models", "/runs",
                          "/data-versions", "/metrics"]}


@app.get("/models")
def models() -> dict:
    """Catégories connues + qualité et provenance du modèle servi (lecture Registry)."""
    from core import storage, tracking

    tracking.setup_mlflow()
    return {
        "categories": storage.list_categories(client, settings.minio_bucket),
        "models": tracking.list_champions(),
    }


@app.get("/runs")
def runs(limit: int = 20, category: Optional[str] = None) -> dict:
    """Derniers runs d'entraînement (métriques + traçabilité code/données)."""
    from core import tracking

    tracking.setup_mlflow()
    return {"runs": tracking.recent_runs(limit=limit, category=category)}


@app.get("/data-versions")
def data_versions() -> dict:
    """Grille des versions de données : `v0` = 10 %, `v1` = 20 %… `v9` = 100 %.

    `data_version` accepte l'index renvoyé ici (0..9) ou la chaîne 'full' ; les
    sous-ensembles sont imbriqués (v0 ⊂ v1 ⊂ …) : c'est une simulation de
    l'accumulation de données dans le temps.
    """
    versions = padim.data_versions()
    return {
        "versions": versions,
        "fractions": [v["fraction"] for v in versions],
        "note": "data_version = index de la grille, ou 'full' pour 100 %",
    }


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
    try:
        model, meta = padim.fit_from_minio(
            client, settings.minio_bucket, category,
            img_size=req.img_size,
            data_version=req.data_version,
            fraction=req.fraction,
        )
    except Exception:  # échec d'entraînement : compté, puis remonté en erreur 500
        metrics.observe_training(category, "error", time.time() - t0)
        raise

    if req.eval:
        meta.update(padim.evaluate_on_test(client, settings.minio_bucket, model,
                                           category, req.img_size))

    # Nom d'artefact explicite : `bottle.p40.npz` (40 %) ou `bottle.npz` (100 %)
    artifact = padim.save_model(model, MODEL_DIR / padim.artifact_name(category, meta))

    meta.update({
        "artifact": artifact,
        "elapsed_s": round(time.time() - t0, 1),
    })

    # Traçabilité : commit du code + manifeste des images réellement utilisées
    versioning.add_code_info(meta)
    manifest = versioning.minio_manifest(client, settings.minio_bucket, category, meta)

    # Suivi MLflow (même helper que les scripts)
    from core import tracking

    if tracking.setup_mlflow():
        info = tracking.log_training(
            meta, artifact, run_name=category,
            register_model=f"padim-{category}" if req.register else None,
            manifest=manifest,
        )
        if info:
            meta["mlflow_run_id"] = info["run_id"]
            if info.get("model_version") is not None:
                meta["mlflow_model_version"] = info["model_version"]
            # La promotion ne concerne que la version qu'on vient d'enregistrer.
            if req.register and req.promote:
                promo = tracking.promote_if_better(f"padim-{category}")
                meta["mlflow_promotion"] = promo
                # Tracé sur le run lui-même : sans ça, la promotion n'existerait
                # que dans cette réponse HTTP (run déjà fermé à ce stade).
                tracking.log_promotion(info["run_id"], promo, meta=meta)
                metrics.observe_promotion(category, bool(promo.get("promoted")),
                                          auc=meta.get("auc"))

    metrics.observe_training(category, "success", time.time() - t0)
    return meta


@app.post("/predict")
async def predict(category: str = Form(...), file: UploadFile = File(...),
                  heatmap: bool = Form(False)) -> dict:
    """Score d'anomalie d'une image. `heatmap=true` renvoie aussi la carte
    d'anomalie superposée (PNG en base64)."""
    category = category.strip()
    model_path, model_source = _model_path_for(category)
    model = padim.load_model(model_path)

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Fichier image vide.")
    try:
        score, anomaly_map = padim.predict_image(data, model)
    except Exception as exc:  # noqa: BLE001 — image invalide
        raise HTTPException(status_code=422, detail=f"Image illisible : {exc}") from exc

    result = {
        "category": category,
        "model": model_path.name,
        "model_source": model_source,
        "score": round(float(score), 2),
        "threshold": model.threshold,
        "anomaly": bool(score > model.threshold) if model.threshold is not None else None,
    }
    if heatmap:
        png = padim.anomaly_overlay(data, anomaly_map, img_size=model.img_size)
        result["heatmap_png"] = base64.b64encode(png).decode("ascii")
    # Monitoring : verdict, distribution des scores, et score/seuil (indépendant de
    # l'échelle) — c'est le signal qui permet de détecter une dérive des données.
    metrics.observe_prediction(category, float(score), model.threshold, result["anomaly"])
    return result
