"""Client HTTP de l'API Anomalies Indus.

L'interface Streamlit n'importe **que** ce module : aucune logique ML, aucun
TensorFlow. Conséquences voulues :

  - l'image Streamlit reste légère et démarre en quelques secondes ;
  - le rendu est strictement celui du service (même code que la CLI, anti
    train/serve skew) ;
  - si l'API est arrêtée, l'interface le dit clairement au lieu de planter.
"""

from __future__ import annotations

import base64
import os
from typing import Any

import requests

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
# L'entraînement peut durer plusieurs minutes (cache TensorFlow froid) : timeout large.
TIMEOUT = float(os.getenv("API_TIMEOUT", "900"))
SHORT_TIMEOUT = float(os.getenv("API_SHORT_TIMEOUT", "5"))


class ApiError(RuntimeError):
    """Erreur de l'API, avec un message directement affichable."""


def _request(method: str, path: str, timeout: float = TIMEOUT, **kwargs) -> Any:
    url = f"{API_BASE_URL}{path}"
    try:
        response = requests.request(method, url, timeout=timeout, **kwargs)
    except requests.RequestException as exc:
        raise ApiError(f"API injoignable ({url}) — {exc}") from exc

    if response.status_code >= 400:
        try:
            detail = response.json().get("detail")
        except ValueError:
            detail = response.text[:200]
        raise ApiError(f"HTTP {response.status_code} — {detail}")
    return response.json()


# ── Lecture ───────────────────────────────────────────────────────────────────
def health() -> dict:
    """Infos de l'API (`GET /`) — sert au voyant d'état."""
    return _request("GET", "/", timeout=SHORT_TIMEOUT)


def models() -> dict:
    """Catégories connues + champion/candidate de chacune (`GET /models`)."""
    return _request("GET", "/models")


def runs(limit: int = 20, category: str | None = None) -> dict:
    """Derniers runs d'entraînement (`GET /runs`)."""
    params = {"limit": limit}
    if category:
        params["category"] = category
    return _request("GET", "/runs", params=params)


def metrics_text() -> str:
    """Métriques Prometheus brutes de l'API (`GET /metrics`)."""
    try:
        response = requests.get(f"{API_BASE_URL}/metrics", timeout=SHORT_TIMEOUT)
    except requests.RequestException as exc:
        raise ApiError(f"API injoignable ({API_BASE_URL}/metrics) — {exc}") from exc
    return response.text


# ── Actions ───────────────────────────────────────────────────────────────────
def predict(category: str, image: bytes, filename: str = "image.png",
            heatmap: bool = True) -> dict:
    """`POST /predict` — score d'anomalie (+ carte d'anomalie si demandée)."""
    return _request(
        "POST", "/predict",
        data={"category": category, "heatmap": str(bool(heatmap)).lower()},
        files={"file": (filename, image, "image/png")},
    )


def train(payload: dict) -> dict:
    """`POST /training` — entraîne une catégorie (et peut promouvoir le champion)."""
    return _request("POST", "/training", json=payload)


def to_bytes(base64_png: str) -> bytes:
    """Décode le PNG base64 renvoyé par `/predict`."""
    return base64.b64decode(base64_png)
