"""Configuration du projet, chargée depuis l'environnement (.env).

Centralise les variables utilisées par les scripts et l'API :
MinIO (endpoint, identifiants, bucket) + dossier du dataset.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # optionnel si les variables sont déjà exportées
    load_dotenv = None

# Racine du projet = parents[1] de core/config.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if load_dotenv:
    load_dotenv(PROJECT_ROOT / ".env")


def _env(key: str, default: str) -> str:
    return os.getenv(key, default)


@dataclass(frozen=True)
class Settings:
    minio_endpoint: str
    minio_access_key: str
    minio_secret_key: str
    minio_secure: bool
    minio_bucket: str
    dataset_dir: Path


def load_settings() -> Settings:
    """Construit les réglages depuis l'environnement (avec défauts locaux)."""
    return Settings(
        minio_endpoint=_env("MINIO_ENDPOINT", "localhost:9100"),
        minio_access_key=_env("MINIO_ACCESS_KEY", "minioadmin"),
        minio_secret_key=_env("MINIO_SECRET_KEY", "minioadmin"),
        minio_secure=_env("MINIO_SECURE", "false").lower() in {"1", "true", "yes"},
        minio_bucket=_env("MINIO_BUCKET", "mvtec-ad"),
        dataset_dir=Path(_env("DATASET_DIR", "dataset/raw")),
    )
