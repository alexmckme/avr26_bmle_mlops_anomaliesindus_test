"""Helpers MinIO (object store S3) partagés par scripts/ et api/.

Expose un client unique construit depuis `core.config`, pour éviter de
dupliquer la connexion dans ingest_data.py, training.py, predict.py et l'API.
"""

from __future__ import annotations

from minio import Minio

from core.config import Settings


def get_client(settings: Settings) -> Minio:
    """Construit le client S3/MinIO depuis la configuration."""
    return Minio(
        settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        secure=settings.minio_secure,
    )


def ensure_bucket(client: Minio, bucket: str) -> bool:
    """Crée le bucket s'il n'existe pas. Retourne True si créé."""
    if client.bucket_exists(bucket):
        return False
    client.make_bucket(bucket)
    print(f"[INFO] Bucket créé : {bucket}")
    return True
