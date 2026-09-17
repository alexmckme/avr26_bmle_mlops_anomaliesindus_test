# syntax=docker/dockerfile:1
# Serveur MLflow (tracking + registry) — image légère.
# Backend store : SQLite (fichier monté) ; artefacts : MinIO (bucket "mlflow").
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DEFAULT_TIMEOUT=180 PIP_RETRIES=10

# boto3 : indispensable pour écrire les artefacts dans MinIO (S3)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && pip install mlflow boto3

EXPOSE 5000
