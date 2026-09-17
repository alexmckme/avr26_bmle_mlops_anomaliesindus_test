# syntax=docker/dockerfile:1
# Image de l'API Anomalies Indus (FastAPI + PaDiM/TensorFlow).
# Multi-architectures : le même Dockerfile sert linux/amd64 et linux/arm64
# (les wheels TensorFlow sont choisies via les marqueurs pip de requirements.txt).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=180 \
    PIP_RETRIES=10

# libgomp1 : requis par TensorFlow ; curl : healthcheck HTTP
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dépendances (cache pip persistant -> un build interrompu ne repart pas de zéro)
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && pip install -r requirements.txt

# Code applicatif
COPY core ./core
COPY api ./api
COPY scripts ./scripts

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
