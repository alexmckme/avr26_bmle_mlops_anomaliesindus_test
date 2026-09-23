# syntax=docker/dockerfile:1
# Image de l'interface de démonstration (Streamlit).
#
# Volontairement minimale : l'application est un **client HTTP** de l'API, donc
# aucune dépendance ML ici (pas de TensorFlow) — l'image est ~10x plus légère que
# celle de l'API et démarre en quelques secondes.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=180 \
    PIP_RETRIES=10 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip

COPY docker/streamlit/requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Le code de l'UI est aussi monté en volume par docker-compose (édition à chaud).
COPY streamlit_app ./streamlit_app

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app/app.py"]
