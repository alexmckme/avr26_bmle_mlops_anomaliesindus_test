"""DAG de démonstration Airflow — Anomalies Indus.

But, volontairement minimal : prouver l'**automatisation de l'exécution** du
pipeline d'entraînement. Airflow ne contient AUCUNE logique de ML — le DAG appelle
`POST /training` sur l'API, qui reste la seule source de vérité (anti
train/serve skew, et image Airflow légère : pas de TensorFlow).

- Planification : variable `TRAINING_SCHEDULE` (défaut `* * * * *`, toutes les minutes).
- Un run **planifié** est volontairement léger : `register: false` ⇒ seulement les
  métriques MLflow, sans téléverser l'artefact de ~260 Mo dans MinIO (à 1 run/min
  avec enregistrement, ce serait ~15 Go/heure d'artefacts).
- Pour un run **complet** (version enregistrée dans le Registry + promotion
  automatique du champion), déclencher manuellement le DAG depuis l'UI
  (« Trigger DAG w/ config ») avec par exemple :

      {"category": "bottle", "data_version": 4, "eval": true, "register": true}

Toutes les valeurs ci-dessous sont surchargeables via `dag_run.conf`.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

# ── Réglages (variables d'environnement du conteneur, voir docker-compose.yml) ──
API_BASE_URL = os.getenv("API_BASE_URL", "http://api:8000")
TRAINING_SCHEDULE = os.getenv("TRAINING_SCHEDULE", "* * * * *")

# ── Valeurs par défaut d'un run (surchargeables via dag_run.conf) ──────────────
CATEGORY = "bottle"
DATA_VERSION = "0"   # "0".."4" (jeux d'entraînement cumulatifs) ou "full"
EVAL = "True"        # calcule l'AUC sur le split test
REGISTER = "False"   # True => version dans le Registry (+ promotion du champion)

# 1) Vérifier que l'API répond (échec rapide et lisible si la stack est incomplète).
CHECK_API_CMD = f"curl -fsS --max-time 15 {API_BASE_URL}/"

# 2) Déclencher l'entraînement : un simple appel HTTP, rien de plus.
TRAIN_CMD = r"""
set -euo pipefail
payload="{\"category\": \"{{ (dag_run.conf or {}).get('category', '%s') }}\", \"data_version\": \"{{ (dag_run.conf or {}).get('data_version', '%s') }}\", \"eval\": {{ (dag_run.conf or {}).get('eval', %s) | string | lower }}, \"register\": {{ (dag_run.conf or {}).get('register', %s) | string | lower }}}"
echo "POST %s/training  ${payload}"
curl -fsS --max-time 900 -X POST "%s/training" -H 'Content-Type: application/json' -d "${payload}"
""" % (CATEGORY, DATA_VERSION, EVAL, REGISTER, API_BASE_URL, API_BASE_URL)

with DAG(
    dag_id="training_pipeline",
    description="Entraînement PaDiM planifié via POST /training (démo orchestration)",
    doc_md=__doc__,
    start_date=datetime(2026, 1, 1),
    schedule=TRAINING_SCHEDULE,
    catchup=False,                       # pas de rattrapage des intervalles passés
    max_active_runs=1,                   # jamais 2 entraînements en parallèle
    dagrun_timeout=timedelta(minutes=10),
    default_args={"retries": 1, "retry_delay": timedelta(seconds=30)},
    tags=["anomalies-indus", "training", "demo"],
) as dag:
    check_api = BashOperator(
        task_id="check_api",
        bash_command=CHECK_API_CMD,
        doc_md="Vérifie que l'API répond (`GET /`) avant de lancer l'entraînement.",
    )

    train = BashOperator(
        task_id="train",
        bash_command=TRAIN_CMD,
        doc_md=(
            "Appelle `POST /training` (voir `dag_run.conf` pour la catégorie, la "
            "version de données, l'évaluation et l'enregistrement au Registry)."
        ),
    )

    check_api >> train
