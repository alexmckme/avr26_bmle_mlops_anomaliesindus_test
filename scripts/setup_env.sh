#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# Prépare un environnement fraîchement cloné (idempotent).
#
#   ./scripts/setup_env.sh
#
# À lancer une fois après `git clone`, AVANT `docker compose up`.
# ─────────────────────────────────────────────────────────────
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "Configuration de l'environnement dans $PROJECT_ROOT"

# 1. Fichier de configuration
if [ -f .env ]; then
  echo "  = .env existe déjà"
else
  cp .env.example .env
  echo "  + .env créé depuis .env.example"
fi

# 2. Dossiers montés par docker-compose (données / modèles)
mkdir -p dataset/raw models
echo "  + dossiers présents : dataset/raw/, models/"

# 3. Backend SQLite de MLflow : DOIT être un FICHIER
#    (sinon Docker crée un dossier au niveau du bind mount)
if [ -f mlflow.db ]; then
  echo "  = mlflow.db existe déjà"
else
  : > mlflow.db
  echo "  + mlflow.db créé (fichier vide)"
fi

cat <<'EOF'

Étapes suivantes :

  1. Récupérer le dataset MVTec AD (Kaggle) :
       python scripts/download_data.py
     (ou, 100 % Docker :)
       docker run --rm -v "$PWD/dataset:/app/dataset" \
         -v "$HOME/.kaggle:/root/.kaggle:ro" -w /app anomalies-indus-api \
         python scripts/download_data.py --dest /app/dataset/raw

  2. Construire et démarrer la stack :
       docker compose up -d --build

  3. Ingérer les images dans MinIO :
       docker compose run --rm api python scripts/ingest_data.py --category bottle

  4. Entraîner + promouvoir le champion :
       docker compose run --rm api python scripts/training.py --category bottle --eval

  5. Prédire :
       curl -X POST localhost:8000/predict \
            -F category=bottle -F file=@dataset/raw/bottle/test/good/000.png

Interfaces : API http://localhost:8000/docs · MLflow http://localhost:5050 · MinIO http://localhost:9200
EOF
