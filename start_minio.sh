#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# Démarre le serveur MinIO natif (version AGPL, sans licence)
# en arrière-plan. Usage : ./start_minio.sh
#
# Pourquoi les ports 9100/9200 ? 9000/9001 sont déjà utilisés
# par un kernel Jupyter local sur cette machine.
# ─────────────────────────────────────────────────────────────
set -euo pipefail

DATA_DIR="${MINIO_DATA_DIR:-./minio-data}"
LOG_FILE="minio.log"

# Si un serveur répond déjà sur le port de l'API S3, on ne fait rien.
if curl -s -o /dev/null "http://localhost:9100/minio/health/live"; then
    echo "MinIO tourne déjà : API S3 http://localhost:9100 | console http://localhost:9200"
    exit 0
fi

mkdir -p "$DATA_DIR"

nohup env MINIO_ROOT_USER="${MINIO_ROOT_USER:-minioadmin}" \
           MINIO_ROOT_PASSWORD="${MINIO_ROOT_PASSWORD:-minioadmin}" \
    minio server "$DATA_DIR" \
    --address ":9100" \
    --console-address ":9200" \
    > "$LOG_FILE" 2>&1 &

echo "MinIO démarré (PID $!)."
echo "  API S3  (utilisée par les scripts) : http://localhost:9100"
echo "  Console (interface web)            : http://localhost:9200"
echo "  Identifiants                       : minioadmin / minioadmin"
echo "  Logs                               : tail -f $LOG_FILE"
