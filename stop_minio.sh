#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# Arrête le serveur MinIO natif lancé par ./start_minio.sh.
# Usage : ./stop_minio.sh
# ─────────────────────────────────────────────────────────────
if pkill -f "minio server ./minio-data" 2>/dev/null; then
    echo "MinIO arrêté."
else
    echo "Aucun serveur MinIO en cours d'exécution."
fi
