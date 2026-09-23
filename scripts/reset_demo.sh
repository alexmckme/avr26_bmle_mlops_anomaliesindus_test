#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# Remise à blanc pour une démonstration « live ».
#
# Par défaut : on CONSERVE les images MinIO (bucket mvtec-ad) et on remet à
# zéro tout le côté MLflow (artefacts, base des runs, modèles locaux).
#
#   ./scripts/reset_demo.sh            # reset MLflow + modèles (images gardées)
#   ./scripts/reset_demo.sh --full     # + vide aussi le bucket des images
#   ./scripts/reset_demo.sh --yes      # sans demande de confirmation
#   ./scripts/reset_demo.sh --restore  # restaure la dernière sauvegarde
#
# Aucune donnée n'est supprimée : tout est DÉPLACÉ dans une sauvegarde
# ($BACKUP_ROOT/<horodatage>, défaut /tmp/anomalies-demo-backup).
# ─────────────────────────────────────────────────────────────
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

BACKUP_ROOT="${BACKUP_ROOT:-/tmp/anomalies-demo-backup}"
FULL=0
ASSUME_YES=0
RESTORE=0
for arg in "$@"; do
  case "$arg" in
    --full) FULL=1 ;;
    --yes|-y) ASSUME_YES=1 ;;
    --restore) RESTORE=1 ;;
    -h|--help) sed -n '3,14p' "$0"; exit 0 ;;
    *) echo "Option inconnue : $arg" >&2; exit 2 ;;
  esac
done

# ── Restauration de la dernière sauvegarde ──
if [ "$RESTORE" = 1 ]; then
  latest="$(ls -1d "$BACKUP_ROOT"/*/ 2>/dev/null | sort | tail -1 || true)"
  if [ -z "$latest" ]; then
    echo "Aucune sauvegarde dans $BACKUP_ROOT"; exit 1
  fi
  echo "Restauration depuis $latest"
  docker compose down --remove-orphans >/dev/null 2>&1 || true
  while IFS=$'\t' read -r kind dest name; do
    [ -n "${kind:-}" ] || continue
    if [ "$kind" = "PATH" ]; then
      rm -rf "$dest"
      mv "$latest/$name" "$dest"
    else
      mkdir -p "$dest"
      if [ -d "$latest/$name" ]; then
        mv "$latest/$name"/* "$dest"/ 2>/dev/null || true
      fi
    fi
  done < "$latest/MANIFEST"
  rm -f mlflow.db-wal mlflow.db-shm
  echo "Restauration terminée. Relance : docker compose up -d"
  exit 0
fi

if [ "$FULL" = 1 ]; then
  echo "Mode : COMPLET (les images MinIO seront aussi vidées)"
else
  echo "Mode : ciblé (images MinIO conservées, MLflow remis à zéro)"
fi
if [ "$ASSUME_YES" != 1 ]; then
  read -r -p "Confirmer le reset ? [y/N] " answer
  case "$answer" in
    [yY]) ;;
    *) echo "Annulé."; exit 0 ;;
  esac
fi

BACKUP_DIR="$BACKUP_ROOT/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$BACKUP_DIR"
: > "$BACKUP_DIR/MANIFEST"

# Déplace un fichier/dossier entier
move_path() {
  local src="$1" name="$2"
  [ -e "$src" ] || return 0
  mv "$src" "$BACKUP_DIR/$name"
  printf 'PATH\t%s\t%s\n' "$src" "$name" >> "$BACKUP_DIR/MANIFEST"
  echo "  sauvegardé : $src"
}

# Déplace le CONTENU d'un dossier (le dossier lui-même reste en place)
empty_dir() {
  local src="$1" name="$2"
  [ -d "$src" ] || return 0
  if [ -n "$(ls -A "$src" 2>/dev/null)" ]; then
    mkdir -p "$BACKUP_DIR/$name"
    mv "$src"/* "$BACKUP_DIR/$name"/ 2>/dev/null || true
    printf 'CONTENTS\t%s\t%s\n' "$src" "$name" >> "$BACKUP_DIR/MANIFEST"
    echo "  vidé       : $src"
  fi
}

echo "1) Arrêt de la stack…"
docker compose down --remove-orphans >/dev/null 2>&1 || true

echo "2) Sauvegarde + remise à zéro…"
empty_dir "minio-data/mlflow" "mlflow-bucket"
if [ "$FULL" = 1 ]; then
  empty_dir "minio-data/mvtec-ad" "mvtec-ad-bucket"
fi
move_path "models" "models"
move_path "mlflow.db" "mlflow.db"
move_path "mlflow.db-wal" "mlflow.db-wal"
move_path "mlflow.db-shm" "mlflow.db-shm"
move_path "datasets.json" "datasets.json"

# Docker créerait un DOSSIER si le fichier du bind mount est absent
[ -f mlflow.db ] || : > mlflow.db
printf '{\n  "schema": 1,\n  "datasets": {}\n}\n' > datasets.json
mkdir -p models

echo "3) Redémarrage de la stack…"
docker compose up -d >/dev/null
docker compose ps --format '  {{.Name}}\t{{.Status}}'

echo
echo "Sauvegarde : $BACKUP_DIR"
echo "Restaurer  : ./scripts/reset_demo.sh --restore"
echo
echo "Démo :"
echo "  docker compose run --rm api python scripts/training.py --category bottle --eval"
echo "  curl -X POST localhost:8000/predict -F category=bottle -F file=@dataset/raw/bottle/test/good/000.png"
