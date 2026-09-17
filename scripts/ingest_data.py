#!/usr/bin/env python3
"""
Ingestion du dataset MVTec AD dans MinIO (object store S3).

Rôle (démonstration MLOps) :
    Construire la « base de données d'images » en poussant le contenu de
    dataset/raw vers un bucket MinIO. Les futurs endpoints /training et
    /predict liront les images depuis ce bucket (et non depuis le dossier).

Principes :
    - Script à exécuter UNE SEULE FOIS, mais relançable sans risque
      (idempotent : les objets déjà présents et identiques sont ignorés).
    - Pas de base SQL pour l'instant : les métadonnées sont attachées à
      chaque objet en *user metadata* (category, split, label, sha256,
      dimensions). Une table SQL pourra être ajoutée plus tard sans rien
      ré-ingérer.
    - dataset/raw reste la source read-only de référence.

Usage (depuis la racine du projet) :
    python scripts/ingest_data.py                     # ingestion complète
    python scripts/ingest_data.py --dry-run           # simple aperçu, sans upload
    python scripts/ingest_data.py --category bottle   # une seule catégorie (démo)

Configuration (voir .env) :
    MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY,
    MINIO_SECURE, MINIO_BUCKET, DATASET_DIR
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Rend le package `core` importable quel que soit le répertoire d'appel
# (ex. `python scripts/ingest_data.py` lancé depuis la racine du projet).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import load_settings
from core.storage import ensure_bucket, get_client

from minio import Minio  # pour les annotations de type
from minio.error import S3Error
from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
SPLITS_TO_INGEST = {"train", "test"}  # ground_truth (masques) non ingéré ici


# ─────────────────────────────────────────────────────────────
# Découverte des images
# ─────────────────────────────────────────────────────────────
def discover_images(root: Path, categories: set[str] | None = None) -> list[Path]:
    """Parcourt dataset/raw/<category>/{train,test}/<label>/*.<ext>.

    `categories` (optionnel) limite le parcours à certaines catégories.
    """
    images: list[Path] = []
    if not root.is_dir():
        print(f"[ERREUR] Dossier introuvable : {root}", file=sys.stderr)
        sys.exit(2)
    for cat_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if categories and cat_dir.name not in categories:
            continue
        for split_dir in sorted(p for p in cat_dir.iterdir() if p.is_dir()):
            if split_dir.name not in SPLITS_TO_INGEST:
                continue
            for label_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
                for f in sorted(label_dir.iterdir()):
                    if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                        images.append(f)
    return images


def split_from_path(p: Path) -> tuple[str, str, str]:
    """Retourne (category, split, label) à partir du chemin absolu.
    p = .../dataset/raw/<category>/<split>/<label>/<file>"""
    label_dir, split_dir, cat_dir = p.parent, p.parent.parent, p.parent.parent.parent
    return cat_dir.name, split_dir.name, label_dir.name


# ─────────────────────────────────────────────────────────────
# MinIO (client & bucket : voir core/storage.py)
# ─────────────────────────────────────────────────────────────
def object_key(category: str, split: str, label: str, filename: str) -> str:
    return f"raw/{category}/{split}/{label}/{filename}"


def object_sha256(client: Minio, bucket: str, key: str) -> str | None:
    """Renvoie le sha256 stocké en user-metadata, ou None."""
    try:
        obj = client.stat_object(bucket, key)
    except S3Error as e:
        if e.code == "NoSuchKey":
            return None
        raise
    meta = {k.lower(): v for k, v in (obj.metadata or {}).items()}
    return meta.get("x-amz-meta-sha256")


def compute_sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def image_metadata(path: Path, category: str, split: str, label: str) -> dict:
    """Métadonnées attachées à l'objet (préfigure la future table SQL)."""
    sha = compute_sha256(path)
    with Image.open(path) as im:
        width, height = im.size
        image_format = im.format or path.suffix.lstrip(".").upper()
    return {
        "category": category,
        "split": split,
        "label": label,
        "sha256": sha,
        "width": str(width),
        "height": str(height),
        "image-format": image_format,
    }


def upload_image(
    client: Minio, bucket: str, path: Path, verbose: bool = False
) -> tuple[str, int]:
    """Upload d'une image. Retourne (status, taille) avec status in
    {'uploaded', 'skipped'}."""
    category, split, label = split_from_path(path)
    key = object_key(category, split, label, path.name)
    meta = image_metadata(path, category, split, label)

    # Idempotence : si l'objet existe déjà avec le même sha256 -> skip
    existing = object_sha256(client, bucket, key)
    if existing == meta["sha256"]:
        return "skipped", path.stat().st_size

    content_type = f"image/{path.suffix.lstrip('.').lower()}"
    client.fput_object(
        bucket,
        key,
        str(path),
        content_type=content_type,
        metadata=meta,
    )
    if verbose:
        print(f"  + {key}  ({meta['width']}x{meta['height']})")
    return "uploaded", path.stat().st_size


# ─────────────────────────────────────────────────────────────
# Rapport / aide
# ─────────────────────────────────────────────────────────────
def human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def print_category_counts(images: list[Path]) -> None:
    counter: dict[str, Counter] = defaultdict(Counter)
    for p in images:
        category, split, label = split_from_path(p)
        counter[category][split] += 1
    print("\nImages découvertes par catégorie :")
    for cat in sorted(counter):
        parts = ", ".join(f"{s}={n}" for s, n in sorted(counter[cat].items()))
        print(f"  {cat:<12} {parts}")
    print(f"  TOTAL : {len(images)} images")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default=None,
                        help="Dossier source (défaut : DATASET_DIR ou dataset/raw)")
    parser.add_argument("--bucket", default=None,
                        help="Bucket MinIO (défaut : MINIO_BUCKET ou mvtec-ad)")
    parser.add_argument("--category", default=None,
                        help="Limiter à une/des catégories (ex: bottle ou bottle,cable)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Découvre les images et s'arrête avant l'upload")
    parser.add_argument("--verbose", action="store_true",
                        help="Affiche chaque fichier uploadé")
    args = parser.parse_args()

    settings = load_settings()

    dataset_dir = Path(args.dataset_dir or settings.dataset_dir)
    bucket = args.bucket or settings.minio_bucket

    categories = None
    if args.category:
        categories = {c.strip() for c in args.category.split(",") if c.strip()}
        if dataset_dir.is_dir():
            available = {p.name for p in dataset_dir.iterdir() if p.is_dir()}
            unknown = categories - available
            if unknown:
                print(f"[ERREUR] Catégorie(s) inconnue(s) : {', '.join(sorted(unknown))}",
                      file=sys.stderr)
                print(f"          Disponibles : {', '.join(sorted(available))}", file=sys.stderr)
                return 2

    images = discover_images(dataset_dir, categories=categories)
    print_category_counts(images)

    if args.dry_run:
        print("\n[dry-run] Aucun upload effectué.")
        return 0
    if not images:
        print("[INFO] Rien à ingérer.", file=sys.stderr)
        return 1

    client = get_client(settings)
    ensure_bucket(client, bucket)

    counts = Counter()
    total_bytes = 0
    errors: list[str] = []
    total = len(images)
    for i, path in enumerate(images, start=1):
        try:
            status, size = upload_image(client, bucket, path, verbose=args.verbose)
            counts[status] += 1
            total_bytes += size
        except Exception as exc:  # noqa: BLE001 — on continue sur les autres
            counts["errors"] += 1
            errors.append(f"{path}: {exc}")
        if i % 250 == 0:
            print(f"[PROGRESS] {i}/{total} — "
                  f"uploads={counts['uploaded']} skips={counts['skipped']}")

    # Vérification finale : on re-scanne le bucket pour compter le stockage.
    bucket_objects = 0
    bucket_bytes = 0
    for obj in client.list_objects(bucket, recursive=True):
        bucket_objects += 1
        bucket_bytes += obj.size

    print("\n" + "=" * 58)
    print("Résumé de l'ingestion")
    print("=" * 58)
    print(f"  Fichiers découverts      : {total}")
    print(f"  Nouveaux uploads         : {counts['uploaded']}")
    print(f"  Déjà présents (skippés)  : {counts['skipped']}")
    print(f"  Erreurs                  : {counts['errors']}")
    print(f"  Bucket                   : s3://{bucket}/")
    print(f"  Objets dans le bucket    : {bucket_objects}")
    print(f"  Taille stockée           : {human_size(bucket_bytes)}")
    print("=" * 58)

    if errors:
        print("\nErreurs rencontrées (les 10 premières) :")
        for err in errors[:10]:
            print(f"  - {err}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
