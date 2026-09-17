#!/usr/bin/env python3
"""Étape 0 — Récupère le dataset MVTec AD (Kaggle) et le range dans dataset/raw/.

Le script :
  1. télécharge le dataset Kaggle (`--dataset`, défaut ipythonx/mvtec-ad) ;
  2. découvre les dossiers de catégories, quelle que soit l'arborescence du zip
     (un dossier « catégorie » = un dossier contenant `train/good/` ET `test/`) ;
  3. les déplace dans `dataset/raw/<catégorie>/` (structure attendue par
     `scripts/ingest_data.py`).

Idempotent : les catégories déjà présentes sont ignorées (sauf `--force`).
À lancer sur l'HÔTE (le dossier `dataset/raw` est monté en lecture seule dans
le conteneur `api`).

Prérequis — authentification Kaggle (au choix) :
  - recommandé : `kaggle auth login`  (OAuth, identifiants mis en cache localement)
  - token API : `export KAGGLE_API_TOKEN=...`  ou fichier `~/.kaggle/access_token`
  - historique : `~/.kaggle/kaggle.json` (chmod 600) ou `KAGGLE_USERNAME`/`KAGGLE_KEY`

Usage (depuis la racine du projet) :
    python scripts/download_data.py                  # télécharge + range
    python scripts/download_data.py --dry-run        # montre le plan (sans rien faire)
    python scripts/download_data.py --force          # remplace les catégories existantes
    python scripts/download_data.py --from-dir /chemin/vers/zip/extrait
    python scripts/download_data.py --keep-archive   # conserve le .zip téléchargé
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

# Les 15 catégories officielles MVTec AD
MVTEC_CATEGORIES = {
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood",
    "zipper",
}

DEFAULT_DATASET = "ipythonx/mvtec-ad"


def has_kaggle_credentials() -> bool:
    """Vrai si des identifiants Kaggle sont disponibles (OAuth, token ou legacy)."""
    if os.getenv("KAGGLE_API_TOKEN"):
        return True
    if os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"):
        return True
    kaggle_dir = Path.home() / ".kaggle"
    return any((kaggle_dir / name).is_file()
               for name in ("access_token", "kaggle.json"))


def find_category_dirs(root: Path, expected: set[str]) -> dict[str, Path]:
    """Découvre les dossiers de catégories (contiennent train/good/ et test/)."""
    found: dict[str, Path] = {}
    for dirpath, _dirnames, _filenames in os.walk(root):
        path = Path(dirpath)
        if path.name in expected and (path / "train" / "good").is_dir() \
                and (path / "test").is_dir():
            found.setdefault(path.name, path)
    return found


def download_from_kaggle(slug: str, staging: Path, keep_archive: bool) -> None:
    """Télécharge et décompresse le dataset Kaggle dans `staging`."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("[ERREUR] Paquet « kaggle » absent : pip install kaggle", file=sys.stderr)
        sys.exit(2)

    api = KaggleApi()
    api.authenticate()   # lève une erreur si les identifiants sont absents
    print(f"[INFO] Téléchargement de {slug} -> {staging}")
    api.dataset_download_files(slug, path=str(staging), unzip=True, quiet=False)

    if not keep_archive:
        for archive in staging.glob("*.zip"):
            archive.unlink()
            print(f"[INFO] Archive supprimée : {archive.name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET,
                        help=f"Slug Kaggle (défaut : {DEFAULT_DATASET})")
    parser.add_argument("--dest", default="dataset/raw",
                        help="Dossier cible des catégories (défaut : dataset/raw)")
    parser.add_argument("--staging", default="dataset/_kaggle",
                        help="Dossier temporaire de téléchargement")
    parser.add_argument("--from-dir", default=None,
                        help="Utiliser un dossier déjà extrait (pas de téléchargement)")
    parser.add_argument("--force", action="store_true",
                        help="Remplacer les catégories déjà présentes")
    parser.add_argument("--keep-archive", action="store_true",
                        help="Conserver le .zip téléchargé")
    parser.add_argument("--dry-run", action="store_true",
                        help="Afficher le plan sans rien modifier")
    args = parser.parse_args()

    dest = Path(args.dest)

    # ── 1. Récupération des fichiers ──
    if args.from_dir:
        source = Path(args.from_dir)
        if not source.is_dir():
            print(f"[ERREUR] Dossier introuvable : {source}", file=sys.stderr)
            return 2
        print(f"[INFO] Source locale : {source}")
        cleanup_staging = None
    else:
        staging = Path(args.staging)
        if args.dry_run:
            print(f"[INFO] (dry-run) dataset Kaggle visé : {args.dataset}")
            print(f"[INFO] (dry-run) identifiants Kaggle détectés : "
                  f"{'oui' if has_kaggle_credentials() else 'NON'}")
            print("[INFO] (dry-run) aucun téléchargement effectué.")
            source = staging if staging.is_dir() else None
            cleanup_staging = None
            if source is None:
                return 0
        else:
            if not has_kaggle_credentials():
                print("[ERREUR] Identifiants Kaggle introuvables.", file=sys.stderr)
                print("  Authentification (au choix) :", file=sys.stderr)
                print("    1. kaggle auth login                    (recommandé, OAuth)",
                      file=sys.stderr)
                print("    2. export KAGGLE_API_TOKEN=...           (token API)",
                      file=sys.stderr)
                print("       ou fichier ~/.kaggle/access_token", file=sys.stderr)
                print("    3. ~/.kaggle/kaggle.json (chmod 600)     (historique)",
                      file=sys.stderr)
                print("  Tokens : https://www.kaggle.com/settings/api", file=sys.stderr)
                return 2
            staging.mkdir(parents=True, exist_ok=True)
            download_from_kaggle(args.dataset, staging, args.keep_archive)
            source = staging
            cleanup_staging = staging

    # ── 2. Découverte des catégories ──
    print(f"[INFO] Recherche des catégories dans {source}…")
    found = find_category_dirs(source, MVTEC_CATEGORIES)
    if not found:
        print(f"[ERREUR] Aucune catégorie MVTec trouvée dans {source} "
              f"(dossier « train/good » + « test » attendu).", file=sys.stderr)
        return 2

    missing = sorted(MVTEC_CATEGORIES - set(found))
    print(f"[INFO] Catégories trouvées : {len(found)}/15")
    if missing:
        print(f"[WARN] Manquantes : {', '.join(missing)}")

    # ── 3. Placement dans dataset/raw/ ──
    print(f"[INFO] Destination : {dest}")
    moved, skipped = [], []
    dest.mkdir(parents=True, exist_ok=True)

    for name in sorted(found):
        src, dst = found[name], dest / name
        if dst.exists() and not args.force:
            print(f"  = {name} : déjà présent (ignoré)")
            skipped.append(name)
            continue
        print(f"  + {name} -> {dst}")
        if args.dry_run:
            continue
        if dst.exists():
            shutil.rmtree(dst)
        shutil.move(str(src), str(dst))
        moved.append(name)

    # ── 4. Nettoyage ──
    if cleanup_staging is not None and not args.dry_run and not args.keep_archive:
        shutil.rmtree(cleanup_staging, ignore_errors=True)

    print("\n" + "=" * 60)
    print("Téléchargement / rangement du dataset terminé")
    print("=" * 60)
    print(f"  Catégories trouvées : {len(found)}/15")
    print(f"  Placées             : {len(moved)}")
    print(f"  Déjà présentes      : {len(skipped)}")
    if missing:
        print(f"  Manquantes          : {', '.join(missing)}")
    print(f"  Dossier             : {dest}/")
    print("=" * 60)
    if not args.dry_run:
        print("\nÉtape suivante : python scripts/ingest_data.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
