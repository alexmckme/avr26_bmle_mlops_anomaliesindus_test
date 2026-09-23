#!/usr/bin/env python3
"""Traçabilité d'un modèle : quel **code** et quelles **images** l'ont produit ?

Remonte la chaîne complète, depuis le modèle servi jusqu'aux données :

    alias du Registry (champion) -> run MLflow -> commit git + dataset_sha256
                                              -> index datasets.json
                                              -> manifeste (liste des images)

Usage (depuis la racine du projet, ou dans le conteneur api) :
    python scripts/lineage.py --category bottle
    python scripts/lineage.py --category bottle --alias candidate --images
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import core.tracking as tracking
import core.versioning as versioning


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", required=True, help="Catégorie MVTec (ex: bottle)")
    parser.add_argument("--alias", default="champion",
                        help="Alias du Registry à inspecter (défaut : champion)")
    parser.add_argument("--images", action="store_true",
                        help="Afficher la liste complète des images du manifeste")
    args = parser.parse_args()

    if not tracking.setup_mlflow():
        print("MLflow indisponible : impossible de lire le Registry")
        return 1

    import mlflow

    client = mlflow.MlflowClient()
    name = f"padim-{args.category}"
    try:
        version = client.get_model_version_by_alias(name, args.alias)
    except Exception:  # noqa: BLE001 — alias inexistant
        print(f"Aucune version '{args.alias}' pour le modèle {name}")
        return 1

    run = client.get_run(version.run_id)
    sha = version.tags.get("dataset_sha256") or run.data.tags.get("dataset_sha256", "?")

    print(f"\n{name} @{args.alias}  ->  version {version.version}")
    print(f"  run MLflow     : {version.run_id}")
    print(f"  commit du code : {version.tags.get('git_commit', '?')}")
    print(f"  AUC            : {run.data.metrics.get('auc', '?')}")
    print(f"  data_version   : {version.tags.get('data_version', '?')} "
          f"({run.data.params.get('n_train', '?')} images d'entraînement)")
    print(f"  dataset_sha256 : {sha}")

    # 1) Index du repo : ce que représente cette empreinte
    entry = versioning.read_dataset_index().get("datasets", {}).get(sha)
    if entry:
        runs = entry.get("runs", {})
        print(f"  index du repo  : {entry.get('n_images')} images sur "
              f"{entry.get('total_available')} disponibles, "
              f"{runs.get('count', 0)} run(s) enregistrés")
        print(f"                   source : {entry.get('source')}")
    else:
        print("  index du repo  : (empreinte absente de datasets.json)")

    # 2) Manifeste : la liste exacte des images utilisées par CE run
    try:
        manifest = mlflow.artifacts.load_dict(
            f"runs:/{version.run_id}/{versioning.MANIFEST_ARTIFACT}"
        )
    except Exception as exc:  # noqa: BLE001 — artefact absent (run antérieur)
        print(f"  manifeste      : indisponible ({exc})")
        return 0

    images = manifest.get("images", [])
    print(f"  manifeste      : {len(images)} images "
          f"(empreinte recalculée cohérente : {manifest.get('fingerprint_matches')})")
    for image in (images if args.images else images[:3]):
        print(f"      - {image['id']}  ({image['size']} o, etag {image.get('etag', '')[:12]})")
    if not args.images and len(images) > 3:
        print(f"      … et {len(images) - 3} autres images (--images pour tout afficher)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
