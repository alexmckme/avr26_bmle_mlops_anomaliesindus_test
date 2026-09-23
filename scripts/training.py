#!/usr/bin/env python3
"""Entraîne un modèle PaDiM pour UNE catégorie et sauvegarde l'artefact.

Lecture des images « good » depuis MinIO (prefix raw/<catégorie>/train/good)
ou depuis un dossier local (test sans MinIO). Le modèle est enregistré sous
models/<catégorie>.npz (mean + cov_inv + métadonnées).

Usage (depuis la racine du projet) :
    python scripts/training.py --category bottle                 # 100 % (full)
    python scripts/training.py --category bottle --eval          # + AUC rapide
    python scripts/training.py --category bottle --data-version 2 # 30 % du corpus
    python scripts/training.py --category screw  --fraction 0.5   # 50 % direct
    python scripts/training.py --category bottle --source local

Sous-ensembles (« croissance accumulée » par pas de 10 %) :
    data_version = index (0..9) ou 'full' dans la grille DATA_VERSION_FRACTIONS
    [0.1, 0.2, …, 0.9, 1.0] : v0 = 10 %, v1 = 20 %… v9 = 100 %. Chaque version
garde les PREMIERS images du corpus trié -> versions imbriquées simulant un
dataset qui grandit. `fraction` (0..1]) est une alternative directe.
    Défaut : dataset complet (full).

Remarque : 1 modèle par catégorie (les catégories MVTec sont hétérogènes).

Chaque entraînement est enregistré dans MLflow (params, métriques, artefact), les
artefacts sont stockés dans MinIO, et le modèle est promu automatiquement en
« champion » s'il fait au moins aussi bien que le champion en place (comparaison
sur l'AUC ; nécessite --eval). Désactivable avec --no-mlflow / --no-promote.

Traçabilité (versioning code + données) : chaque run porte le **commit git** du
code et un artefact `dataset_manifest.json` listant les images utilisées ; l'index
`datasets.json` (fichier du repo, à committer) catalogue les jeux de données par
leur empreinte `dataset_sha256`. Voir `core/versioning.py`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import core.padim as padim
import core.versioning as versioning


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", required=True, help="Catégorie MVTec (ex: bottle)")
    parser.add_argument("--source", choices=["minio", "local"], default="minio")
    parser.add_argument("--local-dir", default="dataset/raw",
                        help="Dossier racine local (source=local)")
    parser.add_argument("--out", default="models", help="Dossier des artefacts")
    parser.add_argument("--img-size", type=int, default=padim.IMG_SIZE)
    parser.add_argument("--data-version", default=None,
                        # NB : « %% » car argparse formate les aides avec `%`.
                        help=f"Version de données (0..{len(padim.DATA_VERSION_FRACTIONS) - 1} "
                             "ou 'full') : grille par pas de 10 %% ; "
                             "défaut = full (100 %%)")
    parser.add_argument("--fraction", type=float, default=None,
                        help="Fraction directe du corpus (0..1], ex: 0.5")
    parser.add_argument("--eval", action="store_true",
                        help="Calcule l'AUC sur le split test (MinIO uniquement)")
    parser.add_argument("--no-mlflow", action="store_true",
                        help="Désactive le suivi MLflow pour ce run")
    parser.add_argument("--run-name", default=None, help="Nom du run MLflow")
    parser.add_argument("--no-register", action="store_true",
                        help="Ne pas enregistrer le modèle dans le Model Registry")
    parser.add_argument("--no-promote", action="store_true",
                        help="Ne pas promouvoir automatiquement en 'champion'")
    args = parser.parse_args()

    t0 = time.time()

    if args.source == "minio":
        from core.config import load_settings
        from core.storage import get_client

        settings = load_settings()
        client = get_client(settings)
        model, meta = padim.fit_from_minio(
            client, settings.minio_bucket, args.category, img_size=args.img_size,
            data_version=args.data_version, fraction=args.fraction,
        )
        # Manifeste : les images exactes utilisées (traçabilité data)
        manifest = versioning.minio_manifest(client, settings.minio_bucket,
                                             args.category, meta)
        eval_meta = padim.evaluate_on_test(client, settings.minio_bucket, model,
                                           args.category, args.img_size) if args.eval else {}
    else:
        source_dir = Path(args.local_dir) / args.category / "train" / "good"
        model, meta = padim.fit_from_dir(
            source_dir, img_size=args.img_size,
            data_version=args.data_version, fraction=args.fraction,
        )
        manifest = versioning.dir_manifest(Path(args.local_dir), args.category, meta)
        eval_meta = {}

    # Nom d'artefact explicite : `bottle.p40.npz` (40 %) ou `bottle.npz` (100 %)
    out_path = Path(args.out) / padim.artifact_name(args.category, meta)
    artifact = padim.save_model(model, out_path)

    meta.update({
        "artifact": artifact,
        "elapsed_s": round(time.time() - t0, 1),
    })
    meta.update(eval_meta)

    # Commit du code (apparaît dans les params MLflow et dans la sortie du script)
    versioning.add_code_info(meta)

    # Suivi MLflow (dégradation gracieuse si absent/indisponible)
    if not args.no_mlflow:
        import core.tracking as tracking

        if tracking.setup_mlflow():
            register_model = None if args.no_register else f"padim-{args.category}"
            info = tracking.log_training(meta, artifact, run_name=args.run_name,
                                         register_model=register_model,
                                         manifest=manifest)
            if info:
                meta["mlflow_run_id"] = info["run_id"]
                if info.get("model_version") is not None:
                    meta["mlflow_model_version"] = info["model_version"]
                if register_model and not args.no_promote:
                    promo = tracking.promote_if_better(register_model)
                    meta["mlflow_promotion"] = promo
                    tracking.log_promotion(info["run_id"], promo, meta=meta)

    print("\n" + "=" * 60)
    print("Entraînement PaDiM terminé")
    print("=" * 60)
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
