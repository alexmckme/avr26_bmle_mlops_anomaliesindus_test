#!/usr/bin/env python3
"""Entraîne un modèle PaDiM pour UNE catégorie et sauvegarde l'artefact.

Lecture des images « good » depuis MinIO (prefix raw/<catégorie>/train/good)
ou depuis un dossier local (test sans MinIO). Le modèle est enregistré sous
models/<catégorie>.npz (mean + cov_inv + métadonnées).

Usage (depuis la racine du projet) :
    python scripts/training.py --category bottle
    python scripts/training.py --category bottle --eval           # + AUC rapide
    python scripts/training.py --category bottle --data-version 1 # 40 % du corpus
    python scripts/training.py --category screw  --fraction 0.5   # 50 % direct
    python scripts/training.py --category bottle --source local

Sous-ensembles (« croissance accumulée ») :
    data_version = index (0..4) ou 'full' dans la grille DATA_VERSION_FRACTIONS
    [0.2, 0.4, 0.6, 0.8, 1.0] ; chaque version garde les PREMIERS images du
    corpus trié -> versions imbriquées simulant un dataset qui grandit.
    fraction = alternative directe (0..1]. Défaut : dataset complet (full).

Remarque : 1 modèle par catégorie (les catégories MVTec sont hétérogènes).

Chaque entraînement est enregistré dans MLflow (params, métriques, artefact) ;
les artefacts sont stockés dans MinIO. Désactivable avec --no-mlflow.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import core.padim as padim


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", required=True, help="Catégorie MVTec (ex: bottle)")
    parser.add_argument("--source", choices=["minio", "local"], default="minio")
    parser.add_argument("--local-dir", default="dataset/raw",
                        help="Dossier racine local (source=local)")
    parser.add_argument("--out", default="models", help="Dossier des artefacts")
    parser.add_argument("--img-size", type=int, default=padim.IMG_SIZE)
    parser.add_argument("--data-version", default=None,
                        help="Version de données (0..4 ou 'full') dans la grille "
                             "de croissance ; défaut = full")
    parser.add_argument("--fraction", type=float, default=None,
                        help="Fraction directe du corpus (0..1], ex: 0.5")
    parser.add_argument("--eval", action="store_true",
                        help="Calcule l'AUC sur le split test (MinIO uniquement)")
    parser.add_argument("--no-mlflow", action="store_true",
                        help="Désactive le suivi MLflow pour ce run")
    parser.add_argument("--run-name", default=None, help="Nom du run MLflow")
    parser.add_argument("--no-register", action="store_true",
                        help="Ne pas enregistrer le modèle dans le Model Registry")
    parser.add_argument("--promote", action="store_true",
                        help="Promouvoir le modèle en 'champion' si son AUC est meilleure")
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
        eval_meta = padim.evaluate_on_test(client, settings.minio_bucket, model,
                                           args.category, args.img_size) if args.eval else {}
    else:
        source_dir = Path(args.local_dir) / args.category / "train" / "good"
        model, meta = padim.fit_from_dir(
            source_dir, img_size=args.img_size,
            data_version=args.data_version, fraction=args.fraction,
        )
        eval_meta = {}

    # Nom d'artefact porteur de la version (le « full » garde le nom simple)
    if meta.get("full", False):
        out_name = f"{args.category}.npz"
    elif meta.get("data_version") is not None:
        out_name = f"{args.category}.v{meta['data_version']}.npz"
    else:
        out_name = f"{args.category}.f{int(round(meta['fraction'] * 100))}.npz"
    out_path = Path(args.out) / out_name
    artifact = padim.save_model(model, out_path)

    meta.update({
        "artifact": artifact,
        "elapsed_s": round(time.time() - t0, 1),
    })
    meta.update(eval_meta)

    # Suivi MLflow (dégradation gracieuse si absent/indisponible)
    if not args.no_mlflow:
        import core.tracking as tracking

        if tracking.setup_mlflow():
            register_model = None if args.no_register else f"padim-{args.category}"
            info = tracking.log_training(meta, artifact, run_name=args.run_name,
                                         register_model=register_model)
            if info:
                meta["mlflow_run_id"] = info["run_id"]
                if info.get("model_version") is not None:
                    meta["mlflow_model_version"] = info["model_version"]
                if args.promote and register_model:
                    meta["mlflow_promotion"] = tracking.promote_if_better(register_model)

    print("\n" + "=" * 60)
    print("Entraînement PaDiM terminé")
    print("=" * 60)
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
