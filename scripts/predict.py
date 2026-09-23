#!/usr/bin/env python3
"""Prédit le score d'anomalie d'une image avec un modèle PaDiM entraîné.

L'image peut venir d'un fichier local (`--image`) ou d'un objet MinIO (`--key`).
Le modèle est résolu automatiquement : `models/<catégorie>.npz` (dataset complet)
sinon la version la plus récente `models/<catégorie>.v*.npz` / `.f*.npz`.

Usage (depuis la racine du projet, venv activé) :
    python scripts/predict.py --category bottle --image dataset/raw/bottle/test/good/000.png
    python scripts/predict.py --category bottle --key raw/bottle/test/broken_large/000.png
    python scripts/predict.py --category bottle --image img.png --heatmap /tmp/heat.png
    python scripts/predict.py --category bottle --image img.png --json

Le verdict (`anomaly` true/false) nécessite un seuil, calculé lors d'un
entraînement avec `--eval`. Sans seuil, seul le score brut est affiché.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import core.padim as padim


def _read_image_bytes(args) -> bytes:
    """Lit les octets de l'image depuis un fichier local ou un objet MinIO."""
    if args.image:
        return Path(args.image).read_bytes()
    from core.config import load_settings
    from core.storage import get_client

    settings = load_settings()
    client = get_client(settings)
    return client.get_object(settings.minio_bucket, args.key).read()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", required=True, help="Catégorie MVTec (ex: bottle)")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="Chemin d'une image locale")
    src.add_argument("--key", help="Clé d'un objet MinIO (ex: raw/bottle/test/good/000.png)")
    parser.add_argument("--model", default=None,
                        help="Chemin explicite d'un artefact .npz (sinon auto)")
    parser.add_argument("--no-registry", action="store_true",
                        help="Ignorer le Registry MLflow et utiliser models/ local")
    parser.add_argument("--heatmap", default=None,
                        help="Sauvegarder la heatmap d'anomalie dans ce fichier PNG")
    parser.add_argument("--json", action="store_true", help="Sortie au format JSON")
    args = parser.parse_args()

    # 1. Résolution du modèle (champion MLflow si dispo, sinon local)
    try:
        if args.model:
            model_path, origin = Path(args.model), "explicit"
        else:
            import core.tracking as tracking

            model_path, origin = tracking.resolve_inference_model(
                args.category, use_registry=not args.no_registry
            )
    except FileNotFoundError as exc:
        print(f"[ERREUR] {exc}", file=sys.stderr)
        return 1

    # 2. Lecture de l'image
    try:
        data = _read_image_bytes(args)
    except FileNotFoundError as exc:
        print(f"[ERREUR] Image introuvable : {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 — MinIO / réseau / lecture
        print(f"[ERREUR] Lecture de l'image : {exc}", file=sys.stderr)
        return 1

    # 3. Prédiction
    model = padim.load_model(model_path)
    try:
        score, heatmap = padim.predict_image(data, model)
    except Exception as exc:  # noqa: BLE001 — image illisible
        print(f"[ERREUR] Prédiction impossible : {exc}", file=sys.stderr)
        return 1

    anomaly = None if model.threshold is None else bool(score > model.threshold)
    result = {
        "category": args.category,
        "model": model_path.name,
        "model_source": origin,
        "score": round(float(score), 2),
        "threshold": model.threshold,
        "anomaly": anomaly,
    }
    if args.heatmap:
        # Même rendu que `/predict` (heatmap=true) : une seule implémentation.
        out_path = Path(args.heatmap)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(padim.anomaly_overlay(data, heatmap, model.img_size))
        result["heatmap"] = str(out_path)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    print(f"Catégorie : {args.category}")
    print(f"Modèle    : {model_path}  (origine: {origin})")
    print(f"Score     : {result['score']:.0f}")
    if model.threshold is None:
        print("Seuil     : non calculé -> relancer : "
              f"python scripts/training.py --category {args.category} --eval")
        print("Verdict   : indéterminé")
    else:
        print(f"Seuil     : {model.threshold:.0f}")
        print(f"Verdict   : {'ANOMALIE' if anomaly else 'OK'}")
    if args.heatmap:
        print(f"Heatmap   : {result['heatmap']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
