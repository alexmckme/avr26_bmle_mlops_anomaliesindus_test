"""Intégration MLflow — suivi d'expériences (Phase 2).

Backend store : SQLite local (`mlflow.db`).
Artefact store : MinIO (S3), bucket `MLFLOW_ARTIFACT_BUCKET` (défaut : `mlflow`).

Le module est volontairement *optionnel* : si MLflow n'est pas installé ou si
l'appel échoue, l'entraînement continue (tracking simplement désactivé).

Variables d'environnement (voir .env) :
    MLFLOW_TRACKING_URI      (défaut : sqlite:///<racine>/mlflow.db)
    MLFLOW_EXPERIMENT        (défaut : anomalies-indus)
    MLFLOW_ARTIFACT_BUCKET   (défaut : mlflow)
"""

from __future__ import annotations

import os

from core.config import PROJECT_ROOT, load_settings

_ENABLED = False

# Paramètres / métriques récupérés depuis le `meta` renvoyé par core.padim.
_PARAM_KEYS = ("category", "source", "data_version", "fraction", "full",
               "img_size", "ridge", "n_features", "n_train", "grid",
               "feature_layers")
_METRIC_KEYS = ("auc", "n_test", "threshold", "elapsed_s")


def _default_tracking_uri() -> str:
    return f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"


def _param(value) -> str:
    """MLflow n'accepte que des paramètres scalaires (str/int/float/bool)."""
    if isinstance(value, (list, tuple)):
        return ",".join(str(v) for v in value)
    return str(value)


def setup_mlflow(enabled: bool = True) -> bool:
    """Configure MLflow (URI, MinIO pour les artefacts, expérience).

    Retourne True si le tracking est prêt, False sinon (dégradation gracieuse).
    """
    global _ENABLED
    _ENABLED = False
    if not enabled:
        return False

    try:
        import mlflow
    except ImportError:
        print("[WARN] mlflow non installé -> tracking désactivé (pip install 'mlflow[s3]')")
        return False

    settings = load_settings()
    bucket = os.getenv("MLFLOW_ARTIFACT_BUCKET", "mlflow")

    # Le client S3 de MLflow (boto3) doit pointer vers MinIO.
    os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", f"http://{settings.minio_endpoint}")
    os.environ.setdefault("AWS_ACCESS_KEY_ID", settings.minio_access_key)
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", settings.minio_secret_key)
    os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI") or _default_tracking_uri())

    # MLflow ne crée pas le bucket S3 : on s'en assure via le client MinIO.
    try:
        from core.storage import get_client

        client = get_client(settings)
        if not client.bucket_exists(bucket):
            client.make_bucket(bucket)
            print(f"[INFO] Bucket d'artefacts MLflow créé : {bucket}")
    except Exception as exc:  # noqa: BLE001 — MinIO indisponible : on continue
        print(f"[WARN] Bucket d'artefacts '{bucket}' indisponible : {exc}")

    experiment = os.getenv("MLFLOW_EXPERIMENT", "anomalies-indus")
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=f"s3://{bucket}")
    mlflow.set_experiment(experiment)

    _ENABLED = True
    return True


def is_enabled() -> bool:
    return _ENABLED


def log_training(meta: dict, artifact_path: str, run_name: str | None = None,
                 register_model: str | None = None, alias: str = "candidate") -> dict | None:
    """Enregistre un entraînement dans MLflow.

    - Toujours : params, métriques, tags (`dataset_sha256`) et `metadata.json`.
    - `register_model` fourni : logge le modèle en **pyfunc** (PaDiM) et
      l'enregistre dans le **Model Registry** sous ce nom, avec `alias`.
    - Sinon : logge simplement l'artefact `.npz` (mode tracking seul).

    Retourne un dict {run_id, registered_model, model_version} ou None si le
    tracking est désactivé.
    """
    if not _ENABLED:
        return None
    import mlflow

    params = {k: _param(meta[k]) for k in _PARAM_KEYS if k in meta}
    metrics = {k: float(meta[k]) for k in _METRIC_KEYS if meta.get(k) is not None}
    tags = {
        "dataset_sha256": str(meta.get("dataset_sha256", "")),
        "full_dataset": str(meta.get("full", "")),
    }

    with mlflow.start_run(run_name=run_name) as run:
        if params:
            mlflow.log_params(params)
        if metrics:
            mlflow.log_metrics(metrics)
        mlflow.set_tags(tags)
        mlflow.log_dict(meta, "metadata.json")

        model_version = None
        if register_model:
            model_version = _log_and_register_padim(
                artifact_path, register_model, alias
            )
        else:
            mlflow.log_artifact(str(artifact_path), artifact_path="model")

        return {
            "run_id": run.info.run_id,
            "registered_model": register_model,
            "model_version": model_version,
        }


def _log_and_register_padim(artifact_path, register_model: str, alias: str):
    """Logge le modèle PaDiM en pyfunc et l'enregistre (alias `alias`)."""
    import mlflow
    from core.padim_flavor import PadimPyfunc

    kwargs = dict(
        python_model=PadimPyfunc(),
        artifacts={"padim_model": str(artifact_path)},
        code_paths=[str(PROJECT_ROOT / "core")],   # modèle auto-porteur
        pip_requirements=["mlflow", "numpy", "pandas", "Pillow",
                          "tensorflow", "scikit-learn"],
        registered_model_name=register_model,
    )
    try:  # MLflow 3 : `name` ; versions antérieures : `artifact_path`
        model_info = mlflow.pyfunc.log_model(name="model", **kwargs)
    except TypeError:
        model_info = mlflow.pyfunc.log_model(artifact_path="model", **kwargs)

    version = getattr(model_info, "registered_model_version", None)
    if version is not None:
        mlflow.MlflowClient().set_registered_model_alias(register_model, alias, version)
    return version


def promote_if_better(registered_model: str, metric: str = "auc", alias: str = "champion",
                      candidate_alias: str = "candidate") -> dict:
    """Promeut `candidate` en `champion` si sa métrique est >= celle du champion.

    Réponse au besoin « charger la version précédente et comparer avec la nouvelle ».
    """
    import mlflow

    client = mlflow.MlflowClient()
    try:
        candidate = client.get_model_version_by_alias(registered_model, candidate_alias)
    except Exception:  # noqa: BLE001 — pas encore de version candidate
        return {"promoted": False, "reason": f"pas de version '{candidate_alias}'"}

    cand_score = client.get_run(candidate.run_id).data.metrics.get(metric)

    champion = None
    try:
        champion = client.get_model_version_by_alias(registered_model, alias)
    except Exception:  # noqa: BLE001 — pas encore de champion
        champion = None
    champ_score = client.get_run(champion.run_id).data.metrics.get(metric) if champion else None

    better = champion is None or (cand_score is not None
                                  and (champ_score is None or cand_score >= champ_score))
    if better:
        client.set_registered_model_alias(registered_model, alias, candidate.version)

    return {
        "promoted": better,
        "registered_model": registered_model,
        "metric": metric,
        "candidate_version": candidate.version,
        "candidate_score": cand_score,
        "previous_champion_version": champion.version if champion else None,
        "previous_champion_score": champ_score,
    }
