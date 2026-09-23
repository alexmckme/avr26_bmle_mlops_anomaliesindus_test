"""Intégration MLflow — suivi d'expériences et Model Registry.

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
import shutil
from pathlib import Path

from core import versioning
from core.config import PROJECT_ROOT, load_settings

_ENABLED = False

# Paramètres / métriques récupérés depuis le `meta` renvoyé par core.padim.
_PARAM_KEYS = ("category", "source", "data_version", "fraction", "full",
               "img_size", "ridge", "n_features", "n_train", "grid",
               "feature_layers",
               # provenance du code (voir core.versioning.git_info)
               "git_commit", "git_branch", "git_dirty")
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
                 register_model: str | None = None, alias: str = "candidate",
                 manifest: dict | None = None) -> dict | None:
    """Enregistre un entraînement dans MLflow.

    - Toujours : params (dont le **commit du code**), métriques, tags
      (`dataset_sha256`) et `metadata.json`.
    - `manifest` fourni (voir `core.versioning`) : logge `dataset_manifest.json`,
      c'est-à-dire la liste des images utilisées **pour ce run précis**.
    - `register_model` fourni : logge le modèle en **pyfunc** (PaDiM) et
      l'enregistre dans le **Model Registry** sous ce nom, avec `alias` ; la
      provenance (commit + empreinte du dataset) est aussi posée en tags de la
      version, et le dataset est référencé dans `datasets.json`.
    - Sinon : logge simplement l'artefact `.npz` (mode tracking seul).

    Retourne un dict {run_id, registered_model, model_version} ou None si le
    tracking est désactivé.
    """
    if not _ENABLED:
        return None
    import mlflow

    code = versioning.add_code_info(meta)
    params = {k: _param(meta[k]) for k in _PARAM_KEYS if k in meta}
    metrics = {k: float(meta[k]) for k in _METRIC_KEYS if meta.get(k) is not None}
    tags = {
        "dataset_sha256": str(meta.get("dataset_sha256", "")),
        "full_dataset": str(meta.get("full", "")),
        "git_commit": str(meta.get("git_commit", "")),
        "code_source": str(code.get("source", "")),
    }

    with mlflow.start_run(run_name=run_name) as run:
        if params:
            mlflow.log_params(params)
        if metrics:
            mlflow.log_metrics(metrics)
        mlflow.set_tags(tags)
        mlflow.log_dict(meta, "metadata.json")
        if manifest:
            # Traçabilité « données » : les images exactes du run
            mlflow.log_dict(manifest, versioning.MANIFEST_ARTIFACT)

        model_version = None
        if register_model:
            model_version = _log_and_register_padim(
                artifact_path, register_model, alias
            )
            _tag_model_version(register_model, model_version, meta)
        else:
            mlflow.log_artifact(str(artifact_path), artifact_path="model")

        if manifest and register_model:
            # Index `datasets.json` : seulement pour les runs qui enregistrent une
            # version de modèle. Sinon les runs planifiés (un par minute) le
            # réécriraient en boucle sans rien apporter.
            versioning.update_dataset_index(manifest, run_id=run.info.run_id)

        return {
            "run_id": run.info.run_id,
            "registered_model": register_model,
            "model_version": model_version,
        }


def describe_alias(registered_model: str, alias: str = "champion") -> dict | None:
    """Métadonnées d'une version du Registry (sans télécharger l'artefact).

    Retourne {alias, version, run_id, status, auc, threshold, n_train,
    data_version, git_commit, dataset_sha256} ou None si l'alias n'existe pas.
    """
    if not _ENABLED:
        return None
    import mlflow

    client = mlflow.MlflowClient()
    try:
        version = client.get_model_version_by_alias(registered_model, alias)
    except Exception:  # noqa: BLE001 — alias inexistant
        return None

    run = client.get_run(version.run_id)
    info = {
        "alias": alias,
        "version": version.version,
        "run_id": version.run_id,
        "status": version.status,
        "auc": run.data.metrics.get("auc"),
        "threshold": run.data.metrics.get("threshold"),
        "n_train": run.data.params.get("n_train"),
        "data_version": run.data.params.get("data_version"),
    }
    for key in ("git_commit", "dataset_sha256"):
        info[key] = version.tags.get(key) or run.data.tags.get(key)
    return info


def list_champions(prefix: str = "padim-") -> list[dict]:
    """Une entrée par catégorie : champion et candidate (lecture seule)."""
    if not _ENABLED:
        return []
    import mlflow

    client = mlflow.MlflowClient()
    entries = []
    for registered in client.search_registered_models():
        if not registered.name.startswith(prefix):
            continue
        entries.append({
            "category": registered.name[len(prefix):],
            "registered_model": registered.name,
            "champion": describe_alias(registered.name, "champion"),
            "candidate": describe_alias(registered.name, "candidate"),
        })
    return sorted(entries, key=lambda e: e["category"])


def recent_runs(limit: int = 20, category: str | None = None) -> list[dict]:
    """Derniers runs de l'expérience (métriques + provenance)."""
    if not _ENABLED:
        return []
    import mlflow

    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(os.getenv("MLFLOW_EXPERIMENT", "anomalies-indus"))
    if experiment is None:
        return []

    where = f"params.category = '{category}'" if category else ""
    runs = client.search_runs([experiment.experiment_id], filter_string=where,
                              order_by=["start_time DESC"], max_results=limit)
    return [{
        "run_id": run.info.run_id,
        "name": run.data.tags.get("mlflow.runName", ""),
        "status": run.info.status,
        "start_time": run.info.start_time,
        "auc": run.data.metrics.get("auc"),
        "data_version": run.data.params.get("data_version"),
        "n_train": run.data.params.get("n_train"),
        "git_commit": run.data.params.get("git_commit") or run.data.tags.get("git_commit"),
        "dataset_sha256": run.data.tags.get("dataset_sha256"),
    } for run in runs]


def _tag_model_version(registered_model: str, version, meta: dict) -> None:
    """Pose la provenance (code + dataset) sur une version du Registry.

    C'est ce qui permet de remonter du **champion servi** jusqu'au code et aux
    images qui l'ont produit, directement depuis l'UI MLflow.
    """
    if version is None:
        return
    import mlflow

    client = mlflow.MlflowClient()
    for key in ("git_commit", "dataset_sha256", "data_version", "n_train"):
        value = meta.get(key)
        if value is None:
            continue
        try:
            client.set_model_version_tag(registered_model, str(version), key,
                                         _param(value))
        except Exception as exc:  # noqa: BLE001 — tag informatif, jamais bloquant
            print(f"[WARN] tag '{key}' non posé sur {registered_model} v{version} : {exc}")


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

    À métrique égale, la version la plus récente devient championne. Refuse la
    promotion si la métrique est absente (entraînement sans évaluation).
    Réponse au besoin « charger la version précédente et comparer avec la nouvelle ».
    """
    import mlflow

    client = mlflow.MlflowClient()
    try:
        candidate = client.get_model_version_by_alias(registered_model, candidate_alias)
    except Exception:  # noqa: BLE001 — pas encore de version candidate
        return {"promoted": False, "reason": f"pas de version '{candidate_alias}'"}

    cand_score = client.get_run(candidate.run_id).data.metrics.get(metric)
    if cand_score is None:
        return {
            "promoted": False,
            "registered_model": registered_model,
            "metric": metric,
            "candidate_version": candidate.version,
            "reason": f"métrique '{metric}' absente sur la version candidate "
                      f"(relancer l'entraînement avec l'évaluation)",
        }

    champion = None
    try:
        champion = client.get_model_version_by_alias(registered_model, alias)
    except Exception:  # noqa: BLE001 — pas encore de champion
        champion = None
    champ_score = client.get_run(champion.run_id).data.metrics.get(metric) if champion else None

    # À métrique égale, la version la plus récente devient championne (>=)
    better = champion is None or champ_score is None or cand_score >= champ_score
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


# ─────────────────────────────────────────────────────────────
# Inférence : servir le champion du Registry (avec cache local)
# ─────────────────────────────────────────────────────────────
def fetch_champion(registered_model: str, dest_dir="models", alias: str = "champion",
                   category: str | None = None, force: bool = False) -> Path | None:
    """Télécharge (et met en cache) l'artefact `.npz` du champion du Registry.

    Le cache est un fichier `models/<catégorie>.champion.npz` accompagné d'un
    fichier `.version` : si la version en cache correspond au champion courant,
    aucun téléchargement n'est refait. Retourne le chemin du `.npz`, ou None
    si indisponible (pas de champion, MLflow/MinIO injoignable…).
    """
    if not _ENABLED:
        return None
    try:
        import mlflow

        client = mlflow.MlflowClient()
        mv = client.get_model_version_by_alias(registered_model, alias)

        base = category or registered_model
        dest = Path(dest_dir) / f"{base}.{alias}.npz"
        stamp = dest.with_suffix(dest.suffix + ".version")
        if not force and dest.exists() and stamp.exists() \
                and stamp.read_text().strip() == str(mv.version):
            return dest

        # MLflow 3 : les modèles loggés vivent hors des artefacts du run ;
        # l'URI `models:/<nom>@<alias>` télécharge le dossier modèle complet.
        local_dir = mlflow.artifacts.download_artifacts(
            artifact_uri=f"models:/{registered_model}@{alias}"
        )
        npz = next(Path(local_dir).rglob("*.npz"), None)
        if npz is None:
            print(f"[WARN] Aucun .npz dans l'artefact du champion '{registered_model}'")
            return None

        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(npz, dest)
        stamp.write_text(str(mv.version))
        print(f"[INFO] Champion '{registered_model}'@v{mv.version} téléchargé -> {dest}")
        return dest
    except Exception as exc:  # noqa: BLE001 — Registry/MinIO indisponible
        print(f"[WARN] Champion '{registered_model}@{alias}' indisponible : {exc}")
        return None


def resolve_inference_model(category: str, model_dir="models",
                            use_registry: bool = True) -> tuple[Path, str]:
    """Modèle à servir pour une catégorie.

    Ordre de préférence :
      1. **champion MLflow** (téléchargé/caché dans `models/<cat>.champion.npz`) ;
      2. **fichiers locaux** `models/<cat>.npz` ou versionnés (repli).

    Retourne (chemin, origine) avec origine ∈ {'registry', 'local'}.
    """
    import core.padim as padim

    if use_registry:
        setup_mlflow()
        champion = fetch_champion(f"padim-{category}", dest_dir=model_dir, category=category)
        if champion is not None:
            return champion, "registry"
    return padim.resolve_model_path(category, model_dir), "local"
