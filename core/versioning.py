"""Versioning « code + données » — traçabilité des runs MLflow.

Deux questions auxquelles on doit pouvoir répondre des mois plus tard :

  1. « quel **code** a produit ce modèle ? »               -> le commit git
  2. « quelles **images** ont servi à cet entraînement ? » -> le manifeste du dataset

On ne versionne pas les images elles-mêmes (pas de DVC ici) : elles restent la
source de vérité dans MinIO. On versionne la *recette* et le *contenu exact* :

  - `dataset_manifest.json` — artefact **par run** (rangé par MLflow dans MinIO) :
    la liste de chaque image utilisée (clé MinIO, taille, etag), le mode de
    sélection et le commit du code. C'est la correspondance **run -> images**.
  - `datasets.json` — **index dans le repo** (committé) : catalogue des jeux de
    données connus, repérés par leur empreinte `dataset_sha256`, avec le dernier
    run qui les a utilisés. C'est la correspondance **empreinte -> runs**.

Chaîne complète de traçabilité :

    champion (Registry) -> run MLflow -> git_commit + dataset_sha256
                                       -> manifeste (images) + datasets.json

Dégradation gracieuse : si l'index ne peut pas être écrit (disque en lecture
seule, pas de dépôt git…), on avertit et l'entraînement continue.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import core.padim as padim
from core.config import PROJECT_ROOT

# Index des jeux de données (committé : c'est la mémoire du versioning data)
DATASETS_INDEX = PROJECT_ROOT / "datasets.json"
SCHEMA_VERSION = 1
MANIFEST_ARTIFACT = "dataset_manifest.json"  # nom de l'artefact MLflow
_COMMIT_LEN = 12  # hash raccourci : lisible dans l'UI MLflow, non ambigu


# ─────────────────────────────────────────────────────────────
# 1) Commit du code
# ─────────────────────────────────────────────────────────────
def _run_git(*args: str) -> str | None:
    """Exécute `git <args>` dans la racine du projet (None si indisponible)."""
    try:
        proc = subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, capture_output=True,
            text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def _read_dot_git() -> tuple[str | None, str | None]:
    """Lit `.git` sans le binaire git (utile dans les images python-slim).

    Retourne (commit, branche). Le conteneur monte `./.git` en lecture seule.
    """
    git_dir = PROJECT_ROOT / ".git"
    if not git_dir.is_dir():
        return None, None
    try:
        head = (git_dir / "HEAD").read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None, None

    if not head.startswith("ref:"):
        return head or None, None          # HEAD détaché : le hash est écrit en clair

    ref = head.split(":", 1)[1].strip()
    branch = ref.rsplit("/", 1)[-1]
    ref_file = git_dir / ref
    if ref_file.is_file():
        return ref_file.read_text(encoding="utf-8", errors="replace").strip(), branch

    # Référence empaquetée (après gc / clone avec --depth)
    packed = git_dir / "packed-refs"
    if packed.is_file():
        for line in packed.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith(("#", "^")):
                continue
            parts = line.split(None, 1)
            if len(parts) == 2 and parts[1].strip() == ref:
                return parts[0], branch
    return None, branch


def git_info() -> dict:
    """Informations sur l'état du code, best effort (jamais bloquant).

    Ordre de résolution : variable d'environnement `GIT_COMMIT` (stamp de build),
    puis le binaire `git`, puis une lecture directe de `.git/` (cas du conteneur).

    Retourne {commit, branch, dirty, source} ; les valeurs peuvent être None.
    """
    env_commit = os.getenv("GIT_COMMIT") or os.getenv("GIT_SHA")
    if env_commit:
        return {"commit": env_commit.strip()[:_COMMIT_LEN],
                "branch": os.getenv("GIT_BRANCH") or None,
                "dirty": None, "source": "env"}

    commit = _run_git("rev-parse", f"--short={_COMMIT_LEN}", "HEAD")
    if commit:
        status = _run_git("status", "--porcelain", "--untracked-files=no")
        return {"commit": commit,
                "branch": _run_git("rev-parse", "--abbrev-ref", "HEAD"),
                "dirty": bool(status) if status is not None else None,
                "source": "git-cli"}

    commit, branch = _read_dot_git()
    return {"commit": commit[:_COMMIT_LEN] if commit else None,
            "branch": branch, "dirty": None,
            "source": "dot-git" if commit else "inconnu"}


def add_code_info(meta: dict) -> dict:
    """Ajoute le commit du code dans `meta` (idempotent).

    `meta` étant loggé comme params MLflow, imprimé par les scripts et renvoyé par
    `/training`, le commit apparaît partout sans dépendre de MLflow.
    """
    code = git_info()
    if code.get("commit"):
        meta.setdefault("git_commit", code["commit"])
    if code.get("branch"):
        meta.setdefault("git_branch", code["branch"])
    if code.get("dirty") is not None:
        meta.setdefault("git_dirty", code["dirty"])
    return code


# ─────────────────────────────────────────────────────────────
# 2) Manifeste du dataset (images réellement utilisées)
# ─────────────────────────────────────────────────────────────
def list_minio_objects(client, bucket: str, prefix: str) -> list[dict]:
    """Listing enrichi (id, taille, etag) des objets d'un prefix MinIO.

    Volontairement séparé de `core.padim.list_minio_entries` : l'empreinte
    `dataset_sha256` (clé + taille) doit rester stable dans le temps, on n'y
    touche pas. Ici on veut en plus l'`etag` (MD5 du contenu, upload monotère),
    sans requête supplémentaire : il vient du listing lui-même.
    """
    objects = []
    for obj in client.list_objects(bucket, prefix=prefix, recursive=True):
        if obj.object_name.endswith("/"):
            continue
        objects.append({
            "id": obj.object_name,
            "size": int(obj.size or 0),
            "etag": (getattr(obj, "etag", None) or "").strip('"'),
        })
    return sorted(objects, key=lambda o: o["id"])


def _select(objects: list[dict], meta: dict) -> list[dict]:
    """Réapplique la **même** sélection déterministe que l'entraînement."""
    pairs = [(o["id"], o["size"]) for o in objects]
    if meta.get("data_version") is not None:
        subset, _ = padim.subset_entries(pairs, data_version=meta["data_version"])
    else:
        subset, _ = padim.subset_entries(pairs, fraction=meta.get("fraction"))
    by_id = {o["id"]: o for o in objects}
    return [by_id[key] for key, _ in subset]


def build_manifest(objects: list[dict], *, category: str, source: str, meta: dict) -> dict:
    """Construit le manifeste : quelles images, sélectionnées comment, par quel code.

    `fingerprint_matches` vérifie que l'empreinte recalculée ici est bien celle de
    l'entraînement ; False signale donc que le dataset a bougé entre les deux
    (ajout d'images en parallèle) — information précieuse, jamais masquée.
    """
    recomputed = padim.fingerprint([(o["id"], o["size"]) for o in objects])
    return {
        "schema": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "category": category,
        "source": source,
        "split": "train",
        "code": git_info(),
        "selection": {
            "data_version": meta.get("data_version"),
            "fraction": meta.get("fraction"),
            "n_images": len(objects),
            "total_available": meta.get("total_available"),
            "full": meta.get("full"),
            "fractions_grid": list(padim.DATA_VERSION_FRACTIONS),
        },
        "dataset_sha256": recomputed,
        "dataset_sha256_from_training": meta.get("dataset_sha256"),
        "fingerprint_matches": recomputed == meta.get("dataset_sha256"),
        "images": objects,
    }


def minio_manifest(client, bucket: str, category: str, meta: dict) -> dict | None:
    """Manifeste d'un entraînement `--source minio` (None si le listing échoue)."""
    prefix = f"raw/{category}/train/good/"
    try:
        objects = list_minio_objects(client, bucket, prefix)
        return build_manifest(_select(objects, meta), category=category,
                              source=f"s3://{bucket}/{prefix}", meta=meta)
    except Exception as exc:  # noqa: BLE001 — jamais bloquant : l'entraînement est fait
        print(f"[WARN] manifeste du dataset non construit ({prefix}) : {exc}")
        return None


def dir_manifest(root, category: str, meta: dict) -> dict | None:
    """Manifeste d'un entraînement `--source local` (images du disque)."""
    root = Path(root)
    prefix_dir = root / category / "train" / "good"
    try:
        entries = padim.list_dir_entries(prefix_dir)
        objects = [{"id": str(Path(name).relative_to(root)),
                    "size": int(size), "etag": ""} for name, size in entries]
        return build_manifest(_select(objects, meta), category=category,
                              source=f"file://{prefix_dir}", meta=meta)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] manifeste local non construit ({prefix_dir}) : {exc}")
        return None


# ─────────────────────────────────────────────────────────────
# 3) Index des datasets (repo) : empreinte -> quoi + quel run
# ─────────────────────────────────────────────────────────────
def _load_index(path: Path) -> dict:
    if path.is_file() and path.stat().st_size > 0:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            print(f"[WARN] {path.name} illisible : il sera reconstruit")
    return {"schema": SCHEMA_VERSION, "datasets": {}}


def read_dataset_index(path: Path | None = None) -> dict:
    """Lit l'index des datasets ({} si le fichier est absent/corrompu)."""
    return _load_index(Path(path) if path else DATASETS_INDEX)


def _write_index(path: Path, index: dict) -> None:
    """Écrit l'index : atomique si possible, en place sinon.

    Dans le conteneur, `datasets.json` est un **bind mount** : `os.replace()` vers
    un point de montage échoue avec `EBUSY` (« Device or resource busy »). On
    retombe alors sur une écriture en place — acceptable ici car l'index est un
    fichier *dérivé* : s'il est corrompu, il est reconstruit au run suivant et la
    source de vérité (les runs MLflow et leurs manifestes) reste intacte.
    """
    payload = json.dumps(index, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, path)
        return
    except OSError:
        tmp.unlink(missing_ok=True)
    path.write_text(payload, encoding="utf-8")


def update_dataset_index(manifest: dict, run_id: str | None = None,
                         path: Path | None = None) -> dict | None:
    """Enregistre le jeu de données dans `datasets.json` (best effort).

    L'entrée est indexée par `dataset_sha256` : plusieurs runs utilisant le même
    jeu de données partagent donc la même entrée (seul le dernier run et le
    compteur évoluent).
    """
    path = Path(path) if path else DATASETS_INDEX
    sha = manifest.get("dataset_sha256")
    if not sha:
        return None

    selection = manifest.get("selection", {})
    now = manifest.get("generated_at")
    index = _load_index(path)
    datasets = index.setdefault("datasets", {})

    entry = datasets.setdefault(sha, {"first_seen": now})
    entry.update({
        "category": manifest.get("category"),
        "source": manifest.get("source"),
        "n_images": selection.get("n_images"),
        "total_available": selection.get("total_available"),
        "full": selection.get("full"),
        "selection": {"data_version": selection.get("data_version"),
                      "fraction": selection.get("fraction")},
        "last_seen": now,
        "manifest": f"{MANIFEST_ARTIFACT} (artefact du run)",
    })
    if run_id:
        runs = entry.setdefault("runs", {"count": 0, "last_run_id": None})
        runs["count"] = int(runs.get("count", 0)) + 1
        runs["last_run_id"] = run_id

    index["schema"] = SCHEMA_VERSION
    index["updated_at"] = now
    index.setdefault("_note", (
        "Index des jeux de données utilisés par un entraînement enregistré. "
        "Clé = dataset_sha256 (empreinte des images : clés + tailles). "
        "La liste exacte des images de chaque run est dans l'artefact "
        f"'{MANIFEST_ARTIFACT}' de ce run (MLflow/MinIO). Fichier généré : "
        "le committer fait partie du versioning des données."
    ))

    try:
        _write_index(path, index)
    except OSError as exc:
        print(f"[WARN] index des datasets non écrit ({path}) : {exc}")
        return None
    return entry
