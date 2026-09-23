"""PaDiM — modélisation gaussienne multi-échelle pour la détection d'anomalies.

Portage « production » du notebook `05_PaDiM_128.ipynb` vers le pipeline MLOps.

Points clés :
  - Pas de véritable entraînement : extraction de features EfficientNetB0
    (poids ImageNet, réseau gelé) + ajustement d'une gaussienne par position
    spatiale sur les features des images « good » (approche non supervisée).
  - Le preprocessing (resize au décodage, normalisation) est centralisé ici :
    `/training` et `/predict` utiliseront EXACTEMENT les mêmes fonctions
    (anti train/serve skew).
  - Les images originales restent la source de vérité (MinIO) ; le resize
    (IMG_SIZE) est appliqué au moment du décodage pour limiter la RAM.

Le module n'importe TensorFlow qu'à l'exécution des fonctions qui en ont
besoin (extraction de features) : fit/score/save/load restent testables
sans TF.
"""

from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

# ─── Paramètres par défaut (identiques au notebook validé) ───
IMG_SIZE = 128
FEATURE_LAYERS = [
    "block2a_activation",  # début  : (32, 32, 16)
    "block4a_activation",  # milieu : (16, 16, 112)
    "block6a_activation",  # fin    : (8, 8, 192)
]
RIDGE = 1e-6          # régularisation ajoutée à la diagonale des covariances
BATCH_SIZE = 32

# Grille de croissance simulée du dataset (fractions du corpus « good » dispo).
# data_version = i -> on garde les premiers round(frac_i * N) images (ordre de
# tri déterministe) : sous-ensembles imbriqués simulant l'accumulation de
# données dans le temps (v0 ⊂ v1 ⊂ … ⊂ full). Pas de 10 % : v0 = 10 %, v1 = 20 %,
# … v9 = 100 % (= 'full'), et cela quelle que soit la taille N de la catégorie.
DATA_VERSION_FRACTIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
_IMAGE_EXT = {".png", ".jpg", ".jpeg"}
_extractor_cache: dict = {}


# ─────────────────────────────────────────────────────────────
# Chargement d'images (resize au décodage — RAM maîtrisée)
# ─────────────────────────────────────────────────────────────
def _bytes_to_array(data: bytes, img_size: int = IMG_SIZE) -> np.ndarray:
    """Décode des bytes image et redimensionne immédiatement -> uint8 (0-255)."""
    with Image.open(io.BytesIO(data)) as im:
        im = im.convert("RGB")
        if im.size != (img_size, img_size):
            im = im.resize((img_size, img_size), Image.LANCZOS)
        return np.asarray(im, dtype=np.uint8)


def list_dir_entries(path) -> list:
    """Liste triée (déterministe) des images d'un dossier local."""
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"Dossier introuvable : {path}")
    return sorted(
        (str(f), f.stat().st_size)
        for f in path.rglob("*")
        if f.is_file() and f.suffix.lower() in _IMAGE_EXT
    )


def load_images_from_dir(path, img_size: int = IMG_SIZE, entries=None):
    """Charge les images d'un dossier local (mode test sans MinIO).

    Retourne (images np.uint8 (N, img, img, 3), entries [(nom, taille)]).
    """
    path = Path(path)
    if entries is None:
        entries = list_dir_entries(path)
    images = [_bytes_to_array(Path(name).read_bytes(), img_size) for name, _ in entries]
    if not images:
        raise FileNotFoundError(f"Aucune image trouvée dans : {path}")
    return np.stack(images), entries


def list_minio_entries(client, bucket: str, prefix: str) -> list:
    """Liste triée (déterministe) des objets images d'un prefix MinIO."""
    entries = []
    for obj in client.list_objects(bucket, prefix=prefix, recursive=True):
        if not obj.object_name.endswith("/"):
            entries.append((obj.object_name, obj.size))
    return sorted(entries, key=lambda e: e[0])


def load_images_from_minio(client, bucket: str, prefix: str, img_size: int = IMG_SIZE,
                           entries=None):
    """Charge les images d'un prefix MinIO (clé objet = source de vérité).

    Retourne (images np.uint8 (N, img, img, 3), entries [(clé, taille)]).
    """
    if entries is None:
        entries = list_minio_entries(client, bucket, prefix)
    images = []
    for name, _ in entries:
        data = client.get_object(bucket, name).read()
        images.append(_bytes_to_array(data, img_size))
    if not images:
        raise FileNotFoundError(f"Aucune image sous s3://{bucket}/{prefix}")
    return np.stack(images), entries


# ─────────────────────────────────────────────────────────────
# Sous-échantillonnage déterministe (« croissance accumulée »)
# ─────────────────────────────────────────────────────────────
def subset_entries(entries, data_version=None, fraction=None,
                   fractions=None) -> tuple:
    """Sélectionne un sous-ensemble « accumulé » (imbriqué) des entries.

    Sémantique : simule un dataset qui grandit au fil du temps. Les images
    sont triées par clé (ordre déterministe) ; `data_version` = index
    (0..K-1, ou 'full') dans la grille de croissance => on garde les
    PREMIERS n_images, donc chaque version est un sur-ensemble des
    précédentes. `fraction` (0..1]) est une alternative directe qui
    court-circuite la grille. Par défaut : dataset complet (100 %).

    Retourne (subset_entries, meta) avec meta = {data_version, fraction,
    n_images, total_available, full}.
    """
    entries = sorted(entries, key=lambda e: e[0])
    total = len(entries)
    grid = list(fractions or DATA_VERSION_FRACTIONS)

    if fraction is not None:
        frac = max(0.0, min(1.0, float(fraction)))
        version = None
    else:
        if data_version is None:
            version = len(grid) - 1                     # défaut : 100 %
        elif isinstance(data_version, str):
            version = len(grid) - 1 if data_version == "full" else int(data_version)
        else:
            version = int(data_version)
        version = max(0, min(version, len(grid) - 1))
        frac = float(grid[version])

    # Comptes croissants & imbriqués (évite 2 versions identiques sur petits N)
    counts, taken = [], 0
    for g in grid:
        base = taken + 1 if taken < total else total
        target = max(1, int(round(g * total)))
        taken = min(total, max(base, target))
        counts.append(taken)

    if version is not None:
        n_take = counts[version]
        frac_used = round(n_take / total, 4) if total else 1.0
    else:
        n_take = max(1, int(round(frac * total)))
        frac_used = round(n_take / total, 4) if total else 1.0

    subset = entries[:n_take]
    meta = {
        "data_version": version,       # None si piloté par `fraction`
        "fraction": frac_used,
        "n_images": len(subset),
        "total_available": total,
        "full": len(subset) == total,
    }
    return subset, meta


def fingerprint(entries) -> str:
    """Empreinte SHA-256 du jeu d'images (clés + tailles) -> traçabilité MLflow."""
    h = hashlib.sha256()
    for name, size in sorted(entries):
        h.update(f"{name}:{size}\n".encode())
    return h.hexdigest()


def data_versions(fractions=None) -> list:
    """Décrit la grille de versions de données (pour l'API, la CLI et l'interface).

    Retourne [{index, fraction, percent, label}] ; `index` est la valeur à passer
    à `data_version` (le dernier vaut 100 %, donc équivaut à 'full'). Les libellés
    sont calculés ici pour que la CLI, l'API et l'UI affichent la même chose.
    """
    versions = []
    for index, fraction in enumerate(fractions or DATA_VERSION_FRACTIONS):
        percent = int(round(fraction * 100))
        suffix = " (full)" if percent >= 100 else ""
        versions.append({
            "index": index,
            "fraction": fraction,
            "percent": percent,
            "label": f"v{index} — {percent} %{suffix}",
        })
    return versions


def artifact_name(category: str, meta: dict) -> str:
    """Nom du fichier d'artefact local, avec le **pourcentage explicite**.

    `<cat>.npz` pour 100 %, sinon `<cat>.p<percent>.npz` (ex. `bottle.p40.npz`).
    Le nom dépend de la fraction réellement utilisée et non de l'index de la
    grille : changer la grille ne « renumérote » donc pas les modèles existants.
    """
    percent = int(round(float(meta.get("fraction") or 1.0) * 100))
    return f"{category}.npz" if percent >= 100 else f"{category}.p{percent}.npz"


# ─────────────────────────────────────────────────────────────
# Extraction de features multi-échelles (TensorFlow, lazy)
# ─────────────────────────────────────────────────────────────
def _get_extractor(img_size: int = IMG_SIZE, layers=None):
    """Construit (une seule fois) l'extracteur EfficientNetB0 multi-échelles."""
    import tensorflow as tf
    from tensorflow import keras

    layers = tuple(layers or FEATURE_LAYERS)
    key = (img_size, layers)
    if key in _extractor_cache:
        return _extractor_cache[key]

    base = keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet",
        input_shape=(img_size, img_size, 3),
    )
    base.trainable = False
    outputs = [base.get_layer(name).output for name in layers]
    model = keras.Model(inputs=base.input, outputs=outputs, name="feature_extractor")
    _extractor_cache[key] = model
    return model


def extract_features(images, img_size: int = IMG_SIZE, layers=None,
                     batch_size: int = BATCH_SIZE):
    """Images np.uint8 (N, img_size, img_size, 3) -> features multi-échelles.

    Toutes les échelles sont redimensionnées sur la grille de la couche
    centrale puis concaténées sur l'axe des canaux.
    Retourne (features float32 (N, H, W, C), grid (H, W)).
    """
    import tensorflow as tf

    extractor = _get_extractor(img_size, layers)
    x = tf.keras.applications.efficientnet.preprocess_input(images.astype(np.float32))
    outputs = extractor.predict(x, batch_size=batch_size, verbose=0)

    mid = outputs[len(outputs) // 2]
    grid = (mid.shape[1], mid.shape[2])
    feats = []
    for out in outputs:
        if out.shape[1] != grid[0] or out.shape[2] != grid[1]:
            out = tf.image.resize(out, grid).numpy()
        feats.append(np.asarray(out))
    return np.concatenate(feats, axis=-1).astype(np.float32), grid


# ─────────────────────────────────────────────────────────────
# Modèle PaDiM : fit / score / save / load
# ─────────────────────────────────────────────────────────────
@dataclass
class PadimModel:
    mean: np.ndarray                # (H, W, C) float32
    cov_inv: np.ndarray             # (H, W, C, C) float32
    grid: tuple = (0, 0)
    n_features: int = 0
    n_train: int = 0
    img_size: int = IMG_SIZE
    ridge: float = RIDGE
    feature_layers: tuple = field(default_factory=lambda: tuple(FEATURE_LAYERS))
    category: str = ""
    threshold: float | None = None   # seuil Youden (optionnel, calculé sur le test)

    def metadata(self) -> dict:
        return {
            "category": self.category,
            "grid": list(self.grid),
            "n_features": self.n_features,
            "n_train": self.n_train,
            "img_size": self.img_size,
            "ridge": self.ridge,
            "feature_layers": list(self.feature_layers),
            "threshold": self.threshold,
        }


def fit(features: np.ndarray, ridge: float = RIDGE, category: str = "") -> PadimModel:
    """Ajuste une gaussienne multivariée par position spatiale.

    features : (N, H, W, C). Covariance régularisée puis inversée par position.
    """
    n, h, w, c = features.shape
    mean = features.mean(axis=0).astype(np.float32)          # (H, W, C)
    centered = (features - mean).astype(np.float64)          # stabilité numérique
    cov = np.einsum("nhwc,nhwd->hwcd", centered, centered, optimize=True) / max(n - 1, 1)
    eye = np.eye(c, dtype=np.float64)
    cov_inv = np.linalg.inv(cov + eye[None, None, :, :] * ridge)
    return PadimModel(
        mean=mean,
        cov_inv=cov_inv.astype(np.float32),
        grid=(h, w),
        n_features=c,
        n_train=n,
        ridge=ridge,
        category=category,
    )


def compute_anomaly(features: np.ndarray, model: PadimModel) -> np.ndarray:
    """Distance de Mahalanobis² par position. Retourne (N, H, W)."""
    diff = features - model.mean                              # (N, H, W, C)
    return np.einsum("nhwc,hwcd,nhwd->nhw", diff, model.cov_inv, diff, optimize=True)


def global_score(anomaly_map: np.ndarray) -> np.ndarray:
    """Score global = max des distances sur la grille (standard PaDiM)."""
    return anomaly_map.reshape(anomaly_map.shape[0], -1).max(axis=1)


def save_model(model: PadimModel, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        mean=model.mean,
        cov_inv=model.cov_inv,
        grid=np.array(model.grid, dtype=np.int64),
        n_features=np.array(model.n_features),
        n_train=np.array(model.n_train),
        img_size=np.array(model.img_size),
        ridge=np.array(model.ridge),
        category=np.array(model.category),
        feature_layers=np.array(list(model.feature_layers)),
        threshold=np.array(model.threshold if model.threshold is not None else np.nan),
    )
    return str(path)


def load_model(path) -> PadimModel:
    with np.load(path) as z:
        thr = float(z["threshold"]) if "threshold" in z.files else np.nan
        return PadimModel(
            mean=z["mean"],
            cov_inv=z["cov_inv"],
            grid=tuple(int(v) for v in z["grid"]),
            n_features=int(z["n_features"]),
            n_train=int(z["n_train"]),
            img_size=int(z["img_size"]),
            ridge=float(z["ridge"]),
            category=str(z["category"]),
            feature_layers=tuple(str(s) for s in z["feature_layers"]),
            threshold=(None if np.isnan(thr) else thr),
        )


def _percent_in_name(path) -> int:
    """Extrait le pourcentage d'un nom d'artefact (`bottle.p40.npz` -> 40)."""
    match = re.search(r"\.p(\d+)\.npz$", Path(path).name)
    return int(match.group(1)) if match else 0


def resolve_model_path(category: str, model_dir="models") -> Path:
    """Résout l'artefact local d'une catégorie : `<cat>.npz` (100 %) si présent,
    sinon la version la plus complète disponible (`<cat>.p<percent>.npz`).

    Lève FileNotFoundError si aucun modèle n'existe pour cette catégorie.
    """
    model_dir = Path(model_dir)
    full = model_dir / f"{category}.npz"
    if full.exists():
        return full
    candidates = list(model_dir.glob(f"{category}.*.npz"))
    if not candidates:
        raise FileNotFoundError(
            f"Aucun modèle entraîné pour '{category}' dans {model_dir}/ — lancer : "
            f"python scripts/training.py --category {category} --eval"
        )
    # Le pourcentage est explicite dans le nom : on prend le plus élevé (un tri
    # alphabétique donnerait `p100` avant `p40`, donc le plus petit).
    return max(candidates, key=_percent_in_name)


# ─────────────────────────────────────────────────────────────
# Orchestration (utilisée par scripts/training.py et l'API)
# ─────────────────────────────────────────────────────────────
def fit_from_images(images, category: str = "", img_size: int = IMG_SIZE) -> tuple:
    """Images déjà chargées -> (modèle PaDiM, features extraites)."""
    features, _ = extract_features(images, img_size=img_size)
    model = fit(features, category=category)
    model.img_size = img_size
    return model, features


def fit_from_minio(client, bucket: str, category: str, split: str = "train",
                   label: str = "good", prefix_root: str = "raw",
                   img_size: int = IMG_SIZE, data_version=None,
                   fraction=None) -> tuple:
    """Entraîne un modèle PaDiM pour une catégorie depuis MinIO.

    Lit s3://bucket/<prefix_root>/<category>/<split>/<label>/. Optionnel :
    sous-ensemble déterministe « croissance accumulée » via data_version
    (index 0..K-1 / 'full') ou fraction (0..1].
    Retourne (modèle, metadata).
    """
    prefix = f"{prefix_root}/{category}/{split}/{label}/"
    all_entries = list_minio_entries(client, bucket, prefix)
    entries, sub_meta = subset_entries(all_entries, data_version=data_version,
                                       fraction=fraction)
    images, _ = load_images_from_minio(client, bucket, prefix, img_size=img_size,
                                       entries=entries)
    model, _ = fit_from_images(images, category=category, img_size=img_size)
    metadata = {
        "category": category,
        "source": f"s3://{bucket}/{prefix}",
        "dataset_sha256": fingerprint(entries),
        **sub_meta,
        **model.metadata(),
    }
    return model, metadata


def fit_from_dir(path, category: str = "", img_size: int = IMG_SIZE,
                 data_version=None, fraction=None) -> tuple:
    """Variante locale (test sans MinIO) : dossier .../<category>/train/good."""
    path = Path(path)
    all_entries = list_dir_entries(path)
    entries, sub_meta = subset_entries(all_entries, data_version=data_version,
                                       fraction=fraction)
    images, _ = load_images_from_dir(path, img_size=img_size, entries=entries)
    model, _ = fit_from_images(images, category=category or path.parent.name,
                               img_size=img_size)
    metadata = {
        "category": category or path.parent.name,
        "source": f"local:{path}",
        "dataset_sha256": fingerprint(entries),
        **sub_meta,
        **model.metadata(),
    }
    return model, metadata


def predict_image(image_data: bytes, model: PadimModel) -> tuple:
    """Prédiction d'une image (bytes) -> (score global, heatmap (H, W))."""
    img = _bytes_to_array(image_data, img_size=model.img_size)
    features, _ = extract_features(
        img[None], img_size=model.img_size, layers=model.feature_layers
    )
    anomaly = compute_anomaly(features, model)[0]
    return float(anomaly.max()), anomaly


def _colorize(norm: np.ndarray) -> np.ndarray:
    """Colormap bleu -> vert -> jaune -> rouge (numpy seul, sans matplotlib)."""
    stops = np.array([[0.10, 0.30, 0.90],
                      [0.10, 0.80, 0.45],
                      [0.95, 0.90, 0.15],
                      [0.90, 0.15, 0.15]], dtype=np.float32)
    positions = np.array([0.0, 0.35, 0.70, 1.0], dtype=np.float32)
    return np.stack([np.interp(norm, positions, stops[:, c]) for c in range(3)], axis=-1)


def anomaly_overlay(image_data: bytes, heatmap: np.ndarray,
                    img_size: int = IMG_SIZE, alpha: float = 0.45,
                    max_size: int = 640) -> bytes:
    """Superpose la carte d'anomalie à l'image d'origine -> PNG (bytes).

    Une seule implémentation, partagée par `scripts/predict.py --heatmap` et par
    l'API (`/predict` + heatmap) : le rendu est donc identique partout, comme pour
    le reste du preprocessing (anti train/serve skew).

    L'image de fond garde sa **résolution d'origine** (bornée à `max_size` px) et
    la carte, calculée en `img_size` px, est agrandie par-dessus : l'œil voit
    nettement la zone suspecte. Le plafond de 640 px limite le poids du PNG
    renvoyé en base64 par l'API (~200 Ko au lieu de ~550 Ko en 900×900).
    """
    with Image.open(io.BytesIO(image_data)) as handle:
        base = handle.convert("RGB")
    if max_size and max(base.size) > max_size:
        ratio = max_size / max(base.size)
        base = base.resize((max(1, int(base.width * ratio)),
                            max(1, int(base.height * ratio))), Image.BILINEAR)

    values = np.asarray(heatmap, dtype=np.float32)
    low, high = float(values.min()), float(values.max())
    norm = (values - low) / (high - low) if high > low else np.zeros_like(values)

    colored = Image.fromarray((_colorize(norm) * 255).astype(np.uint8))
    if colored.size != base.size:
        colored = colored.resize(base.size, Image.BILINEAR)

    buffer = io.BytesIO()
    Image.blend(base, colored, alpha=alpha).save(buffer, format="PNG")
    return buffer.getvalue()


def _optimal_threshold(y_true, scores) -> float | None:
    """Seuil Youden (max tpr - fpr) à partir des scores du split test."""
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(y_true, scores)
    idx = int(np.argmax(tpr - fpr))
    return float(thresholds[idx])


def evaluate_on_test(client, bucket: str, model: PadimModel, category: str,
                     img_size: int = IMG_SIZE) -> dict:
    """Évalue un modèle sur le split test MinIO (good + défauts).

    Parcourt s3://bucket/raw/<category>/test/<label>/, score toutes les images
    (par lots) et retourne {'auc', 'n_test', 'threshold'}. Si les deux classes
    sont présentes, calcule aussi le seuil Youden et le stocke dans
    `model.threshold` (persisté à la sauvegarde de l'artefact).
    """
    from sklearn.metrics import roc_auc_score

    prefix = f"raw/{category}/test/"
    images, y_true = [], []
    for key, _ in list_minio_entries(client, bucket, prefix):
        label_dir = key.rsplit("/", 2)[-2]
        data = client.get_object(bucket, key).read()
        images.append(_bytes_to_array(data, img_size=img_size))
        y_true.append(0 if label_dir == "good" else 1)

    metrics = {"auc": None, "n_test": len(y_true), "threshold": None}
    if not images:
        return metrics

    feats, _ = extract_features(np.stack(images), img_size=img_size,
                                layers=model.feature_layers, batch_size=BATCH_SIZE)
    scores = global_score(compute_anomaly(feats, model))

    if len(set(y_true)) >= 2:
        metrics["auc"] = round(float(roc_auc_score(y_true, scores)), 4)
        metrics["threshold"] = _optimal_threshold(list(y_true), list(scores))
        model.threshold = metrics["threshold"]
    return metrics
