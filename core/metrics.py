"""Métriques Prometheus du service d'anomalies.

À quoi ça sert ?
----------------
MLflow répond à « quel modèle est le meilleur ? » (historique, comparaison).
Le monitoring répond à « comment se comporte le système **maintenant** ? » :

  - côté **service** : débit, latence, codes de statut → fourni « clé en main » par
    `prometheus-fastapi-instrumentator` (voir `api/main.py`, préfixe `http_*`) ;
  - côté **modèle**  : nombre de prédictions, part d'anomalies détectées, scores
    rapportés au seuil, AUC du champion → métriques métier définies ci-dessous
    (préfixe `anomalies_*`).

Prometheus *scrape* `GET /metrics` (modèle « pull », toutes les 15 s) et Grafana
affiche ces séries. Aucune image ni donnée individuelle n'est transmise : uniquement
des compteurs et des histogrammes agrégés.

Dégradation gracieuse : si `prometheus_client` n'est pas installé (venv sans
dépendances de monitoring), tous les helpers deviennent des no-ops et l'API
continue de fonctionner normalement — même philosophie que `core/tracking.py`.
"""

from __future__ import annotations

try:
    from prometheus_client import Counter, Gauge, Histogram

    _ENABLED = True
except ImportError:  # pragma: no cover - dépendance de monitoring absente
    _ENABLED = False


class _NoopMetric:
    """Métrique inerte : même interface que prometheus_client, mais sans effet."""

    def labels(self, *args, **kwargs) -> "_NoopMetric":
        return self

    def inc(self, *args, **kwargs) -> None:
        pass

    def observe(self, *args, **kwargs) -> None:
        pass

    def set(self, *args, **kwargs) -> None:
        pass


def _counter(name: str, doc: str, labels: list[str]):
    return Counter(name, doc, labels) if _ENABLED else _NoopMetric()


def _gauge(name: str, doc: str, labels: list[str]):
    return Gauge(name, doc, labels) if _ENABLED else _NoopMetric()


def _histogram(name: str, doc: str, labels: list[str], buckets: tuple[float, ...]):
    return Histogram(name, doc, labels, buckets=buckets) if _ENABLED else _NoopMetric()


# ── Modèle : inférence ────────────────────────────────────────────────────────
PREDICTIONS = _counter(
    "anomalies_predictions_total",
    "Prédictions servies, par catégorie et verdict (anomaly / normal).",
    ["category", "verdict"],
)

PREDICTION_RATIO = _gauge(
    "anomalies_prediction_score_ratio",
    "Dernier score rapporté au seuil du modèle (1.0 = exactement le seuil). "
    "Indépendant de l'échelle des scores, donc comparable entre catégories.",
    ["category"],
)

PREDICTION_SCORE = _histogram(
    "anomalies_prediction_score",
    "Distribution des scores d'anomalie (distance de Mahalanobis²) par catégorie.",
    ["category"],
    # Les scores vont de ~1e6 (sain) à ~1e9 (défaut net) : bornes volontairement larges.
    buckets=(1e6, 5e6, 1e7, 2e7, 5e7, 1e8, 5e8, 1e9),
)

# ── Modèle : entraînement et cycle de vie ─────────────────────────────────────
TRAININGS = _counter(
    "anomalies_trainings_total",
    "Entraînements déclenchés via l'API, par catégorie et statut (success / error).",
    ["category", "status"],
)

TRAINING_DURATION = _histogram(
    "anomalies_training_duration_seconds",
    "Durée d'un entraînement complet (fit + évaluation) en secondes.",
    ["category"],
    buckets=(1, 2.5, 5, 10, 20, 30, 60, 120, 300, 600),
)

CHAMPION_AUC = _gauge(
    "anomalies_champion_auc",
    "AUC du modèle promu 'champion' (renseigné au démarrage depuis le Registry, "
    "puis à chaque promotion). C'est l'indicateur de qualité de ce qui est servi.",
    ["category"],
)

PROMOTIONS = _counter(
    "anomalies_champion_promotions_total",
    "Décisions de promotion en 'champion', par catégorie et résultat (true / false).",
    ["category", "promoted"],
)


# ── Helpers appelés par l'API ─────────────────────────────────────────────────
def observe_prediction(
    category: str,
    score: float,
    threshold: float | None,
    anomaly: bool | None,
) -> None:
    """Enregistre une prédiction : verdict, score, et score/seuil."""
    verdict = "unknown" if anomaly is None else ("anomaly" if anomaly else "normal")
    PREDICTIONS.labels(category=category, verdict=verdict).inc()
    PREDICTION_SCORE.labels(category=category).observe(float(score))
    if threshold:
        PREDICTION_RATIO.labels(category=category).set(float(score) / float(threshold))


def observe_training(category: str, status: str, duration_s: float) -> None:
    """Enregistre un entraînement (statut `success` ou `error`) et sa durée."""
    TRAININGS.labels(category=category, status=status).inc()
    TRAINING_DURATION.labels(category=category).observe(float(duration_s))


def observe_promotion(category: str, promoted: bool, auc: float | None = None) -> None:
    """Enregistre une décision de promotion ; met à jour l'AUC du champion si promu."""
    PROMOTIONS.labels(category=category, promoted=str(bool(promoted)).lower()).inc()
    if promoted and auc is not None:
        CHAMPION_AUC.labels(category=category).set(float(auc))


def refresh_champion_gauges() -> int:
    """Renseigne `anomalies_champion_auc` depuis le Registry MLflow.

    Appelé au démarrage de l'API : sans ça la jauge ne serait alimentée qu'à la
    première promotion, et le tableau de bord Grafana resterait vide alors qu'un
    champion existe déjà. Retourne le nombre de catégories renseignées.
    """
    try:
        import mlflow
    except ImportError:
        return 0

    updated = 0
    try:
        client = mlflow.MlflowClient()
        for registered in client.search_registered_models():
            if not registered.name.startswith("padim-"):
                continue
            category = registered.name[len("padim-"):]
            try:
                version = client.get_model_version_by_alias(registered.name, "champion")
                auc = client.get_run(version.run_id).data.metrics.get("auc")
            except Exception:  # alias absent ou run supprimé : catégorie ignorée
                continue
            if auc is not None:
                CHAMPION_AUC.labels(category=category).set(float(auc))
                updated += 1
    except Exception:  # Registry injoignable : on ne bloque surtout pas le démarrage
        return updated
    return updated
