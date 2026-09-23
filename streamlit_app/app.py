"""Interface de démonstration — Anomalies Indus (Streamlit).

Cette app est un **client HTTP de l'API** : aucune logique ML ici (voir
`api_client.py`). Elle ne sert qu'à rendre la démonstration confortable :

  1. Prédiction   — une image, un verdict, la carte d'anomalie superposée ;
  2. Test par lot — N images du jeu de test, tableau + taux de détection ;
  3. Entraînement — déclencher `POST /training` et voir la promotion du champion ;
  4. Modèle & traçabilité — champion par catégorie, commit du code, empreinte du dataset ;
  5. Monitoring   — état des services + métriques clés de l'API.

Lancement local :   streamlit run streamlit_app/app.py
Lancement Docker :  docker compose up -d streamlit   ->  http://localhost:8501
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import api_client as api  # noqa: E402

DATASET_DIR = Path(os.getenv("DATASET_DIR", "dataset/raw"))
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090").rstrip("/")
MLFLOW_CHECK_URL = os.getenv("MLFLOW_INTERNAL_URL", "http://localhost:5050").rstrip("/")
AIRFLOW_CHECK_URL = os.getenv("AIRFLOW_INTERNAL_URL", "http://localhost:8080").rstrip("/")
LINKS = {
    "MLflow (runs + Registry)": os.getenv("MLFLOW_URL", "http://localhost:5050"),
    "Grafana (monitoring)": os.getenv("GRAFANA_URL", "http://localhost:3000"),
    "Airflow (orchestration)": os.getenv("AIRFLOW_URL", "http://localhost:8080"),
    "MinIO (console objet store)": os.getenv("MINIO_CONSOLE_URL", "http://localhost:9200"),
    # URL navigateur : `api.API_BASE_URL` est un nom de service Docker (api:8000),
    # donc inutilisable depuis le poste de l'utilisateur.
    "API (docs Swagger)": os.getenv("API_DOCS_URL", "http://localhost:8000/docs"),
}

st.set_page_config(page_title="Anomalies Indus", page_icon="🔍", layout="wide")


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=20, show_spinner=False)
def fetch_models() -> dict:
    """`GET /models` (caché 20 s pour ne pas marteler l'API à chaque interaction)."""
    return api.models()


def show_table(data: pd.DataFrame, height: int | None = None) -> None:
    """Affiche un DataFrame (compatible plusieurs versions de Streamlit).

    `height=None` est refusé par Streamlit (il faut un entier, "stretch" ou
    "content") : on ne transmet donc l'argument que s'il est fourni.
    """
    extra = {"height": height} if height is not None else {}
    try:
        st.dataframe(data, width="stretch", hide_index=True, **extra)
    except TypeError:  # Streamlit < 1.49 : pas de paramètre `width`
        st.dataframe(data, use_container_width=True, hide_index=True, **extra)


def model_categories() -> list[str]:
    """Catégories pour lesquelles un modèle est servi."""
    try:
        data = fetch_models()
    except api.ApiError as exc:
        st.error(str(exc))
        return []
    return [m["category"] for m in data.get("models", []) if m.get("champion")]


def bucket_categories() -> list[str]:
    """Catégories présentes dans MinIO (même sans modèle entraîné)."""
    try:
        return fetch_models().get("categories", [])
    except api.ApiError:
        return []


def test_labels(category: str) -> list[str]:
    root = DATASET_DIR / category / "test"
    return sorted(p.name for p in root.iterdir() if p.is_dir()) if root.is_dir() else []


def test_images(category: str, label: str) -> list[Path]:
    root = DATASET_DIR / category / "test" / label
    return sorted(root.glob("*.png")) if root.is_dir() else []


def parse_metrics(text: str) -> dict[str, float]:
    """Extrait quelques métriques clés de la sortie texte de `/metrics`."""
    wanted = ("anomalies_predictions_total", "anomalies_trainings_total",
              "anomalies_champion_auc", "anomalies_prediction_score_ratio")
    found: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#") or not line:
            continue
        name = line.split("{")[0].split(" ")[0]
        if name in wanted:
            try:
                found[f"{name}{line[len(name):line.rindex(' ')]}"] = float(line.rsplit(" ", 1)[1])
            except ValueError:
                continue
    return found


def metric_label(key: str) -> str:
    """`anomalies_champion_auc{category="bottle"}` -> `champion_auc bottle`."""
    name, _, rest = key.partition("{")
    labels = [part.split("=")[-1].strip('"') for part in rest.rstrip("}").split(",") if part]
    return " ".join([name.replace("anomalies_", "")] + labels)


# ─────────────────────────────────────────────────────────────
# 1. Prédiction
# ─────────────────────────────────────────────────────────────
def page_predict() -> None:
    st.header("Prédiction d'une pièce")
    st.caption("Appelle `POST /predict` : c'est le même service que la CLI et l'API.")

    categories = model_categories()
    if not categories:
        st.warning("Aucun modèle entraîné. Va sur la page **Entraînement** pour en créer un.")
        return

    left, right = st.columns([1, 2], gap="large")
    with left:
        category = st.selectbox("Catégorie", categories)
        origin = st.radio("Image", ["Téléverser", "Depuis le jeu de test"], horizontal=False)

        image_bytes: bytes | None = None
        filename = "image.png"
        expected: str | None = None

        if origin == "Téléverser":
            upload = st.file_uploader("Fichier image", type=["png", "jpg", "jpeg"])
            if upload is not None:
                image_bytes, filename = upload.getvalue(), upload.name
        else:
            labels = test_labels(category)
            if not labels:
                st.info(f"Aucune image locale dans {DATASET_DIR / category / 'test'} "
                        "(monte `dataset/raw` pour utiliser cette option).")
            else:
                label = st.selectbox("Type", labels,
                                     index=labels.index("good") if "good" in labels else 0)
                files = test_images(category, label)
                if files:
                    choice = st.selectbox("Image", files, format_func=lambda p: p.name)
                    image_bytes, filename = choice.read_bytes(), choice.name
                    expected = "saine" if label == "good" else f"défectueuse ({label})"

        with_heatmap = st.checkbox("Carte d'anomalie", value=True)
        launch = st.button("Prédire", type="primary", disabled=image_bytes is None)

    if launch and image_bytes is not None:
        with st.spinner("Inférence en cours…"):
            try:
                st.session_state["prediction"] = api.predict(
                    category, image_bytes, filename=filename, heatmap=with_heatmap
                )
            except api.ApiError as exc:
                st.session_state.pop("prediction", None)
                st.error(str(exc))

    result = st.session_state.get("prediction")
    with right:
        if not result or result.get("category") != category:
            st.info("Choisis une image puis clique sur **Prédire**.")
            return

        anomaly = result.get("anomaly")
        threshold = result.get("threshold") or 0
        score = result.get("score") or 0
        ratio = score / threshold if threshold else 0

        message = "ANOMALIE détectée" if anomaly else "Pièce jugée SAINE"
        (st.error if anomaly else st.success)(
            f"**{message}** — score {score:,.0f} / seuil {threshold:,.0f} ({ratio:.2f}×)"
        )
        if expected:
            st.caption(f"Vérité terrain : {expected}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Score (Mahalanobis²)", f"{score:,.0f}")
        col2.metric("Seuil du modèle", f"{threshold:,.0f}")
        col3.metric("Score / seuil", f"{ratio:.2f}×")

        images = st.columns(2)
        images[0].image(image_bytes, caption=f"Image reçue ({filename})")
        if result.get("heatmap_png"):
            images[1].image(api.to_bytes(result["heatmap_png"]),
                            caption="Carte d'anomalie (rouge = zone suspecte)")

        st.caption(f"Modèle servi : `{result.get('model')}` (source : {result.get('model_source')})")


# ─────────────────────────────────────────────────────────────
# 2. Test par lot
# ─────────────────────────────────────────────────────────────
def page_batch() -> None:
    st.header("Test par lot")
    st.caption("Enchaîne les appels `/predict` sur des images du jeu de test : "
               "c'est la démonstration « le modèle attrape-t-il les défauts ? ».")

    categories = model_categories()
    if not categories:
        st.warning("Aucun modèle entraîné : lance d'abord un entraînement.")
        return

    controls = st.columns([1, 1, 1, 1])
    category = controls[0].selectbox("Catégorie", categories, key="batch_category")
    labels = test_labels(category)
    if not labels:
        st.info(f"Jeu de test introuvable dans {DATASET_DIR / category} (monte `dataset/raw`).")
        return

    label = controls[1].selectbox("Type d'images", labels,
                                  index=labels.index("good") if "good" in labels else 0)
    count = controls[2].slider("Nombre d'images", 2, 20, 6)
    go = controls[3].button("Lancer", type="primary")

    if not go:
        return

    paths = test_images(category, label)[:count]
    if not paths:
        st.warning("Pas d'image trouvée.")
        return

    expected_anomaly = label != "good"
    rows = []
    progress = st.progress(0.0, text="Inférence…")
    for index, path in enumerate(paths, start=1):
        try:
            result = api.predict(category, path.read_bytes(), filename=path.name, heatmap=False)
        except api.ApiError as exc:
            st.error(str(exc))
            return
        rows.append({
            "image": path.name,
            "attendu": "anomalie" if expected_anomaly else "saine",
            "verdict": "ANOMALIE" if result["anomaly"] else "saine",
            "correct": (result["anomaly"] == expected_anomaly),
            "score": result["score"],
            "seuil": result["threshold"],
        })
        progress.progress(index / len(paths), text=f"Inférence… {index}/{len(paths)}")
    progress.empty()

    frame = pd.DataFrame(rows)
    correct = int(frame["correct"].sum())
    st.subheader(f"{correct} / {len(frame)} images correctement classées")

    metrics = st.columns(3)
    metrics[0].metric("Taux de bonnes réponses", f"{correct / len(frame):.0%}")
    metrics[1].metric("Type d'image", label)
    metrics[2].metric("Seuil appliqué", f"{frame['seuil'].iloc[0]:,.0f}")

    st.bar_chart(frame.set_index("image")["score"], height=240)
    show_table(frame.drop(columns=["correct"]))


# ─────────────────────────────────────────────────────────────
# 3. Entraînement
# ─────────────────────────────────────────────────────────────
def page_training() -> None:
    st.header("Entraînement")
    st.caption("Déclenche `POST /training`. Un entraînement enregistré peut **promouvoir "
               "le champion** s'il fait au moins aussi bien que celui en place.")

    categories = bucket_categories() or ["bottle"]
    with st.form("training"):
        col1, col2 = st.columns(2)
        category = col1.selectbox("Catégorie (présente dans MinIO)", categories)
        version = col2.selectbox("Version de données (croissance simulée)",
                                 ["full", 0, 1, 2, 3, 4], index=0,
                                 format_func=lambda v: "100 % (full)" if v == "full" else f"v{v}")
        col3, col4, col5 = st.columns(3)
        do_eval = col3.checkbox("Évaluer (AUC + seuil)", value=True)
        register = col4.checkbox("Enregistrer au Registry", value=True)
        promote = col5.checkbox("Promouvoir si meilleur", value=True)
        st.caption("Sans évaluation, pas de promotion possible (comparaison sur l'AUC).")
        submitted = st.form_submit_button("Lancer l'entraînement", type="primary")

    if not submitted:
        return

    payload = {"category": category, "eval": do_eval, "register": register, "promote": promote}
    if version != "full":
        payload["data_version"] = version

    with st.spinner("Entraînement en cours (de 30 s à 3 min selon l'état du cache)…"):
        try:
            result = api.train(payload)
        except api.ApiError as exc:
            st.error(str(exc))
            return

    st.success(f"Terminé en {result.get('elapsed_s')} s — AUC {result.get('auc')}")
    promotion = result.get("mlflow_promotion")
    if promotion:
        if promotion.get("promoted"):
            st.success(f"🏆 Nouveau champion : version {promotion.get('candidate_version')} "
                       f"(AUC {promotion.get('candidate_score')} ≥ "
                       f"{promotion.get('previous_champion_score')})")
        else:
            st.info("Pas de promotion : "
                    f"{promotion.get('reason') or 'version candidate moins bonne que le champion'} "
                    f"(candidate {promotion.get('candidate_score')} vs champion "
                    f"{promotion.get('previous_champion_score')})")

    st.json(result)
    fetch_models.clear()  # rafraîchit la liste des modèles dans l'UI


# ─────────────────────────────────────────────────────────────
# 4. Modèle & traçabilité
# ─────────────────────────────────────────────────────────────
def page_model() -> None:
    st.header("Modèle & traçabilité")
    st.caption("Quel modèle est servi, et **quel code / quelles données** l'ont produit "
               "(voir la section Versioning du README).")

    try:
        data = fetch_models()
    except api.ApiError as exc:
        st.error(str(exc))
        return

    champions = [m for m in data.get("models", []) if m.get("champion")]
    if champions:
        frame = pd.DataFrame([{
            "catégorie": m["category"],
            "champion": f"v{m['champion']['version']}",
            "AUC": m["champion"]["auc"],
            "seuil": m["champion"]["threshold"],
            "images train": m["champion"]["n_train"],
            "data_version": m["champion"]["data_version"] or "—",
            "commit code": m["champion"]["git_commit"] or "—",
            "dataset_sha256": (m["champion"]["dataset_sha256"] or "")[:12] or "—",
            "candidate": (f"v{m['candidate']['version']} (AUC {m['candidate']['auc']})"
                          if m.get("candidate") else "—"),
        } for m in champions])
        st.subheader("Champion par catégorie")
        show_table(frame)
    else:
        st.info("Aucun champion enregistré pour l'instant.")

    st.divider()
    st.subheader("Derniers runs d'entraînement")
    try:
        runs = api.runs(limit=15)["runs"]
    except api.ApiError as exc:
        st.error(str(exc))
        return

    if not runs:
        st.info("Aucun run.")
        return

    frame = pd.DataFrame([{
        "début": pd.to_datetime(r["start_time"], unit="ms").strftime("%d/%m %H:%M:%S"),
        "catégorie": r["name"],
        "statut": r["status"],
        "AUC": r["auc"],
        "images train": r["n_train"],
        "data_version": r["data_version"] or "—",
        "commit": (r["git_commit"] or "—")[:12],
        "dataset": (r["dataset_sha256"] or "—")[:12],
        "run_id": r["run_id"][:8],
    } for r in runs])
    show_table(frame)
    st.caption("Les colonnes `commit` et `dataset` sont la traçabilité demandée : "
               "code + jeu d'images de chaque run. Détail complet : "
               "`python scripts/lineage.py --category <cat>`.")


# ─────────────────────────────────────────────────────────────
# 5. Monitoring
# ─────────────────────────────────────────────────────────────
def page_monitoring() -> None:
    st.header("Monitoring")
    st.caption("État des services et métriques exposées par l'API (`GET /metrics`).")

    def light(label: str, url: str | None, link: str) -> None:
        if url is None:
            return
        try:
            ok = requests.get(url, timeout=3).status_code < 500
        except requests.RequestException:
            ok = False
        st.markdown(f"{'🟢' if ok else '🔴'} **{label}** — [{link}]({link})")

    st.subheader("Services")
    checked, links = st.columns(2, gap="large")
    with checked:
        light("API d'inférence", f"{api.API_BASE_URL}/", f"{api.API_BASE_URL}/docs")
        light("Prometheus", f"{PROMETHEUS_URL}/-/ready", "http://localhost:9090")
        light("MLflow", f"{MLFLOW_CHECK_URL}/health", LINKS["MLflow (runs + Registry)"])
        light("Airflow", f"{AIRFLOW_CHECK_URL}/health", LINKS["Airflow (orchestration)"])
    with links:
        st.markdown("\n".join(f"- [{label}]({url})" for label, url in LINKS.items()))
        st.caption("Grafana et Airflow vivent dans des profils Docker : "
                   "`docker compose --profile monitoring up -d`")

    st.divider()
    st.subheader("Métriques de l'API")
    try:
        values = parse_metrics(api.metrics_text())
    except api.ApiError as exc:
        st.error(str(exc))
        return

    predictions = {k: v for k, v in values.items() if k.startswith("anomalies_predictions_total")}
    if predictions:
        cols = st.columns(len(predictions))
        for col, (key, value) in zip(cols, predictions.items()):
            col.metric(f"prédictions {metric_label(key).split()[-1]}", f"{value:,.0f}")
    aucs = {k: v for k, v in values.items() if k.startswith("anomalies_champion_auc")}
    trainings = {k: v for k, v in values.items() if k.startswith("anomalies_trainings_total")}
    other = st.columns(max(len(aucs) + len(trainings), 1))
    for col, (key, value) in zip(other, {**aucs, **trainings}.items()):
        decimals = 4 if "auc" in key else 0
        col.metric(metric_label(key), f"{value:,.{decimals}f}")

    if len(values) > 8:
        st.caption("Le reste des séries est disponible dans Grafana.")
    st.caption("Ces compteurs repartent de zéro à chaque redémarrage de l'API "
               "(l'historique reste dans Prometheus).")


# ─────────────────────────────────────────────────────────────
def main() -> None:
    st.sidebar.title("🔍 Anomalies Indus")
    st.sidebar.caption("Détection d'anomalies PaDiM — démonstration")

    page = st.sidebar.radio(
        "Page",
        ["Prédiction", "Test par lot", "Entraînement", "Modèle & traçabilité", "Monitoring"],
        label_visibility="collapsed",
    )

    try:
        version = api.health().get("version")
        st.sidebar.success(f"API connectée (v{version})")
    except api.ApiError as exc:
        st.sidebar.error("API injoignable")
        st.sidebar.caption(str(exc))

    st.sidebar.divider()
    st.sidebar.markdown("**Liens**")
    for label, url in LINKS.items():
        st.sidebar.markdown(f"- [{label}]({url})")

    {
        "Prédiction": page_predict,
        "Test par lot": page_batch,
        "Entraînement": page_training,
        "Modèle & traçabilité": page_model,
        "Monitoring": page_monitoring,
    }[page]()


main()
