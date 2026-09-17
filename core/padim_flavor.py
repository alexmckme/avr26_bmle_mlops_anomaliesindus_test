"""Flavor MLflow `pyfunc` pour PaDiM (modèle custom).

Permet d'enregistrer PaDiM comme un **modèle MLflow standard** (donc
enregistrable dans le Model Registry) alors que ce n'est pas un modèle
scikit-learn/keras classique.

Le wrapper réutilise `core.padim` tel quel :
  - `load_context` charge l'artefact `.npz` (mean + cov_inv + seuil) ;
  - `predict` appelle `core.padim.predict_image` sur chaque image.

L'entrée de `predict` est une image (`bytes`) ou une liste d'images (`bytes`).
La sortie est un DataFrame avec les colonnes `score`, `threshold`, `anomaly`.
"""

from __future__ import annotations

import mlflow.pyfunc
import pandas as pd

import core.padim as padim


class PadimPyfunc(mlflow.pyfunc.PythonModel):
    """Modèle PaDiM exposé comme modèle MLflow (`pyfunc`)."""

    def load_context(self, context) -> None:
        self.model = padim.load_model(context.artifacts["padim_model"])

    def predict(self, context, model_input):
        if isinstance(model_input, (bytes, bytearray)):
            items = [model_input]
        else:
            items = list(model_input)

        rows = []
        for item in items:
            data = bytes(item)
            score, _ = padim.predict_image(data, self.model)
            anomaly = (None if self.model.threshold is None
                       else bool(score > self.model.threshold))
            rows.append({
                "score": float(score),
                "threshold": self.model.threshold,
                "anomaly": anomaly,
            })
        return pd.DataFrame(rows)
