"""
src/shap_explain.py  —  Explicabilidad SHAP por estudiante  v1
--------------------------------------------------------------
Genera valores SHAP para el pipeline XGBoost entrenado.

El pipeline tiene la forma:
    ImbPipeline([("pre", Pipeline([imputer, scaler])), ("smote", ...), ("clf", XGBClassifier)])

Para SHAP necesitamos:
  1. Transformar X con el preprocesador ("pre") para obtener la matriz que
     ve el XGBClassifier.
  2. Crear un TreeExplainer sobre el clasificador interno.
  3. Devolver los SHAP values en el espacio de features originales
     (los nombres no cambian porque sólo hay impute + scale, sin PCA).

Uso rápido:
    from src.shap_explain import explicar_estudiante, explicar_lote

    feats = calcular_features(evaluaciones, cfg, comportamiento)
    resultado = explicar_estudiante(modelo, feats)
    # resultado["shap_values"]   → list[float], uno por feature
    # resultado["importancias"]  → list[dict] ordenadas por |shap|
    # resultado["base_value"]    → float (log-odds base del modelo)
"""

from __future__ import annotations

import numpy  as np
import pandas as pd

from src.config import FEATURES, FEATURE_LABELS

try:
    import shap
    _SHAP_DISPONIBLE = True
except ImportError:
    _SHAP_DISPONIBLE = False


# ── Helpers internos ──────────────────────────────────────────────────────────

def _extraer_preprocesador_y_clf(modelo):
    """
    Descompone el ImbPipeline en (preprocesador, clasificador).
    Soporta tanto ImbPipeline como sklearn Pipeline estándar.
    """
    # ImbPipeline y Pipeline comparten la interfaz de named_steps
    from sklearn.pipeline import Pipeline
    if "pre" in modelo.named_steps:
        pre = modelo.named_steps["pre"]
    else:
        steps = [(k,v) for k,v in modelo.named_steps.items() if k != "clf"]
        pre = Pipeline(steps)
    clf = modelo.named_steps.get("clf") or modelo.named_steps.get("classifier")
    if pre is None or clf is None:
        raise ValueError(
            "No se encontraron los pasos 'pre' y 'clf' en el pipeline. "
            f"Pasos disponibles: {list(modelo.named_steps.keys())}"
        )
    return pre, clf


def _transformar_X(preprocesador, feats: dict) -> np.ndarray:
    """Aplica imputer + scaler al vector de features."""
    X = pd.DataFrame([feats])[FEATURES]
    return preprocesador.transform(X)


# ── API pública ───────────────────────────────────────────────────────────────

def shap_disponible() -> bool:
    """Devuelve True si la librería shap está instalada."""
    return _SHAP_DISPONIBLE


def explicar_estudiante(
    modelo,
    feats: dict,
    top_n: int = 15,
) -> dict:
    """
    Calcula SHAP values para UN estudiante.

    Parámetros
    ----------
    modelo  : pipeline ImbPipeline cargado con joblib
    feats   : dict de 15 features normalizada (salida de calcular_features)
    top_n   : cuántas features incluir en la lista ordenada (máx 15)

    Retorna
    -------
    dict con:
      shap_values  : list[float]  — valor SHAP por feature (mismo orden que FEATURES)
      base_value   : float        — log-odds base del modelo
      importancias : list[dict]   — [{feature, label, shap, valor, direccion}, ...]
                                    ordenadas por |shap| descendente
      error        : str | None   — si shap no está instalado
    """
    if not _SHAP_DISPONIBLE:
        return {
            "shap_values":  [],
            "base_value":   None,
            "importancias": [],
            "error": "Instala shap: pip install shap",
        }

    pre, clf = _extraer_preprocesador_y_clf(modelo)
    X_trans  = _transformar_X(pre, feats)

    explainer   = shap.TreeExplainer(clf)
    shap_output = explainer(X_trans)

    # shap_output.values tiene forma (1, n_features, n_classes) para multi-output
    # o (1, n_features) para binary con output_type="margin"
    vals = shap_output.values[0]
    if vals.ndim == 2:
        # Tomar la clase positiva (riesgo = 1)
        vals = vals[:, 1]

    base_val = float(explainer.expected_value[1] if hasattr(explainer.expected_value, "__len__")
                     else explainer.expected_value)

    # Construir lista legible ordenada por importancia absoluta
    importancias = []
    for i, feat_name in enumerate(FEATURES):
        sv = float(vals[i])
        importancias.append({
            "feature":   feat_name,
            "label":     FEATURE_LABELS.get(feat_name, feat_name),
            "shap":      round(sv, 5),
            "valor":     round(float(feats.get(feat_name, 0)), 4),
            "direccion": "aumenta_riesgo" if sv > 0 else "reduce_riesgo",
        })

    importancias.sort(key=lambda x: abs(x["shap"]), reverse=True)

    return {
        "shap_values":  [round(float(v), 5) for v in vals],
        "base_value":   round(base_val, 5),
        "importancias": importancias[:top_n],
        "error":        None,
    }


def explicar_lote(
    modelo,
    lista_feats: list[dict],
    nombres: list[str] | None = None,
) -> pd.DataFrame:
    """
    Calcula SHAP values para una lista de estudiantes.

    Parámetros
    ----------
    modelo       : pipeline cargado
    lista_feats  : lista de dicts de features (uno por estudiante)
    nombres      : lista de nombres opcionales

    Retorna
    -------
    DataFrame con columnas: nombre, <feature>_shap × 15, shap_max_feature
    """
    if not _SHAP_DISPONIBLE:
        raise ImportError("Instala shap: pip install shap")

    pre, clf = _extraer_preprocesador_y_clf(modelo)

    X = pd.DataFrame(lista_feats)[FEATURES]
    X_trans = pre.transform(X)

    explainer   = shap.TreeExplainer(clf)
    shap_output = explainer(X_trans)

    vals = shap_output.values
    if vals.ndim == 3:
        vals = vals[:, :, 1]   # clase positiva

    filas = []
    for i, feats in enumerate(lista_feats):
        sv = vals[i]
        row = {"nombre": (nombres[i] if nombres else f"est_{i+1}")}
        for j, feat in enumerate(FEATURES):
            row[f"{feat}_shap"] = round(float(sv[j]), 5)
        # Feature con mayor impacto (positivo o negativo)
        max_idx = int(np.argmax(np.abs(sv)))
        row["shap_max_feature"] = FEATURES[max_idx]
        row["shap_max_label"]   = FEATURE_LABELS.get(FEATURES[max_idx], FEATURES[max_idx])
        filas.append(row)

    return pd.DataFrame(filas)


def resumen_importancia_global(
    modelo,
    lista_feats: list[dict],
) -> pd.DataFrame:
    """
    Calcula la importancia global de cada feature como mean(|SHAP|)
    sobre una lista de estudiantes.

    Retorna DataFrame con columnas: feature, label, importancia_media
    ordenado descendente.
    """
    if not _SHAP_DISPONIBLE:
        raise ImportError("Instala shap: pip install shap")

    pre, clf = _extraer_preprocesador_y_clf(modelo)
    X        = pd.DataFrame(lista_feats)[FEATURES]
    X_trans  = pre.transform(X)

    explainer   = shap.TreeExplainer(clf)
    shap_output = explainer(X_trans)
    vals = shap_output.values
    if vals.ndim == 3:
        vals = vals[:, :, 1]

    mean_abs = np.abs(vals).mean(axis=0)

    return pd.DataFrame({
        "feature":           FEATURES,
        "label":             [FEATURE_LABELS.get(f, f) for f in FEATURES],
        "importancia_media": [round(float(v), 5) for v in mean_abs],
    }).sort_values("importancia_media", ascending=False).reset_index(drop=True)
