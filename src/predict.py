"""
src/predict.py  —  Predicción con evaluaciones dinámicas  v2
-------------------------------------------------------------
Mantiene la API original (predecir_estudiante / predecir_lote)
pero acepta el nuevo formato de evaluaciones dinámicas.

COMPATIBILIDAD: si el modelo pkl fue entrenado con las 9 features del CSV
original, puedes seguir usando predict_legacy() mientras migras.
"""

import joblib
import pandas as pd
import numpy  as np

from src.config   import MODEL_PATH, MODEL_INFO_PATH, FEATURES, DECISION_UMBRAL, LABEL_MAP
from src.features import calcular_features, evaluar_detalle_variables


# ── Carga ──────────────────────────────────────────────────────────────────

def cargar_modelo():
    return joblib.load(MODEL_PATH)

def cargar_info_modelo() -> dict:
    try:
        return joblib.load(MODEL_INFO_PATH)
    except FileNotFoundError:
        return {}

def modelo_disponible() -> bool:
    return MODEL_PATH.exists()


# ── Predicción con evaluaciones dinámicas ─────────────────────────────────

def predecir_estudiante_dinamico(
    modelo,
    evaluaciones: list[dict],
    cfg: dict,
    comportamiento: dict | None = None,
) -> dict:
    """
    Predice el riesgo de UN estudiante usando evaluaciones dinámicas.

    Parámetros
    ----------
    modelo         : pipeline cargado
    evaluaciones   : lista de dicts con claves slug, peso, valor (puede ser None), orden, nombre
                     Ejemplo:
                     [
                       {"slug":"parcial",    "nombre":"Parcial 1", "peso":25, "valor":72, "orden":1},
                       {"slug":"proyecto",   "nombre":"Proyecto",  "peso":40, "valor":None,"orden":2},
                       {"slug":"puntos_extra","nombre":"Bonus",    "peso":0,  "valor":5,  "orden":3},
                     ]
    cfg            : dict con "nota_minima_aprobacion" y "escala_maxima"
    comportamiento : dict opcional con asistencia, tareas_a_tiempo, tareas_total,
                     visitas_tutoria, materias_paralelo, es_repitente.
                     Si es None, los features de comportamiento se imputan con mediana.

    Retorna
    -------
    dict con clase, probabilidad, etiqueta, nivel_riesgo, features, detalle
    """
    umbral = cargar_info_modelo().get("umbral", DECISION_UMBRAL)

    # 1. Calcular el vector de 15 features normalizadas
    feats = calcular_features(evaluaciones, cfg, comportamiento)

    # 2. Construir el DataFrame en el orden que el modelo espera
    X = pd.DataFrame([feats])[FEATURES]

    # 3. Predicción
    proba = float(modelo.predict_proba(X)[0][1])
    clase = int(proba >= umbral)

    # 4. Nivel de riesgo en 3 bandas
    if proba >= 0.70:
        nivel = "Alto"
    elif proba >= umbral:
        nivel = "Moderado"
    else:
        nivel = "Bajo"

    # 5. Detalle legible por evaluación (para la UI)
    detalle = evaluar_detalle_variables(evaluaciones, cfg, comportamiento)

    return {
        "clase":        clase,
        "probabilidad": round(proba, 4),
        "etiqueta":     LABEL_MAP[clase],
        "nivel_riesgo": nivel,
        "features":     feats,    # el vector interno (útil para SHAP)
        "detalle":      detalle,
    }


def predecir_lote_dinamico(
    modelo,
    estudiantes: list[dict],   # [{"nombre": ..., "evaluaciones": [...], "cfg": {...}, "comportamiento": {...}}, ...]
) -> pd.DataFrame:
    """
    Predice sobre una lista de estudiantes con evaluaciones dinámicas.
    Cada estudiante puede tener su propio cfg y comportamiento (materias diferentes).
    La clave "comportamiento" es opcional; si falta se imputa con mediana.
    """
    umbral = cargar_info_modelo().get("umbral", DECISION_UMBRAL)
    filas  = []

    for est in estudiantes:
        comp  = est.get("comportamiento")
        feats = calcular_features(est["evaluaciones"], est["cfg"], comp)
        X     = pd.DataFrame([feats])[FEATURES]
        proba = float(modelo.predict_proba(X)[0][1])
        pred  = int(proba >= umbral)

        filas.append({
            "nombre":               est.get("nombre", "—"),
            **feats,
            "probabilidad_riesgo":  round(proba, 4),
            "prediccion":           pred,
            "etiqueta":             LABEL_MAP[pred],
            "nivel_riesgo":         "Alto" if proba >= 0.70 else "Moderado" if proba >= umbral else "Bajo",
        })

    return pd.DataFrame(filas)


# ── API legada (modelo entrenado con CSV de 9 variables) ─────────────────────
# Mantiene compatibilidad mientras no se reentrenan los datos.

def predecir_estudiante(modelo, datos_dict: dict) -> dict:
    """
    Predicción con el formato original del CSV (9 variables fijas).
    Sigue funcionando con el modelo pkl ya entrenado.
    """
    from src.config import DECISION_UMBRAL as DU

    umbral = cargar_info_modelo().get("umbral", DU)

    CSV_FEATURES = [
        "nota_parcial_1", "nota_parcial_2", "porcentaje_asistencia",
        "tareas_entregadas", "horas_estudio_semanal", "participa_tutoria",
        "promedio_ciclo_anterior", "nivel_socioeconomico", "acceso_internet",
    ]

    df    = pd.DataFrame([datos_dict])[CSV_FEATURES]
    proba = float(modelo.predict_proba(df)[0][1])
    clase = int(proba >= umbral)

    if proba >= 0.70:    nivel = "Alto"
    elif proba >= umbral: nivel = "Moderado"
    else:                nivel = "Bajo"

    return {
        "clase":        clase,
        "probabilidad": round(proba, 4),
        "etiqueta":     LABEL_MAP[clase],
        "nivel_riesgo": nivel,
    }