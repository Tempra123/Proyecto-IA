"""
src/features.py  —  Motor de normalización dinámica  v2
---------------------------------------------------------
Transforma evaluaciones + datos de comportamiento/contexto
en el vector de 15 features que el modelo XGBoost espera.

GARANTÍA CENTRAL:
  La salida siempre es el mismo dict con las mismas 14 claves,
  sin importar cuántas evaluaciones tenga la materia ni qué
  datos de comportamiento estén disponibles (los faltantes
  se imputan con mediana, igual que antes).

Uso rápido:
    from src.features import calcular_features

    evaluaciones = [
        {"slug": "parcial",  "peso": 30, "valor": 72, "orden": 1, "escala_slug": "100"},
        {"slug": "proyecto", "peso": 40, "valor": 80, "orden": 2, "escala_slug": "100"},
        {"slug": "puntos_extra", "peso": 0, "valor": 5, "orden": 3, "escala_slug": "100"},
    ]
    cfg = {"nota_minima_aprobacion": 51, "escala_maxima": 100}

    comportamiento = {
        "asistencia":        0.85,   # fracción [0-1]
        "tareas_a_tiempo":   4,      # entregadas a tiempo
        "tareas_total":      5,      # total de tareas asignadas
        "visitas_tutoria":   2,      # número de visitas (0-5+)
        "materias_paralelo": 3,      # materias cursando en paralelo
        "es_repitente":      False,  # True si ya cursó esta materia antes
    }

    features = calcular_features(evaluaciones, cfg, comportamiento)
    # → {"nota_ponderada": 0.79, "asistencia": 0.85, ...}  — 15 keys
"""

from __future__ import annotations
import numpy as np
from typing import Any
from src.escalas import normalizar, es_aprobado


# ── Constantes de normalización ───────────────────────────────────────────────
_MAX_VISITAS_TUTORIA  = 5    # cap: 5+ visitas → 1.0
_MAX_MATERIAS         = 8    # cap: 8+ materias → 1.0
_UMBRAL_ASISTENCIA    = 0.70 # por debajo → alerta_asistencia = 1
_UMBRAL_CAIDA_BRUSCA  = 0.25 # caída de más de 25 puntos norm → caida_brusca = 1


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def calcular_features(
    evaluaciones:  list[dict[str, Any]],
    cfg:           dict[str, Any],
    comportamiento: dict[str, Any] | None = None,
) -> dict[str, float]:
    """
    Parámetros
    ----------
    evaluaciones   : lista de dicts — slug, peso, valor, orden, escala_slug
    cfg            : dict — "nota_minima_aprobacion", "escala_maxima"
    comportamiento : dict opcional con claves:
                       asistencia        (float 0-1, ej: 0.85)
                       tareas_a_tiempo   (int, nro entregadas a tiempo)
                       tareas_total      (int, nro total de tareas)
                       visitas_tutoria   (int, 0-5+)
                       materias_paralelo (int, 1-8+)
                       es_repitente      (bool)
                     Cualquier clave ausente se imputa con mediana global.

    Retorna
    -------
    dict con 15 features normalizadas en [0,1] (o [-1,1] para tendencia).
    """
    comp = comportamiento or {}

    # ── Bloque académico (6 features originales) ─────────────────────────────
    academico = _calcular_academico(evaluaciones, cfg)

    # ── Bloque comportamiento (3 features nuevas) ────────────────────────────
    comportamiento_feats = _calcular_comportamiento(comp)

    # ── Bloque contexto (2 features nuevas) ──────────────────────────────────
    contexto_feats = _calcular_contexto(comp)

    # ── Bloque alertas tempranas (3 features nuevas) ─────────────────────────
    alertas_feats = _calcular_alertas(evaluaciones, comp)

    # ── Imputación global de None ─────────────────────────────────────────────
    todas = {**academico, **comportamiento_feats, **contexto_feats, **alertas_feats}
    vals_num = [v for v in todas.values() if v is not None]
    mediana  = float(np.median(vals_num)) if vals_num else 0.5

    return {k: round(v if v is not None else mediana, 4) for k, v in todas.items()}


# ─────────────────────────────────────────────────────────────────────────────
# BLOQUE 1 — ACADÉMICO (las 6 features originales, sin cambios)
# ─────────────────────────────────────────────────────────────────────────────

def _calcular_academico(
    evaluaciones: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> dict[str, float | None]:

    regulares = [e for e in evaluaciones if not _es_extra(e["slug"])]
    extras    = [e for e in evaluaciones if     _es_extra(e["slug"])]

    for e in regulares + extras:
        e["_norm"] = normalizar(e.get("valor"), e.get("escala_slug", "100"))

    regulares_con_nota = [e for e in regulares if e["_norm"] is not None]
    total_regulares    = max(len(regulares), 1)

    # Feature 1: nota_ponderada
    suma_pesos = sum(e["peso"] for e in regulares_con_nota)
    nota_ponderada = (
        sum(e["_norm"] * e["peso"] for e in regulares_con_nota) / suma_pesos
        if suma_pesos > 0 else None
    )

    # Feature 2: ratio_completadas
    ratio_completadas = len(regulares_con_nota) / total_regulares

    # Feature 3: tendencia (regresión lineal)
    secuenciales = sorted(regulares_con_nota, key=lambda e: e.get("orden", 0))
    if len(secuenciales) >= 2:
        X = np.array([e.get("orden", i+1) for i, e in enumerate(secuenciales)], dtype=float)
        y = np.array([e["_norm"] for e in secuenciales], dtype=float)
        x_m, y_m = X.mean(), y.mean()
        num   = ((X - x_m) * (y - y_m)).sum()
        denom = ((X - x_m) ** 2).sum()
        tendencia = float(np.clip(num / denom if denom != 0 else 0.0, -1.0, 1.0))
    else:
        tendencia = 0.0

    # Feature 4: nota_minima
    nota_minima = (
        min(e["_norm"] for e in regulares_con_nota)
        if regulares_con_nota else None
    )

    # Feature 5: bonus_extra
    bonus_extra = min(
        sum(e["_norm"] for e in extras if e["_norm"] is not None), 0.1
    )

    # Feature 6: ratio_reprobados
    reprobadas = sum(
        1 for e in regulares_con_nota
        if es_aprobado(e.get("valor"), e.get("escala_slug", "100")) is False
    )
    ratio_reprobados = reprobadas / total_regulares

    return {
        "nota_ponderada":    nota_ponderada,
        "ratio_completadas": ratio_completadas,
        "tendencia":         tendencia,
        "nota_minima":       nota_minima,
        "bonus_extra":       bonus_extra,
        "ratio_reprobados":  ratio_reprobados,
    }


# ─────────────────────────────────────────────────────────────────────────────
# BLOQUE 2 — COMPORTAMIENTO (3 features nuevas)
# ─────────────────────────────────────────────────────────────────────────────

def _calcular_comportamiento(comp: dict[str, Any]) -> dict[str, float | None]:
    """
    asistencia          : % de clases asistidas → [0,1]
    ratio_tareas_tiempo : tareas_a_tiempo / tareas_total → [0,1]
    visitas_tutoria     : nro visitas cap en 5 → [0,1]
    """
    # Asistencia: el docente ingresa fracción directa [0-1]
    asistencia = comp.get("asistencia")
    if asistencia is not None:
        asistencia = float(np.clip(asistencia, 0.0, 1.0))

    # Ratio tareas a tiempo
    ta_tiempo = comp.get("tareas_a_tiempo")
    ta_total  = comp.get("tareas_total")
    if ta_tiempo is not None and ta_total is not None and int(ta_total) > 0:
        ratio_tareas_tiempo = float(np.clip(int(ta_tiempo) / int(ta_total), 0.0, 1.0))
    else:
        ratio_tareas_tiempo = None

    # Visitas a tutoría
    visitas = comp.get("visitas_tutoria")
    if visitas is not None:
        visitas_tutoria = float(np.clip(int(visitas) / _MAX_VISITAS_TUTORIA, 0.0, 1.0))
    else:
        visitas_tutoria = None

    return {
        "asistencia":          asistencia,
        "ratio_tareas_tiempo": ratio_tareas_tiempo,
        "visitas_tutoria":     visitas_tutoria,
    }


# ─────────────────────────────────────────────────────────────────────────────
# BLOQUE 3 — CONTEXTO DEL ESTUDIANTE (2 features nuevas)
# ─────────────────────────────────────────────────────────────────────────────

def _calcular_contexto(comp: dict[str, Any]) -> dict[str, float | None]:
    """
    carga_academica : materias_paralelo / 8 → [0,1]
    es_repitente    : 0.0 o 1.0
    """
    materias = comp.get("materias_paralelo")
    carga_academica = (
        float(np.clip(int(materias) / _MAX_MATERIAS, 0.0, 1.0))
        if materias is not None else None
    )

    repitente = comp.get("es_repitente")
    es_repitente = (
        1.0 if repitente else 0.0
        if repitente is not None else None
    )

    return {
        "carga_academica": carga_academica,
        "es_repitente":    es_repitente,
    }


# ─────────────────────────────────────────────────────────────────────────────
# BLOQUE 4 — ALERTAS TEMPRANAS (3 features nuevas)
# ─────────────────────────────────────────────────────────────────────────────

def _calcular_alertas(
    evaluaciones: list[dict[str, Any]],
    comp: dict[str, Any],
) -> dict[str, float | None]:
    """
    caida_brusca        : 1 si alguna nota cae >0.25 respecto a la anterior [0,1]
    varianza_notas      : std de notas norm (inestabilidad) → cap 0.5 → [0,1]
    nota_primer_parcial : nota norm del parcial con orden más bajo [0,1] o None
    alerta_asistencia   : 1 si asistencia < 70% [0,1]
    """
    # Normalizar las notas aquí también, en caso de que _calcular_alertas
    # se llame de forma independiente (sin pasar por _calcular_academico).
    from src.escalas import normalizar as _norm_fn
    for e in evaluaciones:
        if "_norm" not in e:
            e["_norm"] = _norm_fn(e.get("valor"), e.get("escala_slug", "100"))

    # Obtener notas normalizadas de evaluaciones regulares
    regulares_norm = [
        (e.get("orden", 99), e["_norm"])
        for e in evaluaciones
        if not _es_extra(e["slug"]) and e.get("_norm") is not None
    ]
    regulares_norm.sort(key=lambda x: x[0])
    notas_ord = [n for _, n in regulares_norm]

    # Caída brusca
    if len(notas_ord) >= 2:
        caidas = [
            notas_ord[i-1] - notas_ord[i]
            for i in range(1, len(notas_ord))
        ]
        caida_brusca = 1.0 if max(caidas) >= _UMBRAL_CAIDA_BRUSCA else 0.0
    else:
        caida_brusca = None   # no hay suficientes notas todavía

    # Varianza de notas
    if len(notas_ord) >= 2:
        std = float(np.std(notas_ord))
        varianza_notas = float(np.clip(std / 0.5, 0.0, 1.0))  # 0.5 = std máx razonable
    else:
        varianza_notas = None

    # Nota del primer parcial (orden más bajo entre los de slug "parcial")
    parciales = [
        (e.get("orden", 99), e["_norm"])
        for e in evaluaciones
        if e["slug"] == "parcial" and e.get("_norm") is not None
    ]
    nota_primer_parcial = min(parciales, key=lambda x: x[0])[1] if parciales else None

    # Alerta asistencia
    asistencia = comp.get("asistencia")
    if asistencia is not None:
        alerta_asistencia = 1.0 if float(asistencia) < _UMBRAL_ASISTENCIA else 0.0
    else:
        alerta_asistencia = None

    return {
        "caida_brusca":        caida_brusca,
        "varianza_notas":      varianza_notas,
        "nota_primer_parcial": nota_primer_parcial,
        "alerta_asistencia":   alerta_asistencia,
    }


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _es_extra(slug: str) -> bool:
    from src.config import TIPOS_EVALUACION
    return TIPOS_EVALUACION.get(slug, {}).get("es_extra", False)


def evaluar_detalle_variables(
    evaluaciones:   list[dict[str, Any]],
    cfg:            dict[str, Any],
    comportamiento: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Devuelve análisis legible por evaluación para la UI.
    Retorna lista de dicts con:
      nombre, tipo, valor, valor_norm, peso, estado ("ok"|"alerta"|"pendiente")
    """
    comp    = comportamiento or {}
    detalle = []

    for e in evaluaciones:
        valor       = e.get("valor")
        escala_slug = e.get("escala_slug", "100")
        norm        = normalizar(valor, escala_slug)
        es_ex       = _es_extra(e["slug"])

        if valor is None:
            estado = "pendiente"
            msg    = "Aún no rendida"
        elif es_ex:
            estado = "ok"
            msg    = f"+{valor} puntos extra"
        elif es_aprobado(valor, escala_slug):
            estado = "ok"
            msg    = f"Aprobada ({valor} en escala {escala_slug})"
        else:
            estado = "alerta"
            msg    = f"Insuficiente ({valor} en escala {escala_slug})"

        detalle.append({
            "nombre":     e.get("nombre", e["slug"]),
            "tipo":       e["slug"],
            "valor":      valor,
            "valor_norm": norm,
            "peso":       e.get("peso", 0),
            "es_extra":   es_ex,
            "estado":     estado,
            "mensaje":    msg,
        })

    return detalle