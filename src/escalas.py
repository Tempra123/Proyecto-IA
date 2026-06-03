"""
src/escalas.py  —  Conversor universal de escalas
--------------------------------------------------
Convierte cualquier nota (numérica, porcentaje, letra)
a un valor normalizado en [0.0, 1.0].

Regla de oro: el modelo NUNCA recibe el valor crudo.
Siempre recibe el valor normalizado independiente de la escala original.
"""

from __future__ import annotations
import numpy as np
from src.config import ESCALAS, LETRAS_A_DECIMAL, UMBRAL_APROBACION_POR_ESCALA


def normalizar(valor: float | str | None, escala_slug: str) -> float | None:
    """
    Convierte un valor crudo a [0.0, 1.0] según la escala indicada.
    Devuelve None si el valor es None (evaluación pendiente).

    Parámetros
    ----------
    valor       : nota cruda — puede ser float, int, str ("A", "B+") o None
    escala_slug : clave del catálogo ESCALAS ("100", "70", "letra", "pct", etc.)

    Ejemplos
    --------
    normalizar(75,   "100")   → 0.75
    normalizar(50,   "70")    → 0.714
    normalizar("B+", "letra") → 0.85
    normalizar(85,   "pct")   → 0.85
    normalizar(None, "100")   → None
    """
    if valor is None:
        return None

    cfg = ESCALAS.get(escala_slug)
    if cfg is None:
        raise ValueError(f"Escala desconocida: '{escala_slug}'. "
                         f"Opciones: {list(ESCALAS.keys())}")

    tipo = cfg["tipo"]

    if tipo == "letra":
        key = str(valor).strip().upper()
        if key not in LETRAS_A_DECIMAL:
            raise ValueError(f"Letra no reconocida: '{valor}'. "
                             f"Válidas: {list(LETRAS_A_DECIMAL.keys())}")
        return LETRAS_A_DECIMAL[key]

    # Numérico o porcentaje
    try:
        v = float(valor)
    except (TypeError, ValueError):
        raise ValueError(f"No se puede convertir '{valor}' a número para escala '{escala_slug}'")

    maximo = float(cfg["maximo"])
    return float(np.clip(v / maximo, 0.0, 1.0))


def es_aprobado(valor: float | str | None, escala_slug: str) -> bool | None:
    """
    Devuelve True/False/None según si la nota aprueba.
    None = evaluación pendiente.
    """
    if valor is None:
        return None

    umbral_crudo = UMBRAL_APROBACION_POR_ESCALA.get(escala_slug)
    cfg = ESCALAS.get(escala_slug, {})

    if cfg.get("tipo") == "letra":
        # Para letras comparamos posición en la tabla
        orden = list(LETRAS_A_DECIMAL.keys())
        try:
            idx_valor  = orden.index(str(valor).strip().upper())
            idx_umbral = orden.index(str(umbral_crudo).strip().upper())
            return idx_valor <= idx_umbral   # menor índice = mejor nota
        except ValueError:
            return None

    # Numérico
    try:
        return float(valor) >= float(umbral_crudo)
    except (TypeError, ValueError):
        return None


def validar_valor(valor: float | str | None, escala_slug: str) -> str | None:
    """
    Valida que el valor sea coherente con la escala.
    Retorna mensaje de error (str) o None si es válido.
    """
    if valor is None:
        return None  # pendiente es válido

    cfg = ESCALAS.get(escala_slug)
    if cfg is None:
        return f"Escala '{escala_slug}' no existe"

    if cfg["tipo"] == "letra":
        if str(valor).strip().upper() not in LETRAS_A_DECIMAL:
            return f"Letra inválida: '{valor}'"
        return None

    try:
        v = float(valor)
    except (TypeError, ValueError):
        return f"'{valor}' no es un número válido para escala '{escala_slug}'"

    maximo = float(cfg["maximo"])
    if v < 0 or v > maximo:
        return f"Valor {v} fuera de rango [0, {maximo:.0f}]"

    return None
