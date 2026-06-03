"""
src/config.py  —  Configuración centralizada v5
------------------------------------------------
Añade soporte para evaluaciones dinámicas sin romper
la compatibilidad con el modelo XGBoost ya entrenado.
"""

from pathlib import Path

# ── Rutas ──────────────────────────────────────────────────────────────────
ROOT_DIR        = Path(__file__).resolve().parent.parent
DATA_DIR        = ROOT_DIR / "data"
MODELS_DIR      = ROOT_DIR / "models"

DATA_PATH       = DATA_DIR / "datos_estudiantes.csv"
MODEL_PATH      = MODELS_DIR / "modelo_xgb.pkl"
MODEL_INFO_PATH = MODELS_DIR / "info_modelo.pkl"

# ── Catálogo de tipos de evaluación (se amplía sin tocar el modelo) ─────────
# slug          → clave interna que usa el código
# etiqueta      → texto visible al docente
# es_extra      → True = puntos bonus (peso 0, no entra al promedio base)
# permite_rec   → True = puede tener recuperatorio
TIPOS_EVALUACION = {
    "parcial":      {"etiqueta": "Parcial",           "es_extra": False, "permite_rec": True},
    "proyecto":     {"etiqueta": "Proyecto",           "es_extra": False, "permite_rec": False},
    "tarea":        {"etiqueta": "Tarea / Trabajo",    "es_extra": False, "permite_rec": False},
    "laboratorio":  {"etiqueta": "Laboratorio",        "es_extra": False, "permite_rec": True},
    "examen_final": {"etiqueta": "Examen Final",       "es_extra": False, "permite_rec": False},
    "puntos_extra": {"etiqueta": "Puntos Extra",       "es_extra": True,  "permite_rec": False},
}

# ── Features que el modelo XGBoost espera (NUNCA cambian) ───────────────────
# Estas son las variables NORMALIZADAS que salen de features.py,
# completamente independientes del nombre de cada evaluación.
FEATURES = [
    # ── Bloque académico (6 originales) ───────────────────────────────────
    "nota_ponderada",       # promedio ponderado de evaluaciones regulares [0-1]
    "ratio_completadas",    # fracción de evaluaciones entregadas [0-1]
    "tendencia",            # pendiente regresión lineal sobre notas [-1,1]
    "nota_minima",          # peor nota normalizada registrada [0-1]
    "bonus_extra",          # puntos extra acumulados, cap 10% [0-0.1]
    "ratio_reprobados",     # fracción de evaluaciones con nota < umbral [0-1]
    # ── Bloque comportamiento (3 nuevas) ──────────────────────────────────
    "asistencia",           # % clases asistidas [0-1]
    "ratio_tareas_tiempo",  # tareas entregadas a tiempo / total [0-1]
    "visitas_tutoria",      # visitas al docente, cap 5 → [0-1]
    # ── Bloque contexto (2 nuevas) ────────────────────────────────────────
    "carga_academica",      # materias en paralelo / 8 [0-1]
    "es_repitente",         # 1.0 si ya cursó esta materia antes [0,1]
    # ── Bloque alertas tempranas (3 nuevas) ───────────────────────────────
    "caida_brusca",         # 1 si alguna nota cae >25 pts norm [0,1]
    "varianza_notas",       # std de notas norm, cap 0.5 → [0-1]
    "nota_primer_parcial",  # nota norm del primer parcial [0-1]
    "alerta_asistencia",    # 1 si asistencia < 70% [0,1]
]

# ── Columnas del CSV de entrenamiento (datos sintéticos) ────────────────────
# Se mantienen para compatibilidad con generar_datos.py y train.py actuales
CSV_FEATURES = [
    "nota_parcial_1",
    "nota_parcial_2",
    "porcentaje_asistencia",
    "tareas_entregadas",
    "horas_estudio_semanal",
    "participa_tutoria",
    "promedio_ciclo_anterior",
    "nivel_socioeconomico",
    "acceso_internet",
]
NUMERIC_COLS = [
    "nota_parcial_1", "nota_parcial_2", "porcentaje_asistencia",
    "tareas_entregadas", "horas_estudio_semanal", "promedio_ciclo_anterior",
]
BINARY_COLS  = ["participa_tutoria", "acceso_internet"]
CATEG_COLS   = ["nivel_socioeconomico"]
TARGET       = "riesgo_reprobacion"

# ── Etiquetas legibles ───────────────────────────────────────────────────────
LABEL_MAP     = {0: "SIN RIESGO", 1: "CON RIESGO"}
NIVEL_SOC_MAP = {0: "Bajo", 1: "Medio", 2: "Alto"}
BINARIO_MAP   = {0: "No", 1: "Sí"}

# ── Catálogo de escalas soportadas ──────────────────────────────────────────
# Cada escala define cómo convertir un valor crudo a [0.0, 1.0]
# "tipo": "numerico" | "letra" | "porcentaje"
ESCALAS = {
    "100":  {"tipo": "numerico",    "maximo": 100,  "etiqueta": "Sobre 100"},
    "70":   {"tipo": "numerico",    "maximo": 70,   "etiqueta": "Sobre 70"},
    "20":   {"tipo": "numerico",    "maximo": 20,   "etiqueta": "Sobre 20"},
    "10":   {"tipo": "numerico",    "maximo": 10,   "etiqueta": "Sobre 10"},
    "pct":  {"tipo": "porcentaje",  "maximo": 100,  "etiqueta": "Porcentaje (0-100%)"},
    "letra":{"tipo": "letra",       "maximo": None, "etiqueta": "Letras (A-F)"},
}

# Tabla de conversión de letras a [0,1]
LETRAS_A_DECIMAL = {
    "A+": 1.00, "A": 0.95, "A-": 0.90,
    "B+": 0.85, "B": 0.80, "B-": 0.75,
    "C+": 0.70, "C": 0.65, "C-": 0.60,
    "D+": 0.55, "D": 0.50, "D-": 0.45,
    "F":  0.00,
}

# Umbral de aprobación por escala (en valor crudo)
UMBRAL_APROBACION_POR_ESCALA = {
    "100":   51,
    "70":    36,   # 51% de 70
    "20":    10,
    "10":    5,
    "pct":   51,
    "letra": "D",  # D o superior = aprobado
}

# ── Hiperparámetros de entrenamiento ────────────────────────────────────────
RANDOM_STATE    = 42
TEST_SIZE       = 0.20
CV_FOLDS        = 5
DECISION_UMBRAL = 0.45

FEATURE_LABELS = {
    "nota_ponderada":       "Nota ponderada",
    "ratio_completadas":    "Evaluaciones completadas",
    "tendencia":            "Tendencia",
    "nota_minima":          "Nota mínima",
    "bonus_extra":          "Puntos extra",
    "ratio_reprobados":     "Ratio reprobados",
    "asistencia":           "Asistencia",
    "ratio_tareas_tiempo":  "Tareas a tiempo",
    "visitas_tutoria":      "Visitas a tutoría",
    "carga_academica":      "Carga académica",
    "es_repitente":         "Es repitente",
    "caida_brusca":         "Caída brusca de notas",
    "varianza_notas":       "Varianza de notas",
    "nota_primer_parcial":  "Nota primer parcial",
    "alerta_asistencia":    "Alerta de asistencia",
}