"""
tests/test_features.py  —  Tests unitarios para src/features.py
---------------------------------------------------------------
Ejecutar:
    pytest tests/test_features.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.features import calcular_features, evaluar_detalle_variables
from src.config   import FEATURES


# ── Fixtures reutilizables ────────────────────────────────────────────────────

@pytest.fixture
def cfg_base():
    return {"nota_minima_aprobacion": 51, "escala_maxima": 100}


@pytest.fixture
def evaluaciones_tipo_a():
    return [
        {"slug": "parcial",  "nombre": "Parcial 1", "peso": 25, "valor": 80,  "orden": 1, "escala_slug": "100"},
        {"slug": "parcial",  "nombre": "Parcial 2", "peso": 25, "valor": 70,  "orden": 2, "escala_slug": "100"},
        {"slug": "proyecto", "nombre": "Proyecto",  "peso": 40, "valor": 85,  "orden": 3, "escala_slug": "100"},
        {"slug": "tarea",    "nombre": "Tarea",     "peso": 10, "valor": 90,  "orden": 4, "escala_slug": "100"},
    ]


@pytest.fixture
def evaluaciones_con_pendiente():
    return [
        {"slug": "parcial",  "nombre": "Parcial 1", "peso": 50, "valor": 60,  "orden": 1, "escala_slug": "100"},
        {"slug": "parcial",  "nombre": "Parcial 2", "peso": 50, "valor": None, "orden": 2, "escala_slug": "100"},
    ]


@pytest.fixture
def evaluaciones_con_extra():
    return [
        {"slug": "parcial",      "nombre": "Parcial 1",  "peso": 100, "valor": 70,  "orden": 1, "escala_slug": "100"},
        {"slug": "puntos_extra", "nombre": "Bonus",       "peso": 0,   "valor": 8,   "orden": 99,"escala_slug": "100"},
    ]


@pytest.fixture
def evaluaciones_letras():
    return [
        {"slug": "proyecto", "nombre": "Proyecto", "peso": 60, "valor": "A-", "orden": 1, "escala_slug": "letra"},
        {"slug": "tarea",    "nombre": "Tarea 1",  "peso": 20, "valor": 75,   "orden": 2, "escala_slug": "100"},
        {"slug": "tarea",    "nombre": "Tarea 2",  "peso": 20, "valor": 80,   "orden": 3, "escala_slug": "100"},
    ]


@pytest.fixture
def comportamiento_normal():
    return {
        "asistencia":        0.90,
        "tareas_a_tiempo":   4,
        "tareas_total":      5,
        "visitas_tutoria":   2,
        "materias_paralelo": 4,
        "es_repitente":      False,
    }


@pytest.fixture
def comportamiento_riesgo():
    return {
        "asistencia":        0.55,
        "tareas_a_tiempo":   1,
        "tareas_total":      5,
        "visitas_tutoria":   0,
        "materias_paralelo": 7,
        "es_repitente":      True,
    }


# ══════════════════════════════════════════════════════════════════════════════
# CONTRATO FUNDAMENTAL: siempre devuelve 15 features sin None
# ══════════════════════════════════════════════════════════════════════════════

class TestContratoFundamental:
    def test_devuelve_exactamente_15_features(self, evaluaciones_tipo_a, cfg_base):
        feats = calcular_features(evaluaciones_tipo_a, cfg_base)
        assert len(feats) == 15

    def test_claves_son_las_15_canonicas(self, evaluaciones_tipo_a, cfg_base):
        feats = calcular_features(evaluaciones_tipo_a, cfg_base)
        assert set(feats.keys()) == set(FEATURES)

    def test_ninguna_feature_es_none(self, evaluaciones_tipo_a, cfg_base):
        feats = calcular_features(evaluaciones_tipo_a, cfg_base)
        for k, v in feats.items():
            assert v is not None, f"Feature '{k}' es None"

    def test_ninguna_feature_es_nan(self, evaluaciones_tipo_a, cfg_base):
        import math
        feats = calcular_features(evaluaciones_tipo_a, cfg_base)
        for k, v in feats.items():
            assert not math.isnan(v), f"Feature '{k}' es NaN"

    def test_sin_comportamiento_no_rompe(self, evaluaciones_tipo_a, cfg_base):
        feats = calcular_features(evaluaciones_tipo_a, cfg_base, comportamiento=None)
        assert len(feats) == 15
        for v in feats.values():
            assert v is not None

    def test_evaluaciones_todas_pendientes(self, cfg_base):
        ev = [
            {"slug": "parcial", "nombre": "P1", "peso": 50, "valor": None, "orden": 1, "escala_slug": "100"},
            {"slug": "parcial", "nombre": "P2", "peso": 50, "valor": None, "orden": 2, "escala_slug": "100"},
        ]
        feats = calcular_features(ev, cfg_base)
        assert len(feats) == 15
        for v in feats.values():
            assert v is not None


# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE ACADÉMICO
# ══════════════════════════════════════════════════════════════════════════════

class TestBloqueAcademico:
    def test_nota_ponderada_rango(self, evaluaciones_tipo_a, cfg_base):
        feats = calcular_features(evaluaciones_tipo_a, cfg_base)
        assert 0.0 <= feats["nota_ponderada"] <= 1.0

    def test_ratio_completadas_todas(self, evaluaciones_tipo_a, cfg_base):
        feats = calcular_features(evaluaciones_tipo_a, cfg_base)
        assert feats["ratio_completadas"] == pytest.approx(1.0)

    def test_ratio_completadas_con_pendiente(self, evaluaciones_con_pendiente, cfg_base):
        feats = calcular_features(evaluaciones_con_pendiente, cfg_base)
        assert feats["ratio_completadas"] == pytest.approx(0.5)

    def test_tendencia_rango(self, evaluaciones_tipo_a, cfg_base):
        feats = calcular_features(evaluaciones_tipo_a, cfg_base)
        assert -1.0 <= feats["tendencia"] <= 1.0

    def test_tendencia_sola_evaluacion(self, cfg_base):
        ev = [{"slug": "parcial", "nombre": "P1", "peso": 100, "valor": 75, "orden": 1, "escala_slug": "100"}]
        feats = calcular_features(ev, cfg_base)
        assert feats["tendencia"] == pytest.approx(0.0)

    def test_tendencia_bajando(self, cfg_base):
        ev = [
            {"slug": "parcial", "nombre": "P1", "peso": 50, "valor": 90, "orden": 1, "escala_slug": "100"},
            {"slug": "parcial", "nombre": "P2", "peso": 50, "valor": 40, "orden": 2, "escala_slug": "100"},
        ]
        feats = calcular_features(ev, cfg_base)
        assert feats["tendencia"] < 0

    def test_bonus_extra_cap(self, evaluaciones_con_extra, cfg_base):
        feats = calcular_features(evaluaciones_con_extra, cfg_base)
        assert feats["bonus_extra"] <= 0.1

    def test_bonus_extra_sin_puntos(self, evaluaciones_tipo_a, cfg_base):
        feats = calcular_features(evaluaciones_tipo_a, cfg_base)
        assert feats["bonus_extra"] == pytest.approx(0.0)

    def test_ratio_reprobados_todos_aprobados(self, evaluaciones_tipo_a, cfg_base):
        feats = calcular_features(evaluaciones_tipo_a, cfg_base)
        assert feats["ratio_reprobados"] == pytest.approx(0.0)

    def test_ratio_reprobados_todos_reprobados(self, cfg_base):
        ev = [
            {"slug": "parcial", "nombre": "P1", "peso": 50, "valor": 40, "orden": 1, "escala_slug": "100"},
            {"slug": "parcial", "nombre": "P2", "peso": 50, "valor": 30, "orden": 2, "escala_slug": "100"},
        ]
        feats = calcular_features(ev, cfg_base)
        assert feats["ratio_reprobados"] == pytest.approx(1.0)

    def test_escala_letras(self, evaluaciones_letras, cfg_base):
        feats = calcular_features(evaluaciones_letras, cfg_base)
        assert feats["nota_ponderada"] > 0.5  # A- es aprobado

    def test_escalas_mixtas_tipo_b(self, cfg_base):
        ev = [
            {"slug": "parcial",     "nombre": "P1",  "peso": 30, "valor": 50, "orden": 1, "escala_slug": "70"},
            {"slug": "parcial",     "nombre": "P2",  "peso": 30, "valor": 45, "orden": 2, "escala_slug": "70"},
            {"slug": "parcial",     "nombre": "P3",  "peso": 30, "valor": 55, "orden": 3, "escala_slug": "70"},
            {"slug": "laboratorio", "nombre": "Lab", "peso": 10, "valor": 15, "orden": 4, "escala_slug": "20"},
        ]
        feats = calcular_features(ev, cfg_base)
        assert 0.0 <= feats["nota_ponderada"] <= 1.0


# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE COMPORTAMIENTO
# ══════════════════════════════════════════════════════════════════════════════

class TestBloqueComportamiento:
    def test_asistencia_pasada_directa(self, evaluaciones_tipo_a, cfg_base, comportamiento_normal):
        feats = calcular_features(evaluaciones_tipo_a, cfg_base, comportamiento_normal)
        assert feats["asistencia"] == pytest.approx(0.90)

    def test_asistencia_clip_maximo(self, evaluaciones_tipo_a, cfg_base):
        comp = {"asistencia": 1.5}  # valor inválido > 1
        feats = calcular_features(evaluaciones_tipo_a, cfg_base, comp)
        assert feats["asistencia"] <= 1.0

    def test_ratio_tareas_tiempo(self, evaluaciones_tipo_a, cfg_base, comportamiento_normal):
        feats = calcular_features(evaluaciones_tipo_a, cfg_base, comportamiento_normal)
        assert feats["ratio_tareas_tiempo"] == pytest.approx(0.8)  # 4/5

    def test_ratio_tareas_sin_tareas(self, evaluaciones_tipo_a, cfg_base):
        comp = {"tareas_a_tiempo": 0, "tareas_total": 0}
        feats = calcular_features(evaluaciones_tipo_a, cfg_base, comp)
        assert feats["ratio_tareas_tiempo"] is not None  # imputado, no None

    def test_visitas_tutoria_cap(self, evaluaciones_tipo_a, cfg_base):
        comp = {"visitas_tutoria": 10}  # más del cap 5
        feats = calcular_features(evaluaciones_tipo_a, cfg_base, comp)
        assert feats["visitas_tutoria"] == pytest.approx(1.0)

    def test_visitas_tutoria_cero(self, evaluaciones_tipo_a, cfg_base, comportamiento_riesgo):
        feats = calcular_features(evaluaciones_tipo_a, cfg_base, comportamiento_riesgo)
        assert feats["visitas_tutoria"] == pytest.approx(0.0)


# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE CONTEXTO
# ══════════════════════════════════════════════════════════════════════════════

class TestBloqueContexto:
    def test_es_repitente_true(self, evaluaciones_tipo_a, cfg_base, comportamiento_riesgo):
        feats = calcular_features(evaluaciones_tipo_a, cfg_base, comportamiento_riesgo)
        assert feats["es_repitente"] == pytest.approx(1.0)

    def test_es_repitente_false(self, evaluaciones_tipo_a, cfg_base, comportamiento_normal):
        feats = calcular_features(evaluaciones_tipo_a, cfg_base, comportamiento_normal)
        assert feats["es_repitente"] == pytest.approx(0.0)

    def test_carga_academica_cap(self, evaluaciones_tipo_a, cfg_base):
        comp = {"materias_paralelo": 10}  # más del cap 8
        feats = calcular_features(evaluaciones_tipo_a, cfg_base, comp)
        assert feats["carga_academica"] == pytest.approx(1.0)

    def test_carga_academica_minima(self, evaluaciones_tipo_a, cfg_base):
        comp = {"materias_paralelo": 1}
        feats = calcular_features(evaluaciones_tipo_a, cfg_base, comp)
        assert feats["carga_academica"] == pytest.approx(1/8)


# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE ALERTAS TEMPRANAS
# ══════════════════════════════════════════════════════════════════════════════

class TestBloqueAlertas:
    def test_alerta_asistencia_baja(self, evaluaciones_tipo_a, cfg_base, comportamiento_riesgo):
        feats = calcular_features(evaluaciones_tipo_a, cfg_base, comportamiento_riesgo)
        assert feats["alerta_asistencia"] == pytest.approx(1.0)  # 55% < 70%

    def test_sin_alerta_asistencia_alta(self, evaluaciones_tipo_a, cfg_base, comportamiento_normal):
        feats = calcular_features(evaluaciones_tipo_a, cfg_base, comportamiento_normal)
        assert feats["alerta_asistencia"] == pytest.approx(0.0)  # 90% >= 70%

    def test_caida_brusca_detectada(self, cfg_base):
        ev = [
            {"slug": "parcial", "nombre": "P1", "peso": 50, "valor": 90, "orden": 1, "escala_slug": "100"},
            {"slug": "parcial", "nombre": "P2", "peso": 50, "valor": 50, "orden": 2, "escala_slug": "100"},
        ]
        feats = calcular_features(ev, cfg_base)
        # Caída de 0.9 a 0.5 = 0.4 > umbral 0.25
        assert feats["caida_brusca"] == pytest.approx(1.0)

    def test_sin_caida_brusca(self, cfg_base):
        ev = [
            {"slug": "parcial", "nombre": "P1", "peso": 50, "valor": 80, "orden": 1, "escala_slug": "100"},
            {"slug": "parcial", "nombre": "P2", "peso": 50, "valor": 75, "orden": 2, "escala_slug": "100"},
        ]
        feats = calcular_features(ev, cfg_base)
        assert feats["caida_brusca"] == pytest.approx(0.0)

    def test_nota_primer_parcial(self, cfg_base):
        ev = [
            {"slug": "parcial",  "nombre": "P1", "peso": 30, "valor": 80, "orden": 1, "escala_slug": "100"},
            {"slug": "parcial",  "nombre": "P2", "peso": 30, "valor": 90, "orden": 2, "escala_slug": "100"},
            {"slug": "proyecto", "nombre": "Py", "peso": 40, "valor": 70, "orden": 3, "escala_slug": "100"},
        ]
        feats = calcular_features(ev, cfg_base)
        # El primer parcial es P1 con 80/100 = 0.8
        assert feats["nota_primer_parcial"] == pytest.approx(0.8)

    def test_nota_primer_parcial_sin_parcial(self, evaluaciones_letras, cfg_base):
        # evaluaciones_letras no tiene slug "parcial", debería imputarse
        feats = calcular_features(evaluaciones_letras, cfg_base)
        assert feats["nota_primer_parcial"] is not None  # imputado

    def test_varianza_notas_notas_iguales(self, cfg_base):
        ev = [
            {"slug": "parcial", "nombre": "P1", "peso": 50, "valor": 70, "orden": 1, "escala_slug": "100"},
            {"slug": "parcial", "nombre": "P2", "peso": 50, "valor": 70, "orden": 2, "escala_slug": "100"},
        ]
        feats = calcular_features(ev, cfg_base)
        assert feats["varianza_notas"] == pytest.approx(0.0)


# ══════════════════════════════════════════════════════════════════════════════
# evaluar_detalle_variables()
# ══════════════════════════════════════════════════════════════════════════════

class TestEvaluarDetalle:
    def test_devuelve_lista(self, evaluaciones_tipo_a, cfg_base):
        det = evaluar_detalle_variables(evaluaciones_tipo_a, cfg_base)
        assert isinstance(det, list)
        assert len(det) == len(evaluaciones_tipo_a)

    def test_estructura_item(self, evaluaciones_tipo_a, cfg_base):
        det = evaluar_detalle_variables(evaluaciones_tipo_a, cfg_base)
        item = det[0]
        for clave in ["nombre", "tipo", "valor", "valor_norm", "peso", "estado", "mensaje"]:
            assert clave in item, f"Falta clave '{clave}' en detalle"

    def test_estado_pendiente(self, evaluaciones_con_pendiente, cfg_base):
        det = evaluar_detalle_variables(evaluaciones_con_pendiente, cfg_base)
        pendiente = next(d for d in det if d["valor"] is None)
        assert pendiente["estado"] == "pendiente"

    def test_estado_ok_aprobado(self, evaluaciones_tipo_a, cfg_base):
        det = evaluar_detalle_variables(evaluaciones_tipo_a, cfg_base)
        # Parcial 1 con 80 debe estar ok
        p1 = next(d for d in det if d["nombre"] == "Parcial 1")
        assert p1["estado"] == "ok"

    def test_estado_alerta_reprobado(self, cfg_base):
        ev = [{"slug": "parcial", "nombre": "P1", "peso": 100, "valor": 30, "orden": 1, "escala_slug": "100"}]
        det = evaluar_detalle_variables(ev, cfg_base)
        assert det[0]["estado"] == "alerta"

    def test_extra_estado_ok(self, evaluaciones_con_extra, cfg_base):
        det = evaluar_detalle_variables(evaluaciones_con_extra, cfg_base)
        bonus = next(d for d in det if d["tipo"] == "puntos_extra")
        assert bonus["estado"] == "ok"
        assert bonus["es_extra"] is True
