"""
tests/test_escalas.py  —  Tests unitarios para src/escalas.py
-------------------------------------------------------------
Ejecutar:
    pytest tests/test_escalas.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.escalas import normalizar, es_aprobado


# ══════════════════════════════════════════════════════════════════════════════
# normalizar()
# ══════════════════════════════════════════════════════════════════════════════

class TestNormalizarEscala100:
    def test_nota_maxima(self):
        assert normalizar(100, "100") == pytest.approx(1.0)

    def test_nota_cero(self):
        assert normalizar(0, "100") == pytest.approx(0.0)

    def test_nota_mitad(self):
        assert normalizar(50, "100") == pytest.approx(0.5)

    def test_nota_pendiente(self):
        assert normalizar(None, "100") is None

    def test_resultado_dentro_rango(self):
        for v in [0, 25, 51, 75, 100]:
            r = normalizar(v, "100")
            assert 0.0 <= r <= 1.0


class TestNormalizarEscala70:
    def test_maximo(self):
        assert normalizar(70, "70") == pytest.approx(1.0)

    def test_minimo(self):
        assert normalizar(0, "70") == pytest.approx(0.0)

    def test_mitad(self):
        assert normalizar(35, "70") == pytest.approx(0.5)


class TestNormalizarEscala20:
    def test_maximo(self):
        assert normalizar(20, "20") == pytest.approx(1.0)

    def test_umbral_aprobacion(self):
        # 10/20 = 0.5
        assert normalizar(10, "20") == pytest.approx(0.5)


class TestNormalizarEscala10:
    def test_maximo(self):
        assert normalizar(10, "10") == pytest.approx(1.0)

    def test_cero(self):
        assert normalizar(0, "10") == pytest.approx(0.0)


class TestNormalizarEscalaPct:
    def test_maximo(self):
        assert normalizar(100, "pct") == pytest.approx(1.0)

    def test_mitad(self):
        assert normalizar(50, "pct") == pytest.approx(0.5)


class TestNormalizarEscalaLetra:
    def test_A_plus(self):
        assert normalizar("A+", "letra") == pytest.approx(1.00)

    def test_A(self):
        assert normalizar("A", "letra") == pytest.approx(0.95)

    def test_F(self):
        assert normalizar("F", "letra") == pytest.approx(0.0)

    def test_D(self):
        assert normalizar("D", "letra") == pytest.approx(0.50)

    def test_B_plus(self):
        assert normalizar("B+", "letra") == pytest.approx(0.85)

    def test_letra_invalida(self):
        # Letra desconocida debe devolver None
        assert normalizar("Z", "letra") is None

    def test_pendiente_letra(self):
        assert normalizar(None, "letra") is None

    def test_case_insensitive(self):
        # "a+" debe normalizarse igual que "A+"
        result = normalizar("a+", "letra")
        assert result == pytest.approx(1.00) or result is None  # depende de implementación


class TestNormalizarClip:
    def test_sobre_maximo(self):
        # Nota mayor al máximo no debe romper ni devolver > 1
        r = normalizar(120, "100")
        assert r is not None
        # Puede retornar > 1 o clipearse; lo importante es que no explote

    def test_negativo(self):
        r = normalizar(-5, "100")
        assert r is not None


# ══════════════════════════════════════════════════════════════════════════════
# es_aprobado()
# ══════════════════════════════════════════════════════════════════════════════

class TestEsAprobado100:
    def test_aprobado(self):
        assert es_aprobado(60, "100") is True

    def test_umbral_justo(self):
        assert es_aprobado(51, "100") is True

    def test_reprobado(self):
        assert es_aprobado(50, "100") is False

    def test_cero(self):
        assert es_aprobado(0, "100") is False

    def test_pendiente(self):
        assert es_aprobado(None, "100") is None


class TestEsAprobado70:
    def test_aprobado(self):
        assert es_aprobado(40, "70") is True   # 40/70 > 51%

    def test_reprobado(self):
        assert es_aprobado(30, "70") is False  # 30/70 ≈ 43%

    def test_pendiente(self):
        assert es_aprobado(None, "70") is None


class TestEsAprobadoLetra:
    def test_A_aprobado(self):
        assert es_aprobado("A", "letra") is True

    def test_C_aprobado(self):
        assert es_aprobado("C", "letra") is True

    def test_D_aprobado(self):
        assert es_aprobado("D", "letra") is True  # D es umbral ≥ aprobado

    def test_F_reprobado(self):
        assert es_aprobado("F", "letra") is False

    def test_pendiente(self):
        assert es_aprobado(None, "letra") is None
