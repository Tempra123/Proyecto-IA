"""
scripts/generar_datos.py  —  Dataset con features dinámicas  v2
----------------------------------------------------------------
Genera un dataset sintético usando las 15 features normalizadas
en lugar de las 9 variables originales del CSV.

Esto permite entrenar el modelo nuevo (flexible) que acepta
cualquier estructura de evaluaciones.

Ejecutar UNA SOLA VEZ:
    python -m scripts.generar_datos --version dinamico

Para mantener compatibilidad con el modelo anterior:
    python -m scripts.generar_datos --version original
"""

import sys
import argparse
import numpy  as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config   import DATA_DIR, DATA_PATH
from src.features import calcular_features

np.random.seed(42)
N = 2000


def generar_original():
    """Dataset con las 9 variables del modelo original."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    nota1      = np.random.normal(60, 18, N).clip(0, 100)
    nota2      = (nota1 * 0.55 + np.random.normal(0, 16, N)).clip(0, 100)
    asistencia = (np.random.beta(7, 2, N) * 100).clip(0, 100)
    tareas     = np.random.randint(0, 11, N)
    horas_est  = np.random.exponential(4.5, N).clip(0, 20)
    tutoria    = np.random.binomial(1, 0.30, N)
    prom_ant   = np.random.normal(62, 15, N).clip(0, 100)
    nivel_soc  = np.random.choice([0, 1, 2], N, p=[0.25, 0.55, 0.20])
    internet   = np.random.binomial(1, 0.75, N)

    nota_final = (
        0.35 * nota1 + 0.35 * nota2 + 0.10 * asistencia
        + 0.08 * (tareas / 10 * 100) + 0.07 * (horas_est / 20 * 100)
        + 0.05 * prom_ant - 2.0 * (nivel_soc == 0).astype(float)
        + 1.5 * tutoria + 0.5 * internet + np.random.normal(0, 4, N)
    ).clip(0, 100)

    riesgo = (nota_final < 51).astype(int)

    df = pd.DataFrame({
        "nota_parcial_1":          nota1.round(1),
        "nota_parcial_2":          nota2.round(1),
        "porcentaje_asistencia":   asistencia.round(1),
        "tareas_entregadas":       tareas,
        "horas_estudio_semanal":   horas_est.round(1),
        "participa_tutoria":       tutoria,
        "promedio_ciclo_anterior": prom_ant.round(1),
        "nivel_socioeconomico":    nivel_soc,
        "acceso_internet":         internet,
        "riesgo_reprobacion":      riesgo,
    })

    for col in ["horas_estudio_semanal", "promedio_ciclo_anterior", "porcentaje_asistencia"]:
        df.loc[np.random.random(N) < 0.04, col] = np.nan

    df.to_csv(DATA_PATH, index=False)
    _print_stats("original (9 vars)", riesgo, DATA_PATH)


def generar_dinamico():
    """
    Dataset con las 15 features normalizadas (6 académicas + 9 nuevas).
    Simula materias con estructuras de evaluación distintas y datos
    de comportamiento/contexto/alertas para cada estudiante.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path_dinamico = DATA_DIR / "datos_dinamicos.csv"
    registros = []

    LETRAS       = ["A+","A","A-","B+","B","B-","C+","C","C-","D+","D","D-","F"]
    PESOS_LETRA  = [0.04,0.08,0.08,0.10,0.12,0.10,0.10,0.10,0.08,0.06,0.06,0.05,0.03]

    for i in range(N):
        # ── Perfil base del estudiante ─────────────────────────────────────
        # Rendimiento base correlaciona con asistencia y contexto
        es_repitente      = np.random.random() < 0.20          # 20% repitentes
        materias_paralelo = np.random.randint(2, 8)
        # Repitentes y carga alta tienden a peor rendimiento base
        penalty  = (0.08 if es_repitente else 0.0) + (materias_paralelo - 4) * 0.02
        base_mu  = float(np.clip(62 - penalty * 10, 40, 75))

        # Asistencia correlacionada con rendimiento (mejor asistencia → mejor nota)
        asistencia_raw = float(np.clip(np.random.beta(7, 2) + np.random.normal(0, 0.05), 0, 1))
        # Penalizar rendimiento si poca asistencia
        if asistencia_raw < 0.70:
            base_mu -= 8

        # Tareas a tiempo (correlacionadas con responsabilidad)
        tareas_total  = np.random.randint(3, 8)
        prob_tiempo   = float(np.clip(asistencia_raw + np.random.normal(0, 0.1), 0.2, 1.0))
        tareas_tiempo = int(np.round(tareas_total * prob_tiempo))
        tareas_tiempo = min(tareas_tiempo, tareas_total)

        # Visitas a tutoría (más visitas si en riesgo — buscando ayuda)
        # (se simula después de ver notas, correlación leve)
        visitas_base = np.random.poisson(1.2)

        cfg = {"nota_minima_aprobacion": 51, "escala_maxima": 100}

        # ── Estructura de evaluaciones por tipo de materia ─────────────────
        tipo_materia = np.random.choice(["A", "B", "C"], p=[0.5, 0.3, 0.2])

        if tipo_materia == "A":
            notas = np.random.normal(base_mu, 18, 4).clip(0, 100)
            ev = [
                {"slug": "parcial",  "peso": 25, "valor": notas[0], "orden": 1, "nombre": "Parcial 1",  "escala_slug": "100"},
                {"slug": "parcial",  "peso": 25, "valor": notas[1], "orden": 2, "nombre": "Parcial 2",  "escala_slug": "100"},
                {"slug": "proyecto", "peso": 40, "valor": notas[2], "orden": 3, "nombre": "Proyecto",   "escala_slug": "100"},
                {"slug": "tarea",    "peso": 10, "valor": notas[3], "orden": 4, "nombre": "Tarea",      "escala_slug": "100"},
            ]
        elif tipo_materia == "B":
            notas_p = np.random.normal(base_mu * 0.7, 14, 3).clip(0, 70)
            nota_l  = np.random.normal(base_mu * 0.2, 5, 1).clip(0, 20)[0]
            ev = [
                {"slug": "parcial",     "peso": 30, "valor": notas_p[0], "orden": 1, "nombre": "Parcial 1", "escala_slug": "70"},
                {"slug": "parcial",     "peso": 30, "valor": notas_p[1], "orden": 2, "nombre": "Parcial 2", "escala_slug": "70"},
                {"slug": "parcial",     "peso": 30, "valor": notas_p[2], "orden": 3, "nombre": "Parcial 3", "escala_slug": "70"},
                {"slug": "laboratorio", "peso": 10, "valor": nota_l,     "orden": 4, "nombre": "Lab Final", "escala_slug": "20"},
            ]
        else:
            # Rendimiento base → letra aproximada
            pct_base = base_mu / 100
            idx_letra = int(np.clip((1 - pct_base) * len(LETRAS), 0, len(LETRAS)-1))
            # Agregar ruido al índice
            idx_letra = int(np.clip(idx_letra + np.random.randint(-2, 3), 0, len(LETRAS)-1))
            letra_proy = LETRAS[idx_letra]
            notas_t    = np.random.normal(base_mu, 16, 2).clip(0, 100)
            ev = [
                {"slug": "proyecto", "peso": 60, "valor": letra_proy, "orden": 1, "nombre": "Proyecto Final", "escala_slug": "letra"},
                {"slug": "tarea",    "peso": 20, "valor": notas_t[0], "orden": 2, "nombre": "Tarea 1",        "escala_slug": "100"},
                {"slug": "tarea",    "peso": 20, "valor": notas_t[1], "orden": 3, "nombre": "Tarea 2",        "escala_slug": "100"},
            ]

        # 5% de probabilidad de tener puntos extra
        if np.random.random() < 0.05:
            ev.append({"slug": "puntos_extra", "peso": 0, "valor": float(np.random.uniform(2, 8)),
                       "orden": 99, "nombre": "Bonus", "escala_slug": "100"})

        # ~8% de evaluaciones pendientes
        for e in ev:
            if e["slug"] != "puntos_extra" and np.random.random() < 0.08:
                e["valor"] = None

        # Comportamiento para esta instancia
        comp = {
            "asistencia":        asistencia_raw,
            "tareas_a_tiempo":   int(tareas_tiempo),
            "tareas_total":      int(tareas_total),
            "visitas_tutoria":   int(visitas_base),
            "materias_paralelo": int(materias_paralelo),
            "es_repitente":      bool(es_repitente),
        }

        feats  = calcular_features(ev, cfg, comp)

        # Repitentes con asistencia baja tienen más riesgo
        umbral_riesgo = 0.51
        riesgo = int(feats["nota_ponderada"] < umbral_riesgo)

        registros.append({
            **feats,
            "tipo_materia":      tipo_materia,
            "riesgo_reprobacion": riesgo,
        })

    df = pd.DataFrame(registros)
    df.to_csv(path_dinamico, index=False)
    _print_stats("dinámico (15 features)", df["riesgo_reprobacion"].values, path_dinamico)
    return path_dinamico

def _print_stats(nombre, riesgo, path):
    r = np.array(riesgo)
    print(f"\nDataset {nombre}:")
    print(f"  Estudiantes   : {len(r)}")
    print(f"  Con riesgo    : {r.sum()} ({r.mean()*100:.1f}%)")
    print(f"  Sin riesgo    : {(1-r).sum()} ({(1-r.mean())*100:.1f}%)")
    print(f"  Guardado en   : {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=["original", "dinamico", "ambos"],
                        default="ambos", help="Qué dataset generar")
    args = parser.parse_args()

    if args.version in ("original", "ambos"):
        generar_original()
    if args.version in ("dinamico", "ambos"):
        generar_dinamico()