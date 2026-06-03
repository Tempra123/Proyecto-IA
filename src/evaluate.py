"""
src/evaluate.py
---------------
Funciones para calcular metricas y generar graficas del modelo.
Usado desde app.py (pestana Metricas del Modelo).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score,
)

from src.config import FEATURE_LABELS

# Paleta de colores calida (consistente con la UI)
COLOR_PRIMARY   = "#C0392B"   # rojo terracota
COLOR_SECONDARY = "#E67E22"   # naranja ambar
COLOR_OK        = "#27AE60"   # verde
COLOR_BG        = "#FDF6EC"   # crema


def calcular_metricas(y_true, y_pred, y_proba) -> dict:
    return {
        "Exactitud (Accuracy)":  round(accuracy_score(y_true, y_pred),  4),
        "Precision":             round(precision_score(y_true, y_pred), 4),
        "Recall (Sensibilidad)": round(recall_score(y_true, y_pred),    4),
        "F1-Score":              round(f1_score(y_true, y_pred),        4),
        "AUC-ROC":               round(roc_auc_score(y_true, y_proba),  4),
    }


def plot_confusion_matrix(y_true, y_pred):
    cm  = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4), facecolor=COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    sns.heatmap(
        cm, annot=True, fmt="d",
        cmap=sns.light_palette(COLOR_PRIMARY, as_cmap=True),
        xticklabels=["Sin Riesgo", "Con Riesgo"],
        yticklabels=["Sin Riesgo", "Con Riesgo"],
        ax=ax, linewidths=0.5
    )
    ax.set_xlabel("Prediccion", fontsize=11)
    ax.set_ylabel("Real",       fontsize=11)
    ax.set_title("Matriz de Confusion", fontsize=13, fontweight="bold", color="#3D2B1F")
    plt.tight_layout()
    return fig


def plot_roc_curve(y_true, y_proba):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc     = auc(fpr, tpr)
    fig, ax     = plt.subplots(figsize=(5, 4), facecolor=COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    ax.plot(fpr, tpr, color=COLOR_PRIMARY, lw=2.5,
            label=f"AUC = {roc_auc:.3f}")
    ax.fill_between(fpr, tpr, alpha=0.15, color=COLOR_PRIMARY)
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Tasa de Falsos Positivos", fontsize=11)
    ax.set_ylabel("Tasa de Verdaderos Positivos", fontsize=11)
    ax.set_title("Curva ROC", fontsize=13, fontweight="bold", color="#3D2B1F")
    ax.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    return fig


def plot_feature_importance(modelo, top_n: int = 9):
    """
    Grafica la importancia de variables del XGBoost.
    Funciona con el pipeline ImbPipeline (accede a named_steps['clf']).
    """
    try:
        clf = modelo.named_steps["clf"]
        importances = clf.feature_importances_

        # Reconstruir nombres de columnas despues del ColumnTransformer
        pre = modelo.named_steps["pre"]
        feature_names = list(
            pre.get_feature_names_out()
        )
        # Acortar prefijos (num__, bin__, cat__)
        clean_names = []
        for fn in feature_names:
            for orig, label in FEATURE_LABELS.items():
                if orig in fn:
                    clean_names.append(label)
                    break
            else:
                clean_names.append(fn.split("__")[-1])

        # Top N features
        idx = np.argsort(importances)[-top_n:]
        fig, ax = plt.subplots(figsize=(7, 4), facecolor=COLOR_BG)
        ax.set_facecolor(COLOR_BG)
        bars = ax.barh(
            [clean_names[i] for i in idx],
            importances[idx],
            color=[COLOR_PRIMARY if importances[i] == importances[idx[-1]]
                   else COLOR_SECONDARY for i in idx]
        )
        ax.set_xlabel("Importancia (ganancia)", fontsize=11)
        ax.set_title("Variables mas influyentes (XGBoost)",
                     fontsize=13, fontweight="bold", color="#3D2B1F")
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"[evaluate] No se pudo graficar importancia: {e}")
        return None


def plot_distribucion_riesgo(df_resultado):
    """
    Grafica la distribucion de probabilidades de riesgo en un lote.
    """
    fig, ax = plt.subplots(figsize=(6, 3), facecolor=COLOR_BG)
    ax.set_facecolor(COLOR_BG)

    con_riesgo    = df_resultado[df_resultado["prediccion"] == 1]["probabilidad_riesgo"]
    sin_riesgo    = df_resultado[df_resultado["prediccion"] == 0]["probabilidad_riesgo"]

    ax.hist(sin_riesgo, bins=20, color=COLOR_OK,      alpha=0.7, label="Sin riesgo")
    ax.hist(con_riesgo, bins=20, color=COLOR_PRIMARY,  alpha=0.7, label="Con riesgo")
    ax.set_xlabel("Probabilidad de riesgo", fontsize=11)
    ax.set_ylabel("Cantidad de estudiantes", fontsize=11)
    ax.set_title("Distribucion de probabilidades", fontsize=12, fontweight="bold", color="#3D2B1F")
    ax.legend()
    plt.tight_layout()
    return fig
