"""
scripts/entrenar.py
-------------------
Punto de entrada para entrenar el modelo.

Uso:
    python -m scripts.entrenar
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.train import entrenar

if __name__ == "__main__":
    print("=" * 55)
    print("  SISTEMA DE RIESGO ACADEMICO — Entrenamiento")
    print("=" * 55)
    metricas = entrenar(verbose=True)
    print("\nEntrenamiento completado. Ejecuta:")
    print("  streamlit run app.py")
