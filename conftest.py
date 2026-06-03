"""conftest.py — Configuración global de pytest."""
import sys
from pathlib import Path

# Asegura que src/ sea importable desde cualquier directorio de tests
sys.path.insert(0, str(Path(__file__).resolve().parent))
