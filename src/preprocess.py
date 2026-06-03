"""
src/preprocess.py
-----------------
Pipeline de preprocesamiento: imputacion, escalado y codificacion.
Se importa desde train.py y predict.py.
"""

import pandas as pd
from sklearn.pipeline       import Pipeline
from sklearn.preprocessing  import StandardScaler, OneHotEncoder
from sklearn.impute          import SimpleImputer
from sklearn.compose         import ColumnTransformer

from src.config import (
    NUMERIC_COLS, BINARY_COLS, CATEG_COLS,
    FEATURES, TARGET, DATA_PATH
)


def build_preprocessor() -> ColumnTransformer:
    """
    Devuelve un ColumnTransformer listo para usarse dentro de un Pipeline.

    - Numericas  : imputacion por mediana + estandarizacion
    - Binarias   : imputacion por moda (sin escalar)
    - Categoricas: imputacion por moda + One-Hot Encoding
    """
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    binary_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])

    categ_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe",     OneHotEncoder(drop="first", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_COLS),
        ("bin", binary_pipe,  BINARY_COLS),
        ("cat", categ_pipe,   CATEG_COLS),
    ])


def load_data(path=None):
    """
    Carga el CSV y separa features / target.
    Retorna (X: DataFrame, y: Series).
    """
    csv_path = path or DATA_PATH
    df = pd.read_csv(csv_path)

    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Columnas faltantes en el CSV: {missing}")

    X = df[FEATURES].copy()
    y = df[TARGET].copy()
    return X, y
