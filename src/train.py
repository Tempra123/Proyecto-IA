"""
src/train.py  —  Entrenamiento v2  (features dinámicas)
--------------------------------------------------------
Entrena el modelo XGBoost sobre las 6 features normalizadas.
Es compatible tanto con el dataset original como con el dinámico.

Uso:
    python -m scripts.generar_datos --version dinamico
    python -m scripts.entrenar --dataset dinamico
"""

import joblib
import numpy  as np
import pandas as pd
from pathlib import Path

from xgboost                 import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.impute          import SimpleImputer
from sklearn.pipeline import Pipeline as ImbPipeline

from src.config import (
    MODEL_PATH, MODEL_INFO_PATH, MODELS_DIR,
    FEATURES, TARGET,
    RANDOM_STATE, TEST_SIZE, CV_FOLDS, DECISION_UMBRAL,
    DATA_DIR,
)


def _cargar_datos(dataset: str = "dinamico"):
    """Carga el dataset correcto según el tipo de entrenamiento."""
    if dataset == "dinamico":
        path = DATA_DIR / "datos_dinamicos.csv"
        if not path.exists():
            raise FileNotFoundError(
                "No existe datos_dinamicos.csv. "
                "Ejecuta: python -m scripts.generar_datos --version dinamico"
            )
        df      = pd.read_csv(path)
        X       = df[FEATURES].copy()
        y       = df[TARGET].copy()
    else:
        # Dataset original (9 variables del CSV)
        from src.preprocess import load_data
        from src.config     import CSV_FEATURES
        X, y = load_data()
    return X, y


def _construir_pipeline(scale_pos_weight: float) -> ImbPipeline:
    """Pipeline: imputación → escala → SMOTE → XGBoost."""
    xgb = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return ImbPipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
    ("clf",     xgb),

    ])
    xgb = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return ImbPipeline([
        ("pre",   preprocesador),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf",   xgb),
    ])


def entrenar(verbose: bool = True, dataset: str = "dinamico") -> dict:
    """
    Entrena el modelo y guarda pkl.

    dataset : "dinamico" (6 features normalizadas) | "original" (9 variables CSV)
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Datos
    X, y = _cargar_datos(dataset)
    if verbose:
        print(f"Dataset '{dataset}': {len(X)} estudiantes | "
              f"Con riesgo: {y.sum()} ({y.mean()*100:.1f}%)")
        if dataset == "dinamico":
            print(f"Features: {list(X.columns)}")

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # 3. Pipeline + grid
    n_neg, n_pos = (y_train == 0).sum(), (y_train == 1).sum()
    spw      = round(n_neg / n_pos, 2)
    pipeline = _construir_pipeline(spw)

    param_grid = {
        "clf__n_estimators":  [100, 200, 300],
        "clf__max_depth":     [3, 5, 7],
        "clf__learning_rate": [0.05, 0.1, 0.2],
        "clf__subsample":     [0.8, 1.0],
    }

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(
        pipeline, param_grid,
        scoring="recall",   # prioridad: no perder estudiantes en riesgo
        cv=cv, n_jobs=-1, verbose=0,
    )

    if verbose:
        print("Entrenando XGBoost con GridSearchCV… (puede tardar ~1-2 min)")
    gs.fit(X_train, y_train)

    if verbose:
        print(f"Mejores params: {gs.best_params_}")
        print(f"Recall CV:      {gs.best_score_:.4f}")

    # 4. Evaluar
    best = gs.best_estimator_
    y_proba = best.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= DECISION_UMBRAL).astype(int)

    metricas = {
        "accuracy":  round(accuracy_score(y_test,  y_pred),  4),
        "precision": round(precision_score(y_test, y_pred),  4),
        "recall":    round(recall_score(y_test,    y_pred),  4),
        "f1":        round(f1_score(y_test,        y_pred),  4),
        "auc_roc":   round(roc_auc_score(y_test,   y_proba), 4),
    }

    if verbose:
        print("\n--- Métricas en conjunto de prueba ---")
        for k, v in metricas.items():
            print(f"  {k:<12}: {v:.4f}")

    # 5. Guardar
    joblib.dump(best, MODEL_PATH)
    joblib.dump({
        "algoritmo":    "XGBoost",
        "metricas":     metricas,
        "best_params":  gs.best_params_,
        "umbral":       DECISION_UMBRAL,
        "features":     list(X.columns),
        "dataset":      dataset,
    }, MODEL_INFO_PATH)

    if verbose:
        print(f"\nModelo guardado en: {MODEL_PATH}")

    return metricas


if __name__ == "__main__":
    entrenar(verbose=True, dataset="dinamico")
