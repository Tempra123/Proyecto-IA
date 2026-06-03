# Sistema de Deteccion de Riesgo Academico

Predice si un estudiante tiene riesgo de reprobar usando **XGBoost** con
optimizacion de hiperparametros via **GridSearchCV** y balanceo de clases con **SMOTE**.

---

## Estructura del proyecto

```
riesgo_v2/
├── app.py                    # Interfaz Streamlit (punto de entrada)
├── requirements.txt
├── data/
│   └── datos_estudiantes.csv  # Generado por scripts/generar_datos.py
├── models/
│   ├── modelo_xgb.pkl         # Modelo entrenado
│   └── info_modelo.pkl        # Metricas y metadatos
├── src/
│   ├── config.py              # Configuracion centralizada (rutas, features, etc.)
│   ├── preprocess.py          # Pipeline de preprocesamiento
│   ├── train.py               # Logica de entrenamiento
│   ├── predict.py             # Prediccion individual y por lote
│   └── evaluate.py            # Metricas y graficas
└── scripts/
    ├── generar_datos.py       # Genera dataset sintetico (una sola vez)
    └── entrenar.py            # Punto de entrada para entrenar
```

---

## Instalacion

```bash
pip install -r requirements.txt
```

## Uso

### 1. Generar datos (una sola vez)
```bash
python -m scripts.generar_datos
```

### 2. Entrenar el modelo
```bash
python -m scripts.entrenar
```

### 3. Lanzar la aplicacion
```bash
streamlit run app.py
```

---

## Por que XGBoost

XGBoost (Gradient Boosting) es el algoritmo mas adecuado para este problema porque:

- **Aprende patrones combinados**: no suma promedios, descubre interacciones entre
  variables (ej: baja asistencia + sin tutoria = riesgo multiplicado).
- **Gradient Boosting secuencial**: cada arbol corrige los errores del anterior,
  logrando alta precision con menos datos.
- **Manejo nativo de desbalance**: `scale_pos_weight` penaliza mas los errores
  en la clase minoritaria (estudiantes en riesgo).
- **Regularizacion integrada (L1/L2)**: evita sobreajuste sin configuracion adicional.
- **Interpretabilidad**: produce importancias de variables claras.

---

## Variables de entrada

| Variable                  | Descripcion                         |
|---------------------------|-------------------------------------|
| nota_parcial_1            | Calificacion primer parcial (0-100) |
| nota_parcial_2            | Calificacion segundo parcial (0-100)|
| porcentaje_asistencia     | Asistencia a clases (%)             |
| tareas_entregadas         | Tareas entregadas a tiempo (0-10)   |
| horas_estudio_semanal     | Horas de estudio por semana         |
| participa_tutoria         | Participa en tutorias (0/1)         |
| promedio_ciclo_anterior   | Promedio del ciclo anterior (0-100) |
| nivel_socioeconomico      | Nivel socioeconomico (0=Bajo, 1=Medio, 2=Alto) |
| acceso_internet           | Acceso a internet en casa (0/1)     |

**Variable objetivo:** `riesgo_reprobacion` (1 = con riesgo, 0 = sin riesgo)

---

*Proyecto academico — Materia: Inteligencia Artificial*
