"""
app.py — Sistema de Riesgo Académico v6
Evaluaciones dinámicas + comportamiento/contexto del estudiante (15 features).
"""

import streamlit as st
import pandas  as pd
import numpy   as np
import joblib

st.set_page_config(page_title="Sistema de Riesgo Académico", page_icon=None, layout="wide")

# CSS TRON LEGACY — FULL REDESIGN
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700;900&family=Share+Tech+Mono&display=swap');

  /* ── FONDO TRON CON GRID PERSPECTIVA ─────────────────────────────── */
  .stApp {
    background-color: #000509;
    background-image:
      linear-gradient(rgba(0,200,255,0.07) 1px, transparent 1px),
      linear-gradient(90deg, rgba(0,200,255,0.07) 1px, transparent 1px),
      radial-gradient(ellipse 80% 50% at 50% 0%, rgba(0,150,255,0.12) 0%, transparent 70%);
    background-size: 60px 60px, 60px 60px, 100% 100%;
    background-attachment: fixed;
  }

  /* ── TIPOGRAFÍA GLOBAL ───────────────────────────────────────────── */
  html, body, [class*="css"] {
    color: #9fd8f0;
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.1rem;
  }

  /* ── SIDEBAR ─────────────────────────────────────────────────────── */
  [data-testid="stSidebar"] {
    background-color: #000c1a;
    border-right: 1px solid #00c8ff22;
  }
  [data-testid="stSidebar"] * { color: #4dbfe0 !important; }

  /* ── HEADER PRINCIPAL ────────────────────────────────────────────── */
  .tron-header-wrap {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem 1rem;
    position: relative;
  }
  .tron-logo-line {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0;
  }
  .tron-slash {
    font-family: 'Orbitron', monospace;
    font-size: 3.6rem;
    font-weight: 900;
    color: #e8640a;
    text-shadow:
      0 0 10px #ff6600cc,
      0 0 30px #ff660066,
      0 0 60px #ff660033;
    letter-spacing: -2px;
    margin-right: 18px;
    line-height: 1;
  }
  .tron-title {
    font-family: 'Orbitron', monospace;
    font-size: 3.6rem;
    font-weight: 900;
    color: #00d4ff;
    text-shadow:
      0 0 8px #00d4ffcc,
      0 0 25px #00d4ff88,
      0 0 60px #00d4ff44,
      0 0 120px #00d4ff22;
    letter-spacing: 8px;
    line-height: 1;
    text-transform: uppercase;
  }
  .tron-underline {
    height: 2px;
    background: linear-gradient(90deg, transparent 0%, #e8640a 20%, #00d4ff 50%, #e8640a 80%, transparent 100%);
    margin: 1rem auto 0 auto;
    max-width: 700px;
    box-shadow: 0 0 12px #00d4ff88, 0 0 6px #e8640a88;
  }
  .tron-subtitulo {
    color: #2e8da8;
    font-size: 0.95rem;
    margin-top: 0.8rem;
    letter-spacing: 5px;
    font-family: 'Share Tech Mono', monospace;
    text-transform: uppercase;
    text-align: center;
  }
  .tron-subtitulo span {
    color: #e8640a;
    font-size: 0.85rem;
  }

  /* ── SECCIÓN TÍTULOS ─────────────────────────────────────────────── */
  .seccion-titulo {
    font-family: 'Orbitron', monospace;
    color: #00d4ff;
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 3px;
    border-bottom: 1px solid #00d4ff33;
    padding-bottom: 6px;
    margin: 1.4rem 0 0.8rem 0;
    text-shadow: 0 0 10px #00d4ff99;
    text-transform: uppercase;
  }

  /* ── CARDS DE RESULTADO ──────────────────────────────────────────── */
  .card-riesgo {
    background: linear-gradient(135deg, #1a0005 0%, #2a0010 100%);
    border: 1px solid #ff2255;
    border-left: 5px solid #ff2255;
    border-radius: 2px;
    padding: 1.6rem 2rem;
    margin-bottom: 1rem;
    box-shadow: 0 0 30px #ff225544, 0 0 60px #ff225511, inset 0 0 40px #ff22550a;
  }
  .card-ok {
    background: linear-gradient(135deg, #00040f 0%, #000a1a 100%);
    border: 1px solid #00d4ff;
    border-left: 5px solid #00d4ff;
    border-radius: 2px;
    padding: 1.6rem 2rem;
    margin-bottom: 1rem;
    box-shadow: 0 0 30px #00d4ff44, 0 0 60px #00d4ff11, inset 0 0 40px #00d4ff0a;
  }

  /* ── VARS DE EVALUACIÓN ──────────────────────────────────────────── */
  .var-alerta {
    background: #130006;
    border-left: 3px solid #ff3355;
    border-radius: 0;
    padding: 0.7rem 1.2rem;
    margin: 5px 0;
    font-size: 1rem;
    color: #ff99bb;
    font-family: 'Share Tech Mono', monospace;
  }
  .var-ok {
    background: #00080f;
    border-left: 3px solid #00d4ff;
    border-radius: 0;
    padding: 0.7rem 1.2rem;
    margin: 5px 0;
    font-size: 1rem;
    color: #55dcff;
    font-family: 'Share Tech Mono', monospace;
  }
  .var-pendiente {
    background: #00091a;
    border-left: 3px solid #2266bb;
    border-radius: 0;
    padding: 0.7rem 1.2rem;
    margin: 5px 0;
    font-size: 1rem;
    color: #6699cc;
    font-family: 'Share Tech Mono', monospace;
  }
  .card-info {
    background: #000d1a;
    border: 1px solid #00d4ff1a;
    border-radius: 2px;
    padding: 1rem 1.4rem;
    margin-bottom: 0.8rem;
  }
  .card-comportamiento {
    background: #000609;
    border: 1px solid #00d4ff1a;
    border-radius: 2px;
    padding: 1rem 1.4rem;
    margin: 0.5rem 0;
  }

  /* ── BOTONES ─────────────────────────────────────────────────────── */
  .stButton > button {
    background: transparent !important;
    color: #00d4ff !important;
    border: 1px solid #00d4ff !important;
    border-radius: 0 !important;
    font-weight: 700 !important;
    font-family: 'Orbitron', monospace !important;
    letter-spacing: 2px !important;
    font-size: 0.9rem !important;
    padding: 0.6rem 1.4rem !important;
    transition: all 0.25s !important;
    text-transform: uppercase !important;
    clip-path: polygon(8px 0%, 100% 0%, calc(100% - 8px) 100%, 0% 100%);
  }
  .stButton > button:hover {
    background: #00d4ff15 !important;
    box-shadow: 0 0 25px #00d4ff66, inset 0 0 20px #00d4ff0a !important;
    color: #fff !important;
    border-color: #00d4ff !important;
  }

  /* ── TABS ────────────────────────────────────────────────────────── */
  .stTabs [data-baseweb="tab"] {
    color: #2e7a96 !important;
    font-weight: 700;
    font-family: 'Orbitron', monospace !important;
    letter-spacing: 2px;
    font-size: 0.82rem !important;
    text-transform: uppercase !important;
    padding: 0.5rem 1.2rem !important;
  }
  .stTabs [aria-selected="true"] {
    color: #00d4ff !important;
    border-bottom: 3px solid #00d4ff !important;
    text-shadow: 0 0 12px #00d4ffaa;
    background: #00d4ff08 !important;
  }
  .stTabs [data-baseweb="tab-list"] {
    border-bottom: 1px solid #00d4ff22 !important;
    gap: 4px !important;
  }

  /* ── MÉTRICAS ────────────────────────────────────────────────────── */
  [data-testid="stMetricValue"] {
    color: #00d4ff !important;
    font-size: 2rem !important;
    font-family: 'Orbitron', monospace !important;
    text-shadow: 0 0 10px #00d4ff88;
  }
  [data-testid="stMetricLabel"] {
    color: #2e8da8 !important;
    font-size: 0.9rem !important;
    letter-spacing: 2px;
    text-transform: uppercase;
    font-family: 'Share Tech Mono', monospace !important;
  }
  [data-testid="metric-container"] {
    background: #000c18;
    border: 1px solid #00d4ff22;
    border-top: 2px solid #00d4ff44;
    border-radius: 0;
    padding: 1rem 1.2rem;
  }

  /* ── INPUTS ──────────────────────────────────────────────────────── */
  .stNumberInput input, .stTextInput input {
    background: #000c18 !important;
    color: #9fd8f0 !important;
    border: 1px solid #00d4ff33 !important;
    border-radius: 0 !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 1.05rem !important;
  }
  .stNumberInput input:focus, .stTextInput input:focus {
    border-color: #00d4ff88 !important;
    box-shadow: 0 0 15px #00d4ff22 !important;
  }
  [data-baseweb="select"] { background: #000c18 !important; border-radius: 0 !important; }
  [data-baseweb="select"] * {
    color: #9fd8f0 !important;
    font-family: 'Share Tech Mono', monospace !important;
  }
  [data-baseweb="select"] [data-baseweb="select-control"] {
    border: 1px solid #00d4ff33 !important;
    border-radius: 0 !important;
  }

  /* ── FILE UPLOADER / MISC ────────────────────────────────────────── */
  [data-testid="stFileUploader"] {
    border: 1px dashed #00d4ff44 !important;
    border-radius: 2px !important;
    background: #000c18 !important;
  }
  [data-testid="stDataFrame"] {
    border: 1px solid #00d4ff1a;
    border-radius: 2px;
  }
  hr { border-color: #00d4ff1a; margin: 1.5rem 0; }

  /* ── BARRA DE PROBABILIDAD ───────────────────────────────────────── */
  .prob-bar-bg {
    background: #000c18;
    border: 1px solid #00d4ff22;
    border-radius: 0;
    height: 42px;
    overflow: hidden;
    margin: 10px 0;
  }

  /* ── LABELS ──────────────────────────────────────────────────────── */
  label, .stSelectbox label, .stNumberInput label, .stTextInput label,
  .stSlider label, .stFileUploader label {
    color: #4dbfe0 !important;
    font-size: 0.95rem !important;
    letter-spacing: 1.5px !important;
    font-family: 'Share Tech Mono', monospace !important;
    text-transform: uppercase !important;
  }
  p, span { font-size: 1.05rem; }
  .stCaption, small { font-size: 0.92rem !important; color: #2e7a96 !important; }

  /* ── EXPANDER ────────────────────────────────────────────────────── */
  .stExpander {
    border: 1px solid #00d4ff1a !important;
    border-left: 3px solid #00d4ff33 !important;
    border-radius: 0 !important;
    background: #000509 !important;
  }
  .stExpander summary {
    color: #4dbfe0 !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 1rem !important;
    letter-spacing: 1px;
  }

  /* ── CODE BLOCKS ─────────────────────────────────────────────────── */
  code, pre {
    font-family: 'Share Tech Mono', monospace !important;
    color: #55dcff !important;
    background: #000c18 !important;
    border: 1px solid #00d4ff22 !important;
    border-radius: 0 !important;
  }

  /* ── DIVIDER ─────────────────────────────────────────────────────── */
  [data-testid="stDivider"] hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent, #00d4ff33, transparent) !important;
  }

  /* ── SLIDER ──────────────────────────────────────────────────────── */
  .stSlider [data-baseweb="slider"] div[role="slider"] {
    background: #00d4ff !important;
    box-shadow: 0 0 10px #00d4ff88 !important;
  }

  /* ── TOGGLE ──────────────────────────────────────────────────────── */
  .stToggle span[data-baseweb="toggle"] {
    background: #001a2a !important;
  }
</style>
""", unsafe_allow_html=True)

# ── HEADER TRON LEGACY ─────────────────────────────────────────────────────────
st.markdown("""
<div class="tron-header-wrap">
  <div class="tron-logo-line">
    <span class="tron-slash">// </span>
    <span class="tron-title">SISTEMA DE RIESGO ACADEMICO</span>
  </div>
  <div class="tron-underline"></div>
  <p class="tron-subtitulo">
    EVALUACIONES DINAMICAS &nbsp;<span>·</span>&nbsp;
    15 FEATURES &nbsp;<span>·</span>&nbsp;
    IA FLEXIBLE
  </p>
</div>
""", unsafe_allow_html=True)
st.divider()


# 
# CONSTANTES
# 

TIPOS_SLUGS = {
  "Parcial":     "parcial",
  "Proyecto":     "proyecto",
  "Tarea / Trabajo": "tarea",
  "Laboratorio":   "laboratorio",
  "Examen Final":   "examen_final",
  "Puntos Extra":   "puntos_extra",
}

OPCIONES_ESCALA = {
  "Sobre 100":     "100",
  "Sobre 70":      "70",
  "Sobre 20":      "20",
  "Sobre 10":      "10",
  "Porcentaje (0-100)": "pct",
  "Letras (A, B+, F…)": "letra",
}

LETRAS_VALIDAS  = ["A+","A","A-","B+","B","B-","C+","C","C-","D+","D","D-","F"]
LETRAS_PENDIENTE = ["(pendiente)"] + LETRAS_VALIDAS
MAXIMOS_ESCALA  = {"100": 100, "70": 70, "20": 20, "10": 10, "pct": 100}


# 
# PANEL 1 — CONFIGURADOR DE EVALUACIONES
# 

def panel_evaluaciones_dinamicas(key_prefix: str = "ev") -> dict:
  """Configura la estructura de evaluaciones de la materia."""
  st.markdown('<p class="seccion-titulo">CONFIGURAR MATERIA</p>', unsafe_allow_html=True)

  col_mat, col_min = st.columns(2)
  with col_mat:
    nombre_materia = st.text_input(
      "NOMBRE DE LA MATERIA", placeholder="Ej: Cálculo I",
      key=f"{key_prefix}_nombre"
    )
  with col_min:
    nota_min = st.number_input(
      "NOTA MINIMA PARA APROBAR (%)", 1, 100, 51,
      key=f"{key_prefix}_nota_min",
      help="Porcentaje mínimo de aprobación. Se aplica independiente de la escala de cada evaluación."
    )

  st.markdown('<p class="seccion-titulo">EVALUACIONES DE LA MATERIA</p>', unsafe_allow_html=True)
  st.caption("Añade cada tipo de evaluación con su nombre, tipo, escala y peso. Los pesos regulares deben sumar 100%.")

  key_ev = f"{key_prefix}_evaluaciones"
  if key_ev not in st.session_state:
    st.session_state[key_ev] = [
      {"nombre": "Parcial 1", "slug": "parcial", "peso": 30, "escala_slug": "100"},
      {"nombre": "Parcial 2", "slug": "parcial", "peso": 30, "escala_slug": "100"},
      {"nombre": "Proyecto", "slug": "proyecto", "peso": 30, "escala_slug": "100"},
      {"nombre": "Tarea 1",  "slug": "tarea",  "peso": 10, "escala_slug": "100"},
    ]

  evaluaciones_cfg = []
  pesos_regulares = 0

  # Cabecera de columnas
  h1, h2, h3, h4, h5 = st.columns([3, 2, 2, 1, 0.5])
  h1.markdown('<span style="color:#5bc8e8;font-size:0.85rem;letter-spacing:1px;">NOMBRE</span>', unsafe_allow_html=True)
  h2.markdown('<span style="color:#5bc8e8;font-size:0.85rem;letter-spacing:1px;">TIPO</span>', unsafe_allow_html=True)
  h3.markdown('<span style="color:#5bc8e8;font-size:0.85rem;letter-spacing:1px;">ESCALA</span>', unsafe_allow_html=True)
  h4.markdown('<span style="color:#5bc8e8;font-size:0.85rem;letter-spacing:1px;">PESO %</span>', unsafe_allow_html=True)

  for i, ev in enumerate(st.session_state[key_ev]):
    with st.container():
      c1, c2, c3, c4, c5 = st.columns([3, 2, 2, 1, 0.5])
      with c1:
        nombre_ev = st.text_input(
          f"Nombre #{i+1}", value=ev["nombre"],
          key=f"{key_prefix}_nom_{i}", label_visibility="collapsed"
        )
      with c2:
        tipo_label = st.selectbox(
          "Tipo", list(TIPOS_SLUGS.keys()),
          index=list(TIPOS_SLUGS.values()).index(ev["slug"]) if ev["slug"] in TIPOS_SLUGS.values() else 0,
          key=f"{key_prefix}_tipo_{i}", label_visibility="collapsed"
        )
        slug = TIPOS_SLUGS[tipo_label]
      with c3:
        escala_actual = ev.get("escala_slug", "100")
        idx_escala  = list(OPCIONES_ESCALA.values()).index(escala_actual) if escala_actual in OPCIONES_ESCALA.values() else 0
        escala_label = st.selectbox(
          "Escala", list(OPCIONES_ESCALA.keys()),
          index=idx_escala,
          key=f"{key_prefix}_escala_{i}", label_visibility="collapsed",
          help="Escala de calificación de esta evaluación"
        )
        escala_slug = OPCIONES_ESCALA[escala_label]
      with c4:
        es_extra = (slug == "puntos_extra")
        peso_ev = st.number_input(
          "Peso %", 0, 100,
          value=0 if es_extra else ev["peso"],
          key=f"{key_prefix}_peso_{i}",
          label_visibility="collapsed",
          disabled=es_extra,
          help="0% para puntos extra"
        )
      with c5:
        if st.button("X", key=f"{key_prefix}_del_{i}", help="Eliminar"):
          st.session_state[key_ev].pop(i)
          st.rerun()

      st.session_state[key_ev][i] = {
        "nombre": nombre_ev, "slug": slug,
        "peso": peso_ev, "escala_slug": escala_slug,
      }
      if not es_extra:
        pesos_regulares += peso_ev
      evaluaciones_cfg.append({
        "nombre": nombre_ev, "slug": slug,
        "peso": peso_ev, "escala_slug": escala_slug,
      })

  col_btn, col_msg = st.columns([2, 3])
  with col_btn:
    if st.button("+ ANADIR EVALUACION", key=f"{key_prefix}_add"):
      st.session_state[key_ev].append({
        "nombre": f"Evaluación {len(st.session_state[key_ev])+1}",
        "slug": "tarea", "peso": 0, "escala_slug": "100"
      })
  with col_msg:
    if pesos_regulares == 100:
      st.markdown('<span style="color:#00c3ff;font-size:0.9rem;">Los pesos suman 100%</span>', unsafe_allow_html=True)
    else:
      st.markdown(f'<span style="color:#ff4466;font-size:0.9rem;">Los pesos suman {pesos_regulares}% — deben sumar 100%</span>', unsafe_allow_html=True)

  return {
    "nombre_materia":     nombre_materia,
    "nota_minima_aprobacion": nota_min,
    "escala_maxima":     100,
    "evaluaciones_plantilla": evaluaciones_cfg,
    "pesos_validos":     pesos_regulares == 100,
  }


# 
# PANEL 2 — COMPORTAMIENTO Y CONTEXTO DEL ESTUDIANTE ← NUEVO (Paso F)
# 

def panel_comportamiento(key_prefix: str = "comp") -> dict:
  """
  Recoge los datos de comportamiento y contexto del estudiante.
  Todos los campos son opcionales — si no se ingresan, features.py
  los imputa con mediana y el sistema sigue funcionando.
  Retorna un dict listo para pasar a calcular_features(..., comportamiento=...).
  """
  with st.expander("Comportamiento y contexto del estudiante (opcional)", expanded=False):
    st.caption(
      "Estos datos mejoran la predicción. Si no los tienes disponibles, "
      "déjalos en sus valores por defecto — el modelo los estimará."
    )

    # Fila 1: Asistencia y tareas 
    st.markdown('<p class="seccion-titulo" style="font-size:0.95rem;">Asistencia y entregas</p>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
      asistencia_pct = st.slider(
        "Asistencia a clases (%)", 0, 100, 85,
        key=f"{key_prefix}_asistencia",
        help="Porcentaje de clases a las que asistió el estudiante"
      )

    with col2:
      tareas_total = st.number_input(
        "Total de tareas asignadas", 0, 30, 5,
        key=f"{key_prefix}_tareas_total",
        help="Cuántas tareas se asignaron en total durante el período"
      )

    with col3:
      tareas_a_tiempo = st.number_input(
        "Tareas entregadas a tiempo", 0, int(tareas_total), int(tareas_total),
        key=f"{key_prefix}_tareas_tiempo",
        help="De las tareas asignadas, cuántas entregó antes del plazo"
      )

    # Validación silenciosa
    if tareas_a_tiempo > tareas_total:
      st.warning("Las tareas a tiempo no pueden superar el total.")
      tareas_a_tiempo = tareas_total

    # Fila 2: Tutoría y contexto 
    st.markdown('<p class="seccion-titulo" style="font-size:0.95rem;">Tutoría y carga académica</p>', unsafe_allow_html=True)
    col4, col5, col6 = st.columns(3)

    with col4:
      visitas_tutoria = st.number_input(
        "Visitas a tutoría / consulta", 0, 10, 0,
        key=f"{key_prefix}_tutoria",
        help="Número de veces que el estudiante consultó al docente o fue a tutoría"
      )

    with col5:
      materias_paralelo = st.number_input(
        "Materias que cursa en paralelo", 1, 10, 4,
        key=f"{key_prefix}_materias",
        help="Total de materias que el estudiante está cursando este período"
      )

    with col6:
      es_repitente = st.toggle(
        "¿Es repitente en esta materia?",
        value=False,
        key=f"{key_prefix}_repitente",
        help="Activa si el estudiante ya cursó esta materia anteriormente"
      )

    # Resumen visual 
    asistencia_norm = asistencia_pct / 100
    alerta_asis   = asistencia_norm < 0.70
    ratio_tareas  = tareas_a_tiempo / tareas_total if tareas_total > 0 else 1.0

    col_r1, col_r2, col_r3 = st.columns(3)
    col_r1.metric(
      "Asistencia",
      f"{asistencia_pct}%",
      delta="Baja" if alerta_asis else " Ok",
      delta_color="inverse" if alerta_asis else "normal"
    )
    col_r2.metric(
      "Tareas a tiempo",
      f"{tareas_a_tiempo}/{tareas_total}",
      delta=f"{ratio_tareas*100:.0f}%"
    )
    col_r3.metric(
      "Visitas tutoría",
      visitas_tutoria,
      delta="Repitente" if es_repitente else "Primera vez"
    )

  return {
    "asistencia":    asistencia_pct / 100,
    "tareas_a_tiempo":  int(tareas_a_tiempo),
    "tareas_total":   int(tareas_total),
    "visitas_tutoria":  int(visitas_tutoria),
    "materias_paralelo": int(materias_paralelo),
    "es_repitente":   bool(es_repitente),
  }


# 
# PREDICCIÓN
# 

@st.cache_resource
def _cargar_modelo():
  from src.predict import cargar_modelo, modelo_disponible
  if not modelo_disponible():
    return None
  return cargar_modelo()


def predecir_y_mostrar(
  evaluaciones_con_notas: list[dict],
  cfg: dict,
  comportamiento: dict | None = None,
  nombre: str = "",
):
  """Llama al motor de predicción y renderiza el resultado."""
  from src.predict import predecir_estudiante_dinamico, modelo_disponible

  comp  = comportamiento or {}
  modelo = _cargar_modelo()

  if modelo is None:
    resultado = _reglas_sin_modelo(evaluaciones_con_notas, cfg, comp)
  else:
    resultado = predecir_estudiante_dinamico(modelo, evaluaciones_con_notas, cfg, comp)

  _mostrar_resultado(resultado, nombre)


def _reglas_sin_modelo(evaluaciones: list[dict], cfg: dict, comportamiento: dict) -> dict:
  """
  Predicción heurística cuando el modelo pkl no está disponible.
  Ahora usa las 15 features incluyendo comportamiento.
  """
  from src.features import calcular_features, evaluar_detalle_variables

  feats = calcular_features(evaluaciones, cfg, comportamiento)

  # Heurística extendida: ahora incluye asistencia y tareas
  score = (
    (1 - feats["nota_ponderada"])  * 0.40
    + (1 - feats["ratio_completadas"]) * 0.15
    + feats["ratio_reprobados"]    * 0.10
    + (1 - feats["asistencia"])    * 0.20
    + (1 - feats["ratio_tareas_tiempo"]) * 0.10
    + feats["es_repitente"]      * 0.05
  )
  score = float(np.clip(score, 0, 1))
  clase = int(score >= 0.45)

  if score >= 0.70:  nivel = "Alto"
  elif score >= 0.45: nivel = "Moderado"
  else:        nivel = "Bajo"

  return {
    "clase":    clase,
    "probabilidad": round(score, 4),
    "etiqueta":   "CON RIESGO" if clase else "SIN RIESGO",
    "nivel_riesgo": nivel,
    "features":   feats,
    "detalle":   evaluar_detalle_variables(evaluaciones, cfg, comportamiento),
    "_sin_modelo": True,
  }


def _mostrar_resultado(res: dict, nombre: str = ""):
  from src.config import FEATURE_LABELS

  prob    = res["probabilidad"]
  clase   = res["clase"]
  nivel   = res["nivel_riesgo"]
  detalle  = res["detalle"]
  nombre_txt = f" — {nombre}" if nombre else ""

  if res.get("_sin_modelo"):
    st.info("Modelo pkl no encontrado — usando predicción por reglas. Entrena el modelo para mayor precisión.")

  col_res, col_barra = st.columns([1, 2])

  with col_res:
    if clase == 1:
      st.markdown(f"""
      <div class="card-riesgo">
        <h2 style="color:#ff3355;margin:0;font-size:1.6rem;font-family:'Orbitron',monospace;letter-spacing:3px;text-shadow:0 0 15px #ff335588;">CON RIESGO{nombre_txt}</h2>
        <p style="color:#ff8899;margin:8px 0 4px 0;font-family:'Share Tech Mono',monospace;">
          Prob. reprobación: <strong style="color:#ff3355;font-size:1.1rem;">{prob*100:.1f}%</strong>
          &nbsp;&middot;&nbsp; Nivel: <span style="color:#e8640a;">{nivel}</span>
        </p>
        <p style="color:#aa4455;font-size:0.9rem;margin-top:8px;letter-spacing:1px;font-family:'Share Tech Mono',monospace;">[ Se recomienda intervención docente ]</p>
      </div>""", unsafe_allow_html=True)
    else:
      st.markdown(f"""
      <div class="card-ok">
        <h2 style="color:#00d4ff;margin:0;font-size:1.6rem;font-family:'Orbitron',monospace;letter-spacing:3px;text-shadow:0 0 15px #00d4ff88;">SIN RIESGO{nombre_txt}</h2>
        <p style="color:#55ccee;margin:8px 0 4px 0;font-family:'Share Tech Mono',monospace;">
          Prob. reprobación: <strong style="color:#00d4ff;font-size:1.1rem;">{prob*100:.1f}%</strong>
          &nbsp;&middot;&nbsp; Nivel: <span style="color:#00d4ff;">{nivel}</span>
        </p>
        <p style="color:#2e7a96;font-size:0.9rem;margin-top:8px;letter-spacing:1px;font-family:'Share Tech Mono',monospace;">[ Rendimiento dentro del umbral aceptable ]</p>
      </div>""", unsafe_allow_html=True)

  with col_barra:
    color_bar = "#ff3355" if clase == 1 else "#00d4ff"
    glow_color = "#ff335566" if clase == 1 else "#00d4ff66"
    st.markdown(f"""
    <p style="color:#4dbfe0;margin-bottom:6px;font-size:0.85rem;letter-spacing:2px;font-family:'Share Tech Mono',monospace;text-transform:uppercase;">PROBABILIDAD DE REPROBACION</p>
    <div class="prob-bar-bg">
      <div style="width:{prob*100:.1f}%;background:{color_bar};height:100%;
            display:flex;align-items:center;padding-left:14px;
            color:#fff;font-weight:700;font-family:'Orbitron',monospace;font-size:0.95rem;
            box-shadow:0 0 20px {glow_color};">
        {prob*100:.1f}%
      </div>
    </div>
    <p style="font-size:0.82rem;color:#2e7a96;margin-top:6px;letter-spacing:1px;font-family:'Share Tech Mono',monospace;">Umbral: 45%</p>
    """, unsafe_allow_html=True)

    # Features internas — ahora con etiquetas legibles de FEATURE_LABELS
    with st.expander("Vector de features del modelo (15 variables)"):
      bloques = {
        "Académico":      ["nota_ponderada","ratio_completadas","tendencia","nota_minima","bonus_extra","ratio_reprobados"],
        "Comportamiento":   ["asistencia","ratio_tareas_tiempo","visitas_tutoria"],
        "Contexto":      ["carga_academica","es_repitente"],
        "Alertas tempranas":  ["caida_brusca","varianza_notas","nota_primer_parcial","alerta_asistencia"],
      }
      for bloque, keys in bloques.items():
        st.markdown(f"<p style='color:#4dbb4d;font-size:0.8rem;margin:8px 0 2px 0;'>{bloque}</p>", unsafe_allow_html=True)
        for k in keys:
          v   = res["features"].get(k, 0)
          label = FEATURE_LABELS.get(k, k)
          color = "#00ff41" if v < 0.5 else "#ff6644"
          # Para features donde mayor = mejor, invertir color
          if k in ("nota_ponderada","ratio_completadas","asistencia","ratio_tareas_tiempo",
               "visitas_tutoria","nota_minima","nota_primer_parcial","bonus_extra","tendencia"):
            color = "#00ff41" if v >= 0.5 else "#ff6644"
          st.markdown(f"""
          <div style="margin:2px 0;font-size:0.83rem;display:flex;justify-content:space-between;">
            <span style="color:#a0e8a0;">{label}</span>
            <span style="color:{color};font-weight:700;">{v:.4f}</span>
          </div>""", unsafe_allow_html=True)

  # SHAP Explicabilidad 
  modelo_actual = _cargar_modelo()
  if modelo_actual is not None and not res.get("_sin_modelo"):
    try:
      from src.shap_explain import explicar_estudiante, shap_disponible
      if shap_disponible():
        with st.expander("Explicabilidad SHAP — ¿Por qué este resultado?"):
          with st.spinner("Calculando SHAP values…"):
            shap_res = explicar_estudiante(modelo_actual, res["features"])
          if shap_res["error"]:
            st.warning(shap_res["error"])
          else:
            st.markdown(
              "<p style='color:#4dbb4d;font-size:0.85rem;margin-bottom:8px;'>"
              "Cada barra muestra cuánto empuja ese factor hacia riesgo (+) o seguridad (−)."
              "</p>", unsafe_allow_html=True
            )
            for item in shap_res["importancias"]:
              sv  = item["shap"]
              color = "#ff4444" if sv > 0 else "#00ff41"
              pct  = min(abs(sv) * 400, 100)  # escalar para visualización
              signo = " +" if sv > 0 else " "
              st.markdown(f"""
              <div style="margin:3px 0;font-size:0.83rem;">
                <div style="display:flex;justify-content:space-between;margin-bottom:2px;">
                  <span style="color:#a0e8a0;">{item['label']}</span>
                  <span style="color:{color};font-weight:700;">{signo}{abs(sv):.4f}</span>
                </div>
                <div style="background:#0d1a0d;border-radius:4px;height:8px;overflow:hidden;">
                  <div style="width:{pct:.1f}%;background:{color};height:100%;border-radius:4px;"></div>
                </div>
              </div>""", unsafe_allow_html=True)
            base = shap_res["base_value"]
            st.markdown(
              f"<p style='color:#4dbb4d;font-size:0.78rem;margin-top:8px;'>"
              f"Valor base del modelo: {base:.4f}</p>",
              unsafe_allow_html=True
            )
    except Exception as _shap_err:
      pass  # SHAP es opcional — si falla, no rompe la UI

  # Detalle por evaluación 
  st.divider()
  alertas  = [d for d in detalle if d["estado"] == "alerta"]
  oks    = [d for d in detalle if d["estado"] == "ok"]
  pendientes = [d for d in detalle if d["estado"] == "pendiente"]

  col_a, col_b = st.columns(2)
  with col_a:
    st.markdown(f'<p class="seccion-titulo">Factores de riesgo ({len(alertas)})</p>', unsafe_allow_html=True)
    if alertas:
      for d in alertas:
        extra_tag = " <em style='color:#ffaa00'>(puntos extra)</em>" if d["es_extra"] else ""
        peso_tag = f" <span style='color:#4dbb4d'>[{d['peso']}%]</span>" if not d["es_extra"] else ""
        st.markdown(f'<div class="var-alerta"><strong>{d["nombre"]}</strong>{peso_tag}{extra_tag}<br>{d["mensaje"]}</div>', unsafe_allow_html=True)
    else:
      st.markdown('<p style="color:#00ff41;">Sin factores de riesgo.</p>', unsafe_allow_html=True)
    if pendientes:
      st.markdown(f'<p class="seccion-titulo" style="margin-top:1rem;">Pendientes ({len(pendientes)})</p>', unsafe_allow_html=True)
      for d in pendientes:
        st.markdown(f'<div class="var-pendiente"><strong>{d["nombre"]}</strong> [{d["peso"]}%]<br>{d["mensaje"]}</div>', unsafe_allow_html=True)

  with col_b:
    st.markdown(f'<p class="seccion-titulo">Aspectos favorables ({len(oks)})</p>', unsafe_allow_html=True)
    if oks:
      for d in oks:
        peso_tag = f" <span style='color:#4dbb4d'>[{d['peso']}%]</span>" if not d["es_extra"] else " <em style='color:#ffaa00'>(extra)</em>"
        st.markdown(f'<div class="var-ok"><strong>{d["nombre"]}</strong>{peso_tag}<br>{d["mensaje"]}</div>', unsafe_allow_html=True)


# 
# TABS
# 

tab1, tab2, tab3 = st.tabs([
  "INGRESO MANUAL",
  "CARGA CSV",
  "COMO FUNCIONA",
])


# TAB 1: INGRESO MANUAL 
with tab1:
  cfg1 = panel_evaluaciones_dinamicas("t1")

  if cfg1["pesos_validos"]:
    st.divider()
    st.markdown('<p class="seccion-titulo">DATOS DEL ESTUDIANTE</p>', unsafe_allow_html=True)
    nombre_est = st.text_input("NOMBRE DEL ESTUDIANTE", placeholder="Ej: Juan Pérez", key="t1_nombre_est")

    # Notas 
    st.caption("Deja en 0 los campos de evaluaciones que el estudiante AÚN NO rindió.")
    ev_plantilla   = cfg1["evaluaciones_plantilla"]
    notas_ingresadas = []
    cols = st.columns(min(len(ev_plantilla), 4))

    for i, ev in enumerate(ev_plantilla):
      col = cols[i % len(cols)]
      with col:
        is_extra  = ev["slug"] == "puntos_extra"
        escala_slug = ev.get("escala_slug", "100")
        label_ev  = f"{ev['nombre']}" + (f" [{ev['peso']}%]" if not is_extra else " [extra]")

        if escala_slug == "letra":
          raw_val = st.selectbox(
            label_ev, LETRAS_PENDIENTE,
            key=f"t1_nota_{i}",
            help="Selecciona la letra o '(pendiente)' si aún no rindió"
          )
          valor_final = None if raw_val == "(pendiente)" else raw_val
        else:
          val_max = MAXIMOS_ESCALA.get(escala_slug, 100)
          nota_val = st.number_input(
            label_ev, 0, val_max, 0,
            key=f"t1_nota_{i}",
            help=f"0 = aún no rendida | escala 0–{val_max}"
          )
          valor_final = nota_val if nota_val > 0 else None

        notas_ingresadas.append({
          "slug":    ev["slug"],
          "nombre":   ev["nombre"],
          "peso":    ev["peso"],
          "valor":    valor_final,
          "orden":    i + 1,
          "escala_slug": escala_slug,
        })

    # Comportamiento 
    st.divider()
    comp1 = panel_comportamiento("t1_comp")

    if st.button("ANALIZAR RIESGO", type="primary", key="t1_btn"):
      predecir_y_mostrar(notas_ingresadas, cfg1, comportamiento=comp1, nombre=nombre_est)


# TAB 2: CARGA CSV 
with tab2:
  cfg2 = panel_evaluaciones_dinamicas("t2")


  if cfg2["pesos_validos"]:
    st.divider()

    with st.expander("Descargar plantilla CSV para esta materia"):
      # Plantilla incluye columnas de comportamiento
      cols_plantilla = (
        ["nombre_estudiante"]
        + [ev["nombre"] for ev in cfg2["evaluaciones_plantilla"]]
        + ["asistencia_pct", "tareas_total", "tareas_a_tiempo",
          "visitas_tutoria", "materias_paralelo", "es_repitente"]
      )
      df_plantilla = pd.DataFrame(columns=cols_plantilla)
      for j in range(3):
        fila = {"nombre_estudiante": f"Estudiante {j+1}"}
        for ev in cfg2["evaluaciones_plantilla"]:
          fila[ev["nombre"]] = ""
        fila.update({
          "asistencia_pct": 85, "tareas_total": 5,
          "tareas_a_tiempo": 5, "visitas_tutoria": 0,
          "materias_paralelo": 4, "es_repitente": 0,
        })
        df_plantilla = pd.concat([df_plantilla, pd.DataFrame([fila])], ignore_index=True)

      st.dataframe(df_plantilla, use_container_width=True)
      st.caption(
        "Columnas de notas: deja vacío si no rindió. "
        "Columnas de comportamiento: asistencia_pct (0-100), es_repitente (0/1)."
      )
      csv_bytes = df_plantilla.to_csv(index=False).encode("utf-8")
      st.download_button("Descargar plantilla", csv_bytes, "plantilla_notas.csv", "text/csv")

    uploaded = st.file_uploader("Subir CSV con notas", type=["csv"], key="t2_upload")

    if uploaded:
      try:
        df_csv = pd.read_csv(uploaded)
        if "nombre_estudiante" not in df_csv.columns:
          st.error("El CSV debe tener una columna 'nombre_estudiante'")
        else:
          st.success(f"{len(df_csv)} estudiantes cargados")
          nombre_sel = st.selectbox("Seleccionar estudiante:", df_csv["nombre_estudiante"].tolist(), key="t2_sel")
          fila    = df_csv[df_csv["nombre_estudiante"] == nombre_sel].iloc[0]

          # Notas
          ev_con_notas = []
          for i, ev in enumerate(cfg2["evaluaciones_plantilla"]):
            raw    = fila.get(ev["nombre"], None)
            val    = None
            escala_ev = ev.get("escala_slug", "100")
            if raw is not None and str(raw).strip() not in ("", "nan"):
              raw_str = str(raw).strip()
              if escala_ev == "letra":
                val = raw_str.upper() if raw_str.upper() in LETRAS_VALIDAS else None
              else:
                try:
                  val = float(raw_str)
                  if val == 0: val = None
                except ValueError:
                  val = None
            ev_con_notas.append({
              "slug": ev["slug"], "nombre": ev["nombre"],
              "peso": ev["peso"], "valor": val,
              "orden": i+1, "escala_slug": escala_ev,
            })

          # Comportamiento desde CSV
          def _safe_float(f, col, default):
            raw = f.get(col, None)
            if raw is None or str(raw).strip() in ("", "nan"): return default
            try: return float(raw)
            except: return default

          comp2 = {
            "asistencia":    _safe_float(fila, "asistencia_pct", 85) / 100,
            "tareas_total":   int(_safe_float(fila, "tareas_total", 5)),
            "tareas_a_tiempo":  int(_safe_float(fila, "tareas_a_tiempo", 5)),
            "visitas_tutoria":  int(_safe_float(fila, "visitas_tutoria", 0)),
            "materias_paralelo": int(_safe_float(fila, "materias_paralelo", 4)),
            "es_repitente":   bool(_safe_float(fila, "es_repitente", 0)),
          }

          # Mostrar resumen de comportamiento leído del CSV
          st.markdown('<p class="seccion-titulo" style="font-size:0.9rem;">Comportamiento leído del CSV</p>', unsafe_allow_html=True)
          cb1, cb2, cb3 = st.columns(3)
          cb1.metric("Asistencia", f"{comp2['asistencia']*100:.0f}%")
          cb2.metric("Tareas a tiempo", f"{comp2['tareas_a_tiempo']}/{comp2['tareas_total']}")
          cb3.metric("Visitas tutoría", comp2["visitas_tutoria"])

          if st.button(f"Analizar a {nombre_sel}", type="primary", key="t2_btn"):
            predecir_y_mostrar(ev_con_notas, cfg2, comportamiento=comp2, nombre=nombre_sel)

      except Exception as e:
        st.error(f"Error al leer el CSV: {e}")



# TAB 3: CÓMO FUNCIONA 
with tab3:
  st.markdown('<p class="seccion-titulo">Cómo funciona la IA flexible</p>', unsafe_allow_html=True)
  conceptos = [
    ("Evaluaciones dinámicas",
     "El docente define cualquier combinación: 3 parciales, 1 proyecto, 2 laboratorios, puntos extra. No hay límite."),
    ("Normalización multi-escala",
     "Cada nota se convierte a [0-1] según su propia escala: sobre 100, sobre 70, sobre 20, porcentaje o letras (A+, B, F…)."),
    (" 15 features universales",
     "El modelo recibe 15 variables agrupadas en 4 bloques: académico (6), comportamiento (3), contexto (2) y alertas tempranas (4)."),
    ("Pesos re-normalizados",
     "Si el estudiante solo tiene 2 de 4 notas disponibles, los pesos se re-normalizan sobre lo disponible. La nota proyectada es justa."),
    ("Comportamiento integrado",
     "Asistencia, tareas a tiempo y visitas a tutoría son opcionales — si no se ingresan, el modelo los estima con la mediana del entrenamiento."),
    ("Alertas tempranas",
     "El sistema detecta caídas bruscas entre evaluaciones, alta varianza de notas y la nota del primer parcial como predictor temprano de riesgo."),
    ("Umbral de decisión en 0.45",
     "Se usa 0.45 en lugar de 0.50 para priorizar recall: preferimos detectar un falso positivo que perder un estudiante en riesgo real."),
  ]
  for titulo, desc in conceptos:
    st.markdown(f"""
    <div style="background:#0a150a;border:1px solid #00ff4122;border-radius:8px;
          padding:0.9rem 1.2rem;margin:0.4rem 0;font-size:0.88rem;">
      <strong style="color:#00ff41;">{titulo}</strong><br>
      <span style="color:#80c880;">{desc}</span>
    </div>""", unsafe_allow_html=True)

  st.divider()
  st.markdown('<p class="seccion-titulo">Tipos de evaluación disponibles</p>', unsafe_allow_html=True)
  tipos_df = pd.DataFrame([
    {"Slug interno": slug, "Etiqueta": label,
     "Es extra (peso 0)": "" if slug == "puntos_extra" else "—",
     "Permite recuperatorio": "" if slug in ("parcial", "laboratorio") else "—"}
    for label, slug in TIPOS_SLUGS.items()
  ])
  st.dataframe(tipos_df, use_container_width=True, hide_index=True)

  st.markdown('<p class="seccion-titulo">Las 15 features del modelo</p>', unsafe_allow_html=True)
  from src.config import FEATURES, FEATURE_LABELS
  bloques_tab4 = {
    "Bloque académico":    FEATURES[:6],
    "Bloque comportamiento":  FEATURES[6:9],
    "Bloque contexto":     FEATURES[9:11],
    "Bloque alertas":     FEATURES[11:],
  }
  for bloque, keys in bloques_tab4.items():
    st.markdown(f"**{bloque}**")
    feat_df = pd.DataFrame([
      {"Feature": k, "Descripción": FEATURE_LABELS.get(k, k)}
      for k in keys
    ])
    st.dataframe(feat_df, use_container_width=True, hide_index=True)
