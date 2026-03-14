import streamlit as st
import pandas as pd
import io
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from llm_analyzer import LLMAnalyzer
from data_processor import DataProcessor
from code_executor import CodeExecutor

# ── Configuración de página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="CELMIA",
    page_icon="🦉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS personalizado ─────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Importar fuentes ── */
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=Space+Mono:wght@400;700&display=swap');

/* ── Tokens de color ── */
:root {
    --bg:        #0f1117;
    --surface:   #191d29;
    --surface2:  #232839;
    --border:    #2e3348;
    --accent:    #6ee7b7;       /* verde menta */
    --accent2:   #818cf8;       /* violeta suave */
    --txt:       #e8eaf0;
    --txt-muted: #8892a4;
    --danger:    #f87171;
    --warn:      #fbbf24;
}

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
    background-color: var(--bg) !important;
    color: var(--txt) !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * {
    color: var(--txt) !important;
}

/* ── Título principal ── */
h1 { 
    font-weight: 700 !important;
    letter-spacing: -0.5px !important;
}

/* ── Subtítulos ── */
h2, h3 {
    font-weight: 600 !important;
    color: var(--txt) !important;
}

/* ── Tarjetas métricas ── */
[data-testid="metric-container"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 16px 20px !important;
}
[data-testid="metric-container"] label {
    color: var(--txt-muted) !important;
    font-size: 0.78rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 1.6rem !important;
    color: var(--accent) !important;
}

/* ── Botón primario ── */
.stButton > button[kind="primary"] {
    background: var(--accent) !important;
    color: #0f1117 !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.2rem !important;
    letter-spacing: 0.02em !important;
    transition: opacity 0.2s !important;
}
.stButton > button[kind="primary"]:hover { opacity: 0.85 !important; }

/* ── Botón secundario ── */
.stButton > button {
    background: var(--surface2) !important;
    color: var(--txt) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

/* ── Dataframes ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}

/* ── Expanders ── */
details {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 4px !important;
    margin-bottom: 8px !important;
}
summary { color: var(--txt-muted) !important; font-size: 0.9rem !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 2px dashed var(--border) !important;
    border-radius: 12px !important;
    padding: 12px !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
}

/* ── Alerts ── */
.stSuccess { background: #052e16 !important; border-color: var(--accent) !important; border-radius: 8px !important; }
.stInfo    { background: #1e1b4b !important; border-color: var(--accent2) !important; border-radius: 8px !important; }
.stWarning { background: #1c1400 !important; border-color: var(--warn) !important; border-radius: 8px !important; }
.stError   { background: #1a0a0a !important; border-color: var(--danger) !important; border-radius: 8px !important; }

/* ── Código ── */
.stCode pre { 
    background: #111827 !important; 
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

/* ── Separador ── */
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

/* ── Gráficas Matplotlib ── */
.stPlotlyChart, .element-container img {
    border-radius: 10px !important;
}

/* ── Chip de estado Ollama ── */
.chip-ok  { display:inline-block; background:#052e16; color:var(--accent); 
             border:1px solid var(--accent); border-radius:999px; 
             padding:3px 12px; font-size:0.78rem; font-weight:600; }
.chip-err { display:inline-block; background:#1a0a0a; color:var(--danger); 
             border:1px solid var(--danger); border-radius:999px; 
             padding:3px 12px; font-size:0.78rem; font-weight:600; }

/* ── Tabla de correlación coloreada ── */
.corr-legend { display:flex; gap:16px; align-items:center; margin-top:8px; flex-wrap:wrap; }
.corr-legend-item { display:flex; align-items:center; gap:6px; font-size:0.8rem; color: #8892a4; }
.corr-dot { width:12px; height:12px; border-radius:50%; }
</style>
""", unsafe_allow_html=True)


# ── Helpers de estilo para matplotlib ────────────────────────────────────────
def _mpl_dark_style():
    """Aplica estilo oscuro coherente a las figuras de matplotlib."""
    plt.rcParams.update({
        "figure.facecolor":  "#191d29",
        "axes.facecolor":    "#191d29",
        "axes.edgecolor":    "#2e3348",
        "axes.labelcolor":   "#8892a4",
        "text.color":        "#e8eaf0",
        "xtick.color":       "#8892a4",
        "ytick.color":       "#8892a4",
        "grid.color":        "#2e3348",
        "grid.linewidth":    0.6,
    })


# ── Init ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def init_analyzer():
    return LLMAnalyzer()


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar(analyzer):
    with st.sidebar:
        st.title("Analizador de datos con IA local")
  

        st.markdown("---")
        st.markdown("**¿Cómo funciona?**")
        st.markdown("""
1. 📂 Sube tu archivo Excel o CSV  
2. 🤖 La IA analiza tu información  
3. 📊 Se generan gráficas automáticamente  
4. 💡 Recibes conclusiones en lenguaje claro  
""")
        st.markdown("---")
        st.markdown("**Ventajas**")
        st.markdown("""
- 🔒 100 % privado — sin internet  
- ⚡ Rápido y sin límites  
- 🧩 Sin conocimientos técnicos requeridos  
                    
""")
        st.markdown("---")

        # Estado de Ollama
        if analyzer.check_ollama():
            st.markdown('<span class="chip-ok">● Ollama conectado</span>', unsafe_allow_html=True)
            st.caption(f"Modelo activo: `{analyzer.model}`")
        else:
            st.markdown('<span class="chip-err">✕ Ollama no disponible</span>', unsafe_allow_html=True)
            st.info("Ejecuta `ollama serve` en tu terminal para activarlo.")


# ── Dashboard automático ──────────────────────────────────────────────────────
def auto_dashboard(df):
    numeric_cols = df.select_dtypes(include=["number"]).columns
    if len(numeric_cols) == 0:
        return

    _mpl_dark_style()

    st.markdown("### 📊 Resumen rápido")

    # ── Métricas (máx 4 columnas) ──
    show_cols = numeric_cols[:4]
    cols = st.columns(len(show_cols))
    for i, col in enumerate(show_cols):
        with cols[i]:
            mean_val  = round(df[col].mean(), 2)
            delta_txt = f"Máx {df[col].max()} · Mín {df[col].min()}"
            st.metric(label=col, value=mean_val, delta=delta_txt)

    st.markdown(" ")

    # ── Histograma ──
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.hist(df[numeric_cols[-1]].dropna(), bins=25,
            color="#6ee7b7", edgecolor="#0f1117", linewidth=0.5)
    ax.set_title(f"Distribución de «{numeric_cols[-1]}»", fontsize=12, color="#e8eaf0", pad=10)
    ax.set_xlabel(numeric_cols[-1], fontsize=10)
    ax.set_ylabel("Frecuencia", fontsize=10)
    ax.grid(axis="y", linestyle="--")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ── Sección de correlación con explicación humana ─────────────────────────────
def render_correlation(corr_df):
    st.markdown("### 🔗 Relación entre variables numéricas")

    # Explicación en lenguaje llano
    st.markdown("""
<div style="
    background:#1e2233; 
    border-left:4px solid #6ee7b7; 
    border-radius:0 10px 10px 0;
    padding:14px 18px; 
    margin-bottom:16px;
    font-size:0.9rem;
    color:#c8cfe0;
    line-height:1.7;
">
<strong style="color:#6ee7b7;">¿Qué significa esto?</strong><br>
Esta tabla muestra si dos cosas <em>suben o bajan juntas</em>.<br><br>
• <strong style="color:#6ee7b7;">Cerca de +1</strong> → cuando una sube, la otra <strong>también sube</strong>. 
  Ejemplo: más horas de estudio → mejores calificaciones.<br>
• <strong style="color:#f87171;">Cerca de −1</strong> → cuando una sube, la otra <strong>baja</strong>. 
  Ejemplo: más velocidad → menos tiempo de viaje.<br>
• <strong style="color:#8892a4;">Cerca de 0</strong> → no hay relación clara entre ellas.<br><br>
<em>Tip: busca los números más alejados de cero — ahí están las pistas más interesantes de tu dataset.</em>
</div>
""", unsafe_allow_html=True)

    # Leyenda visual
    st.markdown("""
<div class="corr-legend">
  <div class="corr-legend-item">
    <div class="corr-dot" style="background:#6ee7b7;"></div> Relación positiva (sube junto)
  </div>
  <div class="corr-legend-item">
    <div class="corr-dot" style="background:#f87171;"></div> Relación negativa (se oponen)
  </div>
  <div class="corr-legend-item">
    <div class="corr-dot" style="background:#374151;"></div> Sin relación clara
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown(" ")

    # Mapa de calor con matplotlib
    _mpl_dark_style()
    import numpy as np

    fig, ax = plt.subplots(figsize=(max(5, len(corr_df)*0.9), max(4, len(corr_df)*0.75)))
    data    = corr_df.values
    labels  = corr_df.columns.tolist()

    # Paleta divergente verde-rojo
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "celmia", ["#f87171", "#191d29", "#6ee7b7"]
    )
    im = ax.imshow(data, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

    # Etiquetas de ejes
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    # Valores dentro de cada celda
    for i in range(len(labels)):
        for j in range(len(labels)):
            val   = data[i, j]
            color = "#0f1117" if abs(val) > 0.5 else "#e8eaf0"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8.5, color=color, fontweight="600")

    # Barra de color
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
    cbar.ax.tick_params(colors="#8892a4", labelsize=8)

    ax.set_title("Mapa de correlación", fontsize=12, color="#e8eaf0", pad=12)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ── Vista previa y estadísticas ───────────────────────────────────────────────
def render_data_overview(df):
    with st.expander("👁️ Vista previa de los datos (primeras 10 filas)"):
        st.dataframe(df.head(10), use_container_width=True)

    with st.expander("📋 Resumen estadístico"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Información general**")
            buffer = io.StringIO()
            df.info(buf=buffer)
            st.text(buffer.getvalue())
        with col2:
            st.markdown("**Estadísticas descriptivas**")
            st.dataframe(df.describe(), use_container_width=True)


# ── Análisis IA ───────────────────────────────────────────────────────────────
def analyze_with_ai(df, processor, analyzer):
    with st.spinner("🧠 Analizando con IA…"):
        dataset_description = processor.describe_dataset(df)
        st.markdown("### 🔍 Análisis de la IA")

        analysis_plan = analyzer.generate_analysis_plan(dataset_description)
        with st.expander("📝 Plan de análisis generado"):
            st.write(analysis_plan)

        code = analyzer.generate_analysis_code(df, dataset_description)
        with st.expander("💻 Código Python generado"):
            st.code(code, language="python")

        executor = CodeExecutor()
        results  = executor.execute_code(code, df)

        if not results["success"]:
            st.warning("⚠️ Corrigiendo código automáticamente…")
            fixed_code = analyzer.fix_code(code, results["error"])
            results    = executor.execute_code(fixed_code, df)

        if results["success"]:
            st.success("✅ Análisis completado exitosamente")

            if results.get("output"):
                st.markdown("### 📊 Resultados")
                st.text(results["output"])

            if results.get("plots"):
                st.markdown("### 📈 Visualizaciones")
                cols = st.columns(2)
                for idx, plot in enumerate(results["plots"]):
                    with cols[idx % 2]:
                        st.pyplot(plot)

            insights = analyzer.generate_insights(dataset_description, results)
            st.markdown("### 💡 Conclusiones clave")
            st.markdown(insights)

            st.download_button(
                "📄 Descargar reporte completo",
                _generate_report(df, analysis_plan, code, insights),
                "reporte_celmia.txt",
                "text/plain"
            )

        else:
            st.error("❌ No se pudo ejecutar el análisis")
            with st.expander("🔍 Detalle del error", expanded=True):
                st.code(results.get("error", "Error desconocido"), language="text")
            with st.expander("💻 Código que causó el error"):
                st.code(results.get("code_used", code), language="python")
            st.info("💡 Intenta hacer clic en «Generar análisis» de nuevo — el código será diferente.")
            if st.button("🔄 Reintentar", type="primary"):
                st.rerun()
            if results.get("output"):
                st.markdown("📝 **Salida parcial**")
                st.text(results["output"])


# ── Dataset de ejemplo ────────────────────────────────────────────────────────
def create_sample_dataset():
    import numpy as np
    np.random.seed(42)
    n = 100
    data = {
        "fecha":    pd.date_range("2024-01-01", periods=n, freq="D"),
        "ventas":   np.random.randint(100, 1000, n),
        "clientes": np.random.randint(10, 100, n),
        "producto": np.random.choice(["A", "B", "C"], n),
        "region":   np.random.choice(["Norte", "Sur", "Este", "Oeste"], n),
        "costo":    np.random.uniform(50, 500, n).round(2),
    }
    df = pd.DataFrame(data)
    df["ganancia"] = df["ventas"] - df["costo"]
    return df


def _generate_report(df, plan, code, insights):
    return f"""REPORTE DE ANÁLISIS — CELMIA
============================

DATASET
-------
Filas:    {df.shape[0]}
Columnas: {df.shape[1]}
Nombres:  {', '.join(df.columns.tolist())}

PLAN DE ANÁLISIS
----------------
{plan}

CÓDIGO GENERADO
---------------
{code}

CONCLUSIONES
------------
{insights}

---
Generado con CELMIA · 100 % privado y offline
"""


# ── App principal ─────────────────────────────────────────────────────────────
def main():
    analyzer = init_analyzer()
    render_sidebar(analyzer)

    # ── Encabezado ──
    st.markdown("# 🦉 CELMIA")
    st.markdown(
        "<p style='color:#8892a4; font-size:1.05rem; margin-top:-8px;'>"
        "Sube tu dataset y obtén conclusiones claras, sin necesitar conocimientos técnicos."
        "</p>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    # ── Carga de archivo ──
    uploaded_file = st.file_uploader(
        "Arrastra tu archivo aquí o haz clic para buscarlo",
        type=["csv", "xlsx", "xls"],
        help="Formatos aceptados: CSV, Excel (.xlsx, .xls)",
        label_visibility="visible"
    )

    if uploaded_file is not None:
        try:
            processor = DataProcessor()
            df        = processor.load_file(uploaded_file)

            # Confirmación
            c1, c2 = st.columns(2)
            c1.success(f"✅ {uploaded_file.name} cargado correctamente")
            c2.info(f"📐 {df.shape[0]:,} filas · {df.shape[1]} columnas")

            st.markdown("---")

            # Dashboard rápido
            auto_dashboard(df)

            st.markdown("---")

            # Correlación
            metrics = processor.basic_metrics(df)
            if "correlation" in metrics:
                render_correlation(metrics["correlation"])
                st.markdown("---")

            # Vista previa + estadísticas
            render_data_overview(df)

            st.markdown("---")

            # Botón IA
            st.markdown("### 🤖 Análisis profundo con IA")
            st.caption(
                "La IA leerá tu dataset, escribirá código de análisis, "
                "generará gráficas y te explicará los hallazgos en español claro."
            )
            if st.button("✨ Generar análisis inteligente", type="primary", use_container_width=True):
                analyze_with_ai(df, processor, analyzer)

        except Exception as e:
            st.error(f"❌ No se pudo procesar el archivo: {e}")
            st.info("Asegúrate de que el archivo sea un CSV o Excel válido.")

    else:
        # Estado vacío
        st.markdown(
            """
<div style="
    text-align:center; 
    padding:60px 20px; 
    color:#8892a4;
    border:2px dashed #2e3348;
    border-radius:16px;
    background:#191d29;
">
    <div style="font-size:3rem; margin-bottom:12px;">📂</div>
    <div style="font-size:1.15rem; font-weight:600; color:#e8eaf0;">
        Todavía no hay datos
    </div>
    <div style="font-size:0.9rem; margin-top:8px;">
        Sube un archivo CSV o Excel para comenzar
    </div>
</div>
""",
            unsafe_allow_html=True
        )

        st.markdown(" ")
        with st.expander("💡 ¿No tienes un dataset? Prueba con un ejemplo"):
            st.caption("Genera datos de ventas ficticias para explorar la herramienta.")
            if st.button("🎲 Generar dataset de ejemplo"):
                sample_df = create_sample_dataset()
                st.dataframe(sample_df, use_container_width=True)
                st.download_button(
                    "📥 Descargar ejemplo (CSV)",
                    sample_df.to_csv(index=False),
                    "dataset_ejemplo.csv",
                    "text/csv"
                )


if __name__ == "__main__":
    main()