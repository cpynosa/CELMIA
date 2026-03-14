import streamlit as st
import pandas as pd
import io
from llm_analyzer import LLMAnalyzer
from data_processor import DataProcessor
from code_executor import CodeExecutor

# Configuración de la página
st.set_page_config(
    page_title="CELMIA",
    page_icon="🦉",
    layout="wide"
)

# Inicializar componentes
@st.cache_resource
def init_analyzer():
    return LLMAnalyzer()

def main():
    st.title("📊 Analizador de datos potenciado con IA")
    st.markdown("""
    ### Sube tu dataset y obtén insights valiosos automáticamente
    **Funciona 100% offline con IA local**
    """)
    
    # Sidebar con información
    with st.sidebar:
        st.title("🦉 CELMIA")
        st.header("🧭 Información")
        st.markdown("""
        **¿Cómo funciona?**
        1. Sube tu archivo CSV o Excel
        2. La IA analiza tus datos
        3. Genera visualizaciones y código automáticamente
        4. Obtén insights accionables
        
        **Requisitos:**
        - Ollama instalado
        
        **Ventajas:**
        - 🔒 100% privado y offline
        - 🚀 Análisis proporcionados por IA
        - 📈 Visualizaciones automáticas
        - 💡 Insights accionables
                    
        Usando el modelo: {analyzer.model}
        """)
        
        # Verificar Ollama
        analyzer = init_analyzer()
        if analyzer.check_ollama():
            st.success("Ollama conectado")
        else:
            st.error("Ollama no disponible")
            st.info("Ejecuta: `ollama serve` en terminal")
    
    # Área principal
    uploaded_file = st.file_uploader(
        "Sube tu dataset (CSV o Excel)",
        type=['csv', 'xlsx', 'xls'],
        help="Formatos soportados: CSV, Excel (.xlsx, .xls)"
    )
    
    if uploaded_file is not None:
        try:

            # Procesar archivo
            processor = DataProcessor()
            df = processor.load_file(uploaded_file)

            auto_dashboard(df)
            
            metrics = processor.basic_metrics(df)
            st.subheader("📈 Métricas Rápidas del Dataset")

            if "correlation" in metrics:
                st.write(f"**Correlación entre variables numéricas:**")
                st.dataframe(metrics['correlation'])
            
                        
            st.success(f"✅ Archivo cargado: {uploaded_file.name}")
            st.info(f"🖼️ Dimensiones: {df.shape[0]} filas × {df.shape[1]} columnas")
            
            # Mostrar preview
            with st.expander("👁️ Vista previa de datos"):
                st.dataframe(df.head(10), use_container_width=True)
            
            # Análisis exploratorio básico
            with st.expander("📋 Resumen estadístico"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Información general:**")
                    buffer = io.StringIO()
                    df.info(buf=buffer)
                    st.text(buffer.getvalue())
                
                with col2:
                    st.write("**Estadísticas descriptivas:**")
                    st.dataframe(df.describe(), use_container_width=True)
            
            # Botón de análisis con IA
            st.markdown("---")
            if st.button("Generar Análisis Inteligente", type="primary", use_container_width=True):
                analyze_with_ai(df, processor, analyzer)
                
        except Exception as e:
            st.error(f"❌ Error al procesar archivo: {str(e)}")
            st.info("Asegúrate de que el archivo tenga el formato correcto")
    
    else:
        # Mostrar ejemplo cuando no hay archivo
        st.info("👆 Sube un archivo para comenzar el análisis")
        
        with st.expander("💡 ¿No tienes un dataset? Prueba con un ejemplo"):
            if st.button("Generar Dataset de Ejemplo"):
                sample_df = create_sample_dataset()
                st.dataframe(sample_df, use_container_width=True)
                st.download_button(
                    "📥 Descargar ejemplo",
                    sample_df.to_csv(index=False),
                    "dataset_ejemplo.csv",
                    "text/csv"
                )

def auto_dashboard(df):

    st.subheader(f"📊 Dashboard automático sobre {df.select_dtypes(include=['number']).columns[0]}")

    numeric_cols = df.select_dtypes(include=['number']).columns

    if len(numeric_cols) > 0:

        col = numeric_cols[0]

        st.metric(
            label=f"Promedio de {col}",
            value=round(df[col].mean(),2)
        )

        st.metric(
            label=f"Máximo de {col}",
            value=df[col].max()
        )

        st.metric(
            label=f"Mínimo de {col}",
            value=df[col].min()
        )

    if len(numeric_cols) > 0:

        import matplotlib.pyplot as plt

        plt.figure()

        df[numeric_cols[0]].hist()

        st.pyplot(plt)

def analyze_with_ai(df, processor, analyzer):
    """Ejecuta el análisis completo con IA"""
    
    with st.spinner("🧠 Analizando tus datos con IA..."):
        # 1. Generar descripción del dataset
        dataset_description = processor.describe_dataset(df)
        
        st.subheader("🔍 Análisis de la IA")
        
        # 2. LLM genera plan de análisis
        analysis_plan = analyzer.generate_analysis_plan(dataset_description)
        
        with st.expander("📝 Plan de Análisis Generado"):
            st.write(analysis_plan)
        
        # 3. Generar código de análisis
        code = analyzer.generate_analysis_code(df, dataset_description)
        
        with st.expander("💻 Código Python Generado"):
            st.code(code, language="python")
        
        # 4. Ejecutar código
        executor = CodeExecutor()

        results = executor.execute_code(code, df)

        if not results["success"]:

            st.warning("⚠️ Corrigiendo código automáticamente...")

            fixed_code = analyzer.fix_code(code, results["error"])

            results = executor.execute_code(fixed_code, df)
        
        if results['success']:
            st.success("✅ Análisis completado exitosamente")
            
            # Mostrar resultados
            if results.get('output'):
                st.subheader("📊 Resultados del Análisis")
                st.text(results['output'])
            
            if results.get('plots'):
                st.subheader("📈 Visualizaciones")
                cols = st.columns(2)
                for idx, plot in enumerate(results['plots']):
                    with cols[idx % 2]:
                        st.pyplot(plot)
            
            # 5. Generar insights finales
            insights = analyzer.generate_insights(dataset_description, results)
            
            st.subheader("💡 Insights Clave")
            st.markdown(insights)
            
            # Descargar reporte
            st.download_button(
                "📄 Descargar Reporte Completo",
                generate_report(df, analysis_plan, code, insights),
                "reporte_analisis.txt",
                "text/plain"
            )
            
        else:
            st.error("❌ Error en la Ejecución del Código")
            
            # Mostrar error detallado
            with st.expander("🔍 Ver Detalles del Error", expanded=True):
                st.code(results.get('error', 'Error desconocido'), language="text")
            
            # Mostrar el código que causó el error
            with st.expander("💻 Código que Causó el Error"):
                st.code(results.get('code_used', code), language="python")
            
            # Sugerencias
            st.info("""
            **💡 Sugerencias:**
            - El modelo puede generar código con errores ocasionalmente
            - Intenta hacer clic en "Generar Análisis" de nuevo (el código será diferente)
            - Si el error persiste, el dataset podría tener un formato inusual
            """)
            
            # Botón para reintentar
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Reintentar Análisis", type="primary"):
                    st.rerun()
            
            # Mostrar output parcial si existe
            if results.get('output'):
                st.subheader("📝 Salida Parcial (antes del error)")
                st.text(results['output'])

def create_sample_dataset():
    """Crea un dataset de ejemplo para demostración"""
    import numpy as np
    
    np.random.seed(42)
    n = 100
    
    data = {
        'fecha': pd.date_range('2024-01-01', periods=n, freq='D'),
        'ventas': np.random.randint(100, 1000, n),
        'clientes': np.random.randint(10, 100, n),
        'producto': np.random.choice(['A', 'B', 'C'], n),
        'region': np.random.choice(['Norte', 'Sur', 'Este', 'Oeste'], n),
        'costo': np.random.uniform(50, 500, n).round(2)
    }
    
    df = pd.DataFrame(data)
    df['ganancia'] = df['ventas'] - df['costo']
    
    return df

def generate_report(df, plan, code, insights):
    """Genera reporte de texto completo"""
    report = f"""
REPORTE DE ANÁLISIS DE DATOS
=============================
Generado automáticamente por CELMIA

INFORMACIÓN DEL DATASET
-----------------------
Filas: {df.shape[0]}
Columnas: {df.shape[1]}
Columnas: {', '.join(df.columns.tolist())}

PLAN DE ANÁLISIS
----------------
{plan}

CÓDIGO GENERADO
---------------
{code}

INSIGHTS CLAVE
--------------
{insights}

---
Generado con CELMIA - 100% privado y offline
    """
    return report

if __name__ == "__main__":
    main()
