import streamlit as st
import pandas as pd
from PIL import Image
import sys
import os

# Definir path para importar módulos locales
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, '../src')
sys.path.append(os.path.abspath(src_path))

try:
    from inference import MemePredictor
except ImportError as e:
    st.error(f"Error crítico de importación: {e}")
    st.stop()

# configuración de la página
st.set_page_config(
    page_title="DIME-MEX",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# estilos
st.markdown("""
    <style>
        /* Tipografía limpia (System Fonts) */
        html, body, [class*="css"] {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            color: #1d1d1f;
        }
        
        /* Ocultar elementos nativos de Streamlit */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Contenedor principal con más aire */
        .block-container {
            padding-top: 3rem;
            padding-bottom: 3rem;
            max-width: 700px;
        }
/* --- BOTÓN ANALIZAR --- */
        div.stButton > button:first-child {
            background-color: #0071e3; /* Azul Apple */
            color: white;
            border: none;
            border-radius: 12px;
            padding: 15px 30px; /* Más grande */
            font-size: 18px;    /* Texto más grande */
            font-weight: 600;
            width: 100%;
            transition: all 0.2s ease;
            box-shadow: 0 4px 6px rgba(0, 113, 227, 0.2);
        }
        
        div.stButton > button:first-child:hover {
            background-color: #0077ED;
            transform: scale(1.01);
        }

        /* --- RADIO BUTTON (La bolita azul) --- */
        div[role="radiogroup"] label > div:first-child {
            background-color: #0071e3 !important;
            border-color: #0071e3 !important;
        }
        /* Estilo de las Métricas (Tarjetas grises tipo iOS) */
        div[data-testid="stMetric"] {
            background-color: #f5f5f7;
            border-radius: 18px;
            padding: 20px;
            text-align: center;
            border: 1px solid #e5e5e5;
        }
        
        div[data-testid="stMetricLabel"] {
            font-size: 14px;
            color: #86868b;
        }
        
        div[data-testid="stMetricValue"] {
            font-size: 24px;
            font-weight: 600;
            color: #1d1d1f;
        }

        /* Imágenes con bordes redondeados */
        img {
            border-radius: 18px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        }
        
        /* File Uploader más limpio */
        div[data-testid="stFileUploader"] {
            margin-top: 20px;
            margin-bottom: 20px;
        }
        
        /* Títulos */
        h1, h2, h3 {
            font-weight: 600;
            letter-spacing: -0.02em;
        }
    </style>
""", unsafe_allow_html=True)

# --- 3. CARGA DEL MOTOR ---
@st.cache_resource
def get_engine():
    return MemePredictor()

try:
    predictor = get_engine()
except Exception as e:
    st.error("El servicio no está disponible en este momento.")
    st.stop()


# Encabezado
st.markdown("<h1 style='text-align: center; margin-bottom: 10px;'>DIME-MEX</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #86868b; font-size: 18px;'>Análisis de contenido mediante IA.</p>", unsafe_allow_html=True)

st.write("") # Espaciador

# Configuración (Acordeón limpio)
with st.expander("Elegir modelo"):
    task_mode = st.radio(
        "Sensibilidad del modelo",
        options=["simple", "complex"],
        format_func=lambda x: "Estándar ( none, inappropriate, hate-speech )" if x == "simple" else "Detallado ( none, inappropriate, sexism, racism, classicis, hate-speech )",
        label_visibility="collapsed"
    )

# Área de Carga
uploaded_file = st.file_uploader("Subir imagen", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    
if uploaded_file is not None:
    # Mostrar imagen
    image = Image.open(uploaded_file)
    st.image(image, use_container_width=True)
    
    # Espaciador
    st.write("") 
    
    # Botón de Acción
    if st.button("Analizar"):
        with st.spinner("Analizando..."):
            try:
                result = predictor.predict(uploaded_file, task=task_mode)
                
                if "error" in result:
                    st.error("No se pudo procesar la imagen.")
                else:
                    st.write("") # Espaciador
                    
                    # --- RESULTADOS (Tarjetas) ---
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(label="Clasificación", value=result['label'])
                    
                    with col2:
                        st.metric(label="Certeza", value=f"{result['confidence']:.1%}")
                    
                    # --- DETALLES (Acordeón inferior) ---
                    st.write("")
                    with st.expander("Detalles del análisis"):
                        st.caption("TEXTO DETECTADO")
                        st.markdown(f"_{result['ocr_text']}_")
                        
                        st.write("")
                        st.caption("PROBABILIDADES")
                        # Crear dataframe limpio para la gráfica
                        chart_data = pd.DataFrame({
                            "Categoría": result['all_labels'],
                            "Probabilidad": result['probabilities']
                        })
                        st.bar_chart(chart_data.set_index("Categoría"), color="#86868b")

            except Exception:
                st.error("Ocurrió un error inesperado durante el análisis.")