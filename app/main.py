import streamlit as st
import pandas as pd
from PIL import Image
import sys
import os

# Configuración de rutas
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from inference import MemePredictor

# --- 1. CONFIGURACIÓN DE PÁGINA (Minimalista) ---
st.set_page_config(
    page_title="DIME-MEX",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- 2. ESTILOS CSS PERSONALIZADOS (Para móvil) ---
st.markdown("""
    <style>
        /* Ocultar menú hamburguesa y footer para limpieza visual */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Ajustar padding para móviles */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        
        /* Botón primario minimalista (Blanco y Negro o color acento del tema) */
        div.stButton > button:first-child {
            width: 100%;
            border-radius: 5px;
            height: 3em;
            font-weight: bold;
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
    st.error(f"Error iniciando el sistema: {e}")
    st.stop()

# --- 4. INTERFAZ DE USUARIO ---

# Título limpio
st.markdown("## Detector de Contenido")
st.markdown("Clasificación de memes utilizando visión y lenguaje natural.")

# ACORDEÓN DE CONFIGURACIÓN
with st.expander("Configuración del Modelo"):
    task_mode = st.radio(
        "Nivel de detalle:",
        options=["simple", "complex"],
        format_func=lambda x: "Básico (3 Categorías)" if x == "simple" else "Detallado (6 Categorías)"
    )

# ÁREA DE DRAG AND DROP
uploaded_file = st.file_uploader("Cargar imagen", type=["jpg", "png", "jpeg"], help="Arrastra tu archivo aquí")

if uploaded_file is not None:
    # Mostrar imagen centrada y ajustada al ancho del móvil
    image = Image.open(uploaded_file)
    st.image(image, use_column_width=True)
    
    # Botón de acción (ocupa todo el ancho por el CSS)
    if st.button("ANALIZAR IMAGEN", type="primary"):
        
        with st.spinner("Procesando..."):
            try:
                # Inferencia
                result = predictor.predict(uploaded_file, task=task_mode)
                
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.divider() # Línea separadora sutil
                    
                    # --- RESULTADOS MINIMALISTAS (st.metric) ---
                    # Usamos columnas para que se vea bien en celular (uno al lado del otro o apilados)
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(label="Clasificación", value=result['label'])
                    
                    with col2:
                        # Formato de porcentaje limpio
                        st.metric(label="Confianza", value=f"{result['confidence']:.1%}")
                    
                    # Barra de progreso simple
                    st.progress(result['confidence'])
                    
                    # --- DETALLES TÉCNICOS (Ocultos por defecto) ---
                    with st.expander("Ver texto extraído"):
                        st.caption("Texto detectado por OCR:")
                        st.text(result["ocr_text"])
                        st.caption("Texto procesado para el modelo:")
                        st.code(result["clean_text"], language="text")
                    
                    # --- GRÁFICA LIMPIA ---
                    st.caption("Distribución de probabilidades")
                    chart_data = pd.DataFrame({
                        "Categoría": result['all_labels'],
                        "Probabilidad": result['probabilities']
                    })
                    # Gráfica de barras horizontal es mejor para leer etiquetas largas en móvil
                    st.bar_chart(chart_data.set_index("Categoría"), color="#333333")

            except Exception as e:
                st.error(f"Error interno: {e}")