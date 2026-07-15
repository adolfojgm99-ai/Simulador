import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import numpy as np
import os

# 1. Configuración general de la página
st.set_page_config(page_title="Simulador Solar IA", layout="centered")

# 2. Encabezados Académicos e Institucionales
st.markdown("<p style='text-align: center; color: gray;'>Programa Delfín 2026 | Estancia Internacional de Investigación</p>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Universidad Veracruzana 🤝 Universidad Tecnológica de Bolívar</h4>", unsafe_allow_html=True)
st.markdown("---")

st.title("☀️ Simulador de Generación Fotovoltaica impulsado por IA")
st.markdown("Sube un archivo climático horario y el algoritmo maestro XGBoost calculará la viabilidad energética de la instalación para la región de Xalapa.")

# 3. Sección Técnica Desplegable (Hardware)
with st.expander("⚙️ Ver Ficha Técnica del Sistema Base (Hardware Modelado)"):
    st.info("Las estimaciones de esta Inteligencia Artificial están calibradas bajo las curvas de eficiencia del siguiente hardware:")
    
    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Módulo Fotovoltaico**")
        st.markdown("* **Marca/Modelo:** Trina Solar Vertex N (TSM-NEG19RC.20)")
        st.markdown("* **Capacidad Unitaria:** 630 W")
        st.markdown("* **Eficiencia:** 23.3%")
        
        # Corrección del botón: leer el archivo en bytes antes de descargarlo
        if os.path.exists("panel.pdf"):
            with open("panel.pdf", "rb") as pdf_file:
                pdf_bytes = pdf_file.read()
                st.download_button(label="📥 Descargar Ficha del Panel", data=pdf_bytes, file_name="Ficha_Panel.pdf", mime='application/pdf')

with colB:
        st.markdown("**Inversor Central (Base de Datos SAM/CEC)**")
        st.markdown("* **Marca/Modelo:** Fronius Primo 5.0-1 (240V)")
        st.markdown("* **Potencia Máxima CA:** 5.0 kW (5000 Wac)")
        st.markdown("* **Eficiencia Ponderada CEC:** 96.93%")
        st.markdown("* **Voltaje Nominal CA:** 240 Vac")
        
        # Botón de descarga
        if os.path.exists("inversor.pdf"):
            with open("inversor.pdf", "rb") as pdf_file:
                pdf_bytes = pdf_file.read()
                st.download_button(label="📥 Descargar Ficha del Inversor", data=pdf_bytes, file_name="Ficha_Inversor.pdf", mime='application/pdf')
st.markdown("---")

# 4. Carga del Cerebro de la IA en Caché
@st.cache_resource
def load_models():
    modelo = joblib.load('modelo_xgboost_solar.pkl')
    scaler = joblib.load('scaler_X.pkl')
    return modelo, scaler

try:
    modelo_xgb, scaler_X = load_models()
except Exception as e:
    st.error("Error al cargar los modelos. Asegúrate de haber subido los archivos .pkl a GitHub.")

# 5. Interfaz de subida de archivos
archivo_csv = st.file_uploader("📂 Sube tu archivo meteorológico (.csv)", type=['csv'])

if archivo_csv is not None:
    try:
        # Limpieza y preparación de datos
        df_clima = pd.read_csv(archivo_csv, skiprows=2)
        df_diurno = df_clima[df_clima['GHI'] > 0].copy()
        
        # Extracción de variables
        X_input = df_diurno[['GHI', 'Temperature', 'Relative Humidity', 'Wind Speed']]
        
        # Predicción DIRECTA (Sin usar el escalador, tal como aprendió el XGBoost)
        predicciones = modelo_xgb.predict(X_input)
        
        # Filtro físico: la energía fotovoltaica no puede ser negativa
        predicciones = np.maximum(0, predicciones)
        
        df_diurno['Generacion_Estimada (kW)'] = predicciones
        energia_total_kwh = df_diurno['Generacion_Estimada (kW)'].sum()
        
        st.success(f"⚡ Generación Anual Proyectada: **{energia_total_kwh:,.2f} kWh/año**")
        
        # Agrupación mensual
        resumen_mensual = df_diurno.groupby('Month').agg(
            Energia_Mensual_kWh=('Generacion_Estimada (kW)', 'sum')
        ).reset_index()
        
        nombres_meses = {1: 'Ene', 2: 'Feb', 3: 'Mar', 4: 'Abr', 5: 'May', 6: 'Jun',
                         7: 'Jul', 8: 'Ago', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dic'}
        resumen_mensual['Mes'] = resumen_mensual['Month'].map(nombres_meses)
        
        # Despliegue de Resultados (Tabla y Gráfica)
        col1, col2 = st.columns([1, 2.5])
        with col1:
            st.write("📊 **Desglose Mensual**")
            st.dataframe(resumen_mensual[['Mes', 'Energia_Mensual_kWh']].round(2), hide_index=True)
        
        with col2:
            st.write("📈 **Curva de Generación**")
            fig = plt.figure(figsize=(8, 4))
            plt.bar(resumen_mensual['Mes'], resumen_mensual['Energia_Mensual_kWh'], color='darkorange', edgecolor='black', alpha=0.8)
            plt.ylabel('Energía (kWh)')
            plt.grid(axis='y', linestyle='--', alpha=0.6)
            plt.tight_layout()
            st.pyplot(fig)
            
    except Exception as e:
        st.error(f"❌ Error al procesar el archivo. Asegúrate de subir el formato correcto. Detalle: {str(e)}")
