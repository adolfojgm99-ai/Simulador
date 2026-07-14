import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Configuración de la página
st.set_page_config(page_title="Simulador Solar IA", layout="centered")

st.title("☀️ Simulador de Generación Fotovoltaica impulsado por IA")
st.markdown("Sube un archivo climático horario y el algoritmo maestro XGBoost calculará la viabilidad energética de la instalación para la región de Xalapa.")

# Cargar modelos en memoria caché para mayor velocidad
@st.cache_resource
def load_models():
    modelo = joblib.load('modelo_xgboost_solar.pkl')
    scaler = joblib.load('scaler_X.pkl')
    return modelo, scaler

try:
    modelo_xgb, scaler_X = load_models()
except Exception as e:
    st.error("Error al cargar los modelos. Asegúrate de haber subido los archivos .pkl a GitHub.")

archivo_csv = st.file_uploader("📂 Sube tu archivo meteorológico (.csv)", type=['csv'])

if archivo_csv is not None:
    try:
        # Limpieza y preparación de datos
        df_clima = pd.read_csv(archivo_csv, skiprows=2)
        df_diurno = df_clima[df_clima['GHI'] > 0].copy()
        
        # Escalado y Predicción
        X_input = df_diurno[['GHI', 'Temperature', 'Relative Humidity', 'Wind Speed']]
        X_escalado = scaler_X.transform(X_input)
        
        predicciones = modelo_xgb.predict(X_escalado)
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
        
        # Columnas para la tabla y la gráfica
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
