import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Configuración de la página
st.set_page_config(page_title="Cálculo de Rendimiento Yx/s", layout="wide")

st.title("Calculadora de Rendimiento Biomasa/Sustrato ($Y_{X/S}$)")
st.markdown("""
Esta aplicación determina el rendimiento celular utilizando el método de regresión lineal 
sobre los diferenciales acumulados, ajustándose a la expresión:
$$Y_{X/S} = \\frac{\Delta X}{-\Delta S}$$
""")

# --- SECCIÓN 1: Ingreso de Datos ---
st.sidebar.header("1. Ingreso de Datos")

# Opción para cargar ejemplo o iniciar en blanco
if st.sidebar.checkbox("Cargar datos de ejemplo"):
    data_init = {
        'Tiempo (h)': [0, 2, 4, 6, 8, 10],
        'Biomasa_X (g/L)': [0.5, 1.2, 2.8, 5.5, 9.1, 12.0],
        'Sustrato_S (g/L)': [30.0, 28.5, 25.0, 18.0, 9.5, 2.0]
    }
    df = pd.DataFrame(data_init)
else:
    # Tabla vacía inicial
    df = pd.DataFrame(columns=['Tiempo', 'Biomasa_X', 'Sustrato_S'])

st.write("### Tabla de Datos Experimentales")
st.info("Edita los valores directamente en la tabla o añade nuevas filas.")
df_input = st.data_editor(df, num_rows="dynamic", use_container_width=True)

# --- SECCIÓN 2: Selección de Variables ---
st.sidebar.header("2. Configuración de Variables")

if not df_input.empty and df_input.shape[1] >= 2:
    cols = df_input.columns.tolist()
    
    col_X = st.sidebar.selectbox("Seleccione columna de Biomasa (X):", cols, index=1 if len(cols)>1 else 0)
    col_S = st.sidebar.selectbox("Seleccione columna de Sustrato (S):", cols, index=2 if len(cols)>2 else 0)
    
    if st.sidebar.button("Calcular Rendimiento"):
        try:
            # Preparación de datos (convertir a numérico por seguridad)
            X_raw = pd.to_numeric(df_input[col_X], errors='coerce')
            S_raw = pd.to_numeric(df_input[col_S], errors='coerce')
            
            # Eliminación de NaNs
            valid_idx = X_raw.notna() & S_raw.notna()
            X = X_raw[valid_idx].values
            S = S_raw[valid_idx].values
            
            # --- CÁLCULO DE DELTAS (ACUMULADOS) ---
            # Delta X = X_t - X_0
            delta_X = X - X[0]
            
            # -Delta S = S_0 - S_t (Sustrato consumido)
            minus_delta_S = S[0] - S
            
            # Filtrar el punto (0,0) si se desea forzar intersección o evitar ruido en t=0
            # Para este ejercicio, usamos todos los puntos para la regresión
            
            # --- REGRESIÓN LINEAL (OLS) ---
            # Variable Independiente (X del modelo): -Delta S
            # Variable Dependiente (Y del modelo): Delta X
            
            # Añadir constante para evaluar el intercepto (debería ser cercano a 0 teóricamente)
            x_model = sm.add_constant(minus_delta_S) 
            model = sm.OLS(delta_X, x_model)
            results = model.fit()
            
            slope = results.params[1] # Pendiente (Yx/s)
            intercept = results.params[0]
            r_squared = results.rsquared
            
            # --- SALIDA DE RESULTADOS ---
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.write("### Resultados Estadísticos")
                st.success(f"**Rendimiento ($Y_{{X/S}}$): {slope:.4f} g/g**")
                
                # Tabla de parámetros
                stats_df = pd.DataFrame({
                    "Parámetro": ["Pendiente ($Y_{X/S}$)", "Intercepto", "$R^2$", "Error Estándar", "Valor-P"],
                    "Valor": [slope, intercept, r_squared, results.bse[1], results.pvalues[1]]
                })
                st.table(stats_df)
                
                # Ecuación de la recta
                sign = "+" if intercept >= 0 else "-"
                st.latex(f"\\Delta X = {slope:.4f} (-\\Delta S) {sign} {abs(intercept):.4f}")
                
            with col2:
                st.write("### Gráfico de Correlación")
                fig, ax = plt.subplots()
                
                # Puntos experimentales
                ax.scatter(minus_delta_S, delta_X, color='blue', label='Datos Experimentales', s=100, alpha=0.7, edgecolors='k')
                
                # Línea de regresión
                x_pred = np.linspace(min(minus_delta_S), max(minus_delta_S), 100)
                y_pred = intercept + slope * x_pred
                ax.plot(x_pred, y_pred, color='red', linestyle='--', label=f'Ajuste Lineal ($R^2={r_squared:.3f}$)')
                
                # Etiquetas con nomenclatura griega
                ax.set_xlabel(r'Sustrato Consumido: $-\Delta S$ (g/L)', fontsize=12)
                ax.set_ylabel(r'Biomasa Producida: $\Delta X$ (g/L)', fontsize=12)
                ax.set_title(r'Determinación de $Y_{X/S}$', fontsize=14)
                ax.legend()
                ax.grid(True, linestyle=':', alpha=0.6)
                
                st.pyplot(fig)
                
            with st.expander("Ver Resumen Estadístico Completo (Statsmodels)"):
                st.text(results.summary())

        except Exception as e:
            st.error(f"Error en el cálculo: {e}. Verifique que los datos sean numéricos.")
else:
    st.warning("Por favor ingrese datos y seleccione las columnas.")
