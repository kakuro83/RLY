# 🧫 Taller de Cinética Microbiana – Modelo de Monod en Streamlit

> ⚠️ **Nota sobre LaTeX en README**: GitHub **no renderiza LaTeX/MathJax** en archivos `README.md`. Por eso, las ecuaciones se presentan abajo en **formato texto** (compatible). En la app de Streamlit **sí** se muestran con LaTeX.

---

## 🎯 Objetivos del programa

A partir de datos experimentales de tiempo, biomasa y sustrato:

1. Calcular el **rendimiento biomasa/sustrato** (Yx/s) mediante regresión lineal.
2. Estimar las **constantes cinéticas de Monod** (μmax y Ks) usando datos diferenciales por intervalos.
3. **Predecir la concentración de biomasa** en el tiempo resolviendo una ecuación diferencial en función de X.
4. Calcular la **biomasa total producida** en un biorreactor de volumen definido.

---

## 📥 Datos de entrada

La app requiere una tabla con las siguientes columnas:

* **t**: tiempo (h)
* **X**: concentración de biomasa (g/L)
* **S**: concentración de sustrato (g/L)

Los datos pueden editarse directamente en la interfaz o cargarse usando el conjunto de datos de ejemplo.

---

## 🧠 Modelo y ecuaciones (formato compatible con GitHub)

### 1) Rendimiento biomasa/sustrato

Yx/s = ΔX / (−ΔS)

El parámetro se obtiene por **regresión lineal** de ΔX vs (−ΔS).

---

### 2) Cinética de Monod (por intervalos)

Para cada intervalo experimental:

μ_obs = (1 / X_prom) · (ΔX / Δt)

con:

X_prom = (X_i + X_f) / 2

S_prom = (S_i + S_f) / 2

La ecuación de Monod se ajusta usando S_prom:

μ(S_prom) = μ_max · S_prom / (K_s + S_prom)

El ajuste se realiza por **regresión no lineal**.

---

### 3) Predicción de biomasa

La evolución de la biomasa se modela con una ecuación diferencial en función de X, sustituyendo el sustrato mediante el balance con el rendimiento:

dX/dt = μ_max · X · (Yx/s · S0 + X0 − X) / (Yx/s · S0 + Yx/s · K_s + X0 − X)

con condición inicial:

X(0) = X0

La ecuación se resuelve numéricamente para predecir X(t).

---

## 🧪 Salidas del programa

* Valor de **Yx/s** y estadísticos de la regresión.
* Estimación de **μ_max** y **K_s**.
* Tabla detallada por intervalos (ΔX, Δt, X_prom, S_prom, μ_obs).
* Gráficas de ajuste y predicción.
* Biomasa final y biomasa total producida en el biorreactor.
* Exportación de resultados en formato **CSV**.

---

## ▶️ Ejecución

Instalar dependencias:

```bash
pip install streamlit pandas numpy matplotlib statsmodels scipy
```

Ejecutar la app:

```bash
streamlit run app.py
```

---

## 📌 Notas finales

* El modelo asume cultivo batch sin inhibición.
* El uso de promedios por intervalo mejora la coherencia entre datos experimentales y cinética.
* La estructura del código está pensada con fines **didácticos**, priorizando claridad y trazabilidad del modelo.

---

📘 *Desarrollado como apoyo para talleres de cinética microbiana y diseño de biorreactores.*
