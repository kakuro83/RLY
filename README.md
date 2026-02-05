# 🧫 Taller de Cinética Microbiana – Modelo de Monod en Streamlit

Esta aplicación en **Streamlit** resuelve un ejercicio práctico de cinética microbiana para determinar el **potencial fermentativo** de un consorcio microbiano usando **lactosa como sustrato**.

El flujo del programa replica paso a paso el planteamiento típico de un curso de **bioprocesos / ingeniería bioquímica**, integrando análisis de datos experimentales, ajuste cinético y predicción.

---

## 🎯 Objetivos del programa

A partir de datos experimentales de tiempo, biomasa y sustrato:

1. **Calcular el rendimiento biomasa/sustrato** (Y_{X/S}) mediante regresión lineal.
2. **Estimar las constantes cinéticas de Monod** (\mu_{max}) y (K_s) usando datos diferenciales por intervalos.
3. **Predecir la concentración de biomasa** en el tiempo resolviendo una ecuación diferencial en función de (X).
4. Calcular la **biomasa total producida** en un biorreactor de volumen definido.

---

## 📥 Datos de entrada

La app requiere una tabla con las siguientes columnas:

* **t**: tiempo (h)
* **X**: concentración de biomasa (g/L)
* **S**: concentración de sustrato (g/L)

Los datos pueden:

* editarse directamente en la interfaz,
* o cargarse usando el conjunto de datos de ejemplo incluido (correspondiente al taller).

---

## 🧠 Modelo y ecuaciones

### 1) Rendimiento biomasa/sustrato

Se asume una relación lineal entre biomasa producida y sustrato consumido:

[
Y_{X/S} = \frac{\Delta X}{-\Delta S}
]

El parámetro se obtiene por **regresión lineal** de (\Delta X) vs (-\Delta S).

---

### 2) Cinética de Monod (por intervalos)

Para cada intervalo experimental:

[
\mu_{obs} = \frac{1}{X_{prom}},\frac{\Delta X}{\Delta t}
]

con:

[
X_{prom} = \frac{X_i + X_f}{2}, \qquad S_{prom} = \frac{S_i + S_f}{2}
]

La ecuación de Monod se ajusta usando (S_{prom}):

[
\mu(S_{prom}) = \mu_{max},\frac{S_{prom}}{K_s + S_{prom}}
]

El ajuste se realiza por **regresión no lineal**.

---

### 3) Predicción de biomasa

La evolución de la biomasa se modela con una ecuación diferencial en función de (X), sustituyendo el sustrato mediante el balance con el rendimiento:

[
\frac{dX}{dt} = \mu_{max}X,\frac{Y_{X/S}S_0 + X_0 - X}{Y_{X/S}S_0 + Y_{X/S}K_s + X_0 - X}
]

con condición inicial:

[
X(0) = X_0
]

La ecuación se resuelve numéricamente para predecir (X(t)).

---

## 🧪 Salidas del programa

La aplicación entrega:

* Valor de **(Y_{X/S})** y estadísticos de la regresión.
* Estimación de **(\mu_{max})** y **(K_s)**.
* Tabla detallada por intervalos ((\Delta X), (\Delta t), (X_{prom}), (S_{prom}), (\mu_{obs})).
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
