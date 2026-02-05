import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Intentamos usar SciPy (recomendado). Si no está, caemos a un ajuste simple + Euler.
SCIPY_OK = True
try:
    from scipy.optimize import curve_fit
    from scipy.integrate import solve_ivp
except Exception:
    SCIPY_OK = False

# ----------------------------
# Configuración de la página
# ----------------------------
st.set_page_config(
    page_title="Taller Monod: Yx/s, μmax, Ks y Predicción",
    page_icon="🧫",
    layout="wide",
)

st.title("🧫 Taller: Rendimiento $Y_{X/S}$, cinética Monod ($\\mu_{max}$, $K_s$) y predicción")

st.caption(
    "App para calcular rendimiento biomasa/sustrato, ajustar cinética de Monod y predecir X(t) en un biorreactor."
)

# Bloque de ecuaciones: mejor render (evita backslashes rotos)
# Bloque de ecuaciones: usar st.latex (más estable que markdown para el taller)
st.markdown("### Modelo y ecuaciones")

st.markdown("**1) Rendimiento (por regresión lineal):**")
st.latex(r"Y_{X/S}=\frac{\Delta X}{-\Delta S}")

st.markdown("**2) Cinética (por intervalos + ajuste no lineal Monod):**")
st.latex(r"\mu_{obs}=\frac{1}{X_{prom}}\,\frac{\Delta X}{\Delta t}")
st.latex(r"\mu\!\left(S_{prom}\right)=\mu_{max}\,\frac{S_{prom}}{K_s + S_{prom}}")
st.latex(r"S_{prom}=\frac{S_i+S_f}{2}")

st.markdown("**3) Predicción de biomasa (ecuación en función de X):**")
st.latex(
    r"\frac{dX}{dt}=\mu_{max}X\,\frac{Y_{X/S}S_0+X_0-X}{Y_{X/S}S_0+Y_{X/S}K_s+X_0-X}"
)

# ----------------------------
# Sidebar: ingreso de datos
# ----------------------------
st.sidebar.header("⚙️ Configuración")
st.sidebar.markdown("Ajusta los datos y parámetros. Luego presiona **Calcular**.")

# Datos de ejemplo
use_example = st.sidebar.checkbox("Cargar datos de ejemplo (taller)", value=True)

if use_example:
    data_init = {
        "t (h)": [0, 8, 16, 24, 32, 40, 48],
        "X (g/L)": [0.5000, 0.8168, 1.3152, 2.0636, 3.0903, 4.2575, 5.1718],
        "S (g/L)": [10.0000, 9.3945, 8.4419, 7.0115, 5.0491, 2.8182, 1.0707],
    }
    df = pd.DataFrame(data_init)
else:
    df = pd.DataFrame(columns=["t (h)", "X (g/L)", "S (g/L)"])

# Editor de datos (arriba)
st.subheader("📥 Datos experimentales")
st.info("Edita los valores directamente o agrega filas. Deben existir columnas de **t**, **X** y **S**.")
df_input = st.data_editor(df, num_rows="dynamic", use_container_width=True)

# ----------------------------
# Helpers
# ----------------------------
def _to_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def clean_and_sort(df_raw: pd.DataFrame, col_t: str, col_x: str, col_s: str) -> pd.DataFrame:
    t = _to_numeric_series(df_raw[col_t])
    x = _to_numeric_series(df_raw[col_x])
    s = _to_numeric_series(df_raw[col_s])
    ok = t.notna() & x.notna() & s.notna()
    d = pd.DataFrame({"t": t[ok], "X": x[ok], "S": s[ok]}).copy()
    d = d.sort_values("t").reset_index(drop=True)
    # Filtrar duplicados de tiempo (nos quedamos con el primero)
    d = d.drop_duplicates(subset=["t"], keep="first").reset_index(drop=True)
    return d


def monod_mu(S, mu_max, Ks):
    S = np.asarray(S, dtype=float)
    return mu_max * S / (Ks + S)


def compute_interval_table(d: pd.DataFrame) -> pd.DataFrame:
    t = d["t"].to_numpy(dtype=float)
    X = d["X"].to_numpy(dtype=float)
    S = d["S"].to_numpy(dtype=float)

    dt = np.diff(t)
    dX = np.diff(X)

    valid = dt > 0

    t0 = t[:-1][valid]
    t1 = t[1:][valid]
    dt = dt[valid]
    dX = dX[valid]

    X0 = X[:-1][valid]
    X1 = X[1:][valid]
    S0 = S[:-1][valid]
    S1 = S[1:][valid]

    dX_dt = dX / dt
    X_avg = (X0 + X1) / 2.0
    S_avg = (S0 + S1) / 2.0

    eps = 1e-12
    mu_obs = dX_dt / np.maximum(X_avg, eps)

    out = pd.DataFrame(
        {
            "t_i (h)": t0,
            "t_f (h)": t1,
            "Δt (h)": dt,
            "X_i (g/L)": X0,
            "X_f (g/L)": X1,
            "ΔX (g/L)": dX,
            "ΔX/Δt (g/L·h)": dX_dt,
            "X_prom (g/L)": X_avg,
            "S_i (g/L)": S0,
            "S_f (g/L)": S1,
            "S_prom (g/L)": S_avg,
            "μ_obs (1/h)": mu_obs,
        }
    )

    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    out = out[out["S_prom (g/L)"] > 0]
    out = out[out["X_prom (g/L)"] > 0]
    out = out[out["μ_obs (1/h)"] >= 0]
    return out.reset_index(drop=True)


def fit_monod(S_avg, mu_obs):
    S_avg = np.asarray(S_avg, dtype=float)
    mu_obs = np.asarray(mu_obs, dtype=float)

    mu0 = max(float(np.max(mu_obs)), 1e-6)
    Ks0 = max(float(np.median(S_avg)), 1e-6)

    if SCIPY_OK:
        popt, pcov = curve_fit(
            monod_mu,
            S_avg,
            mu_obs,
            p0=[mu0, Ks0],
            bounds=([1e-12, 1e-12], [np.inf, np.inf]),
            maxfev=20000,
        )
        mu_max, Ks = popt
        perr = (
            np.sqrt(np.diag(pcov))
            if pcov is not None and pcov.size
            else np.array([np.nan, np.nan])
        )
        return float(mu_max), float(Ks), float(perr[0]), float(perr[1])

    # Fallback sin SciPy: búsqueda en rejilla
    mu_candidates = np.linspace(mu0 * 0.2, mu0 * 2.0, 200)
    Ks_candidates = np.linspace(
        max(float(np.min(S_avg)) * 0.05, 1e-6),
        max(float(np.max(S_avg)) * 2.0, 1e-6),
        200,
    )

    best = None
    best_sse = np.inf
    for mu_c in mu_candidates:
        for Ks_c in Ks_candidates:
            pred = monod_mu(S_avg, mu_c, Ks_c)
            sse = float(np.sum((mu_obs - pred) ** 2))
            if sse < best_sse:
                best_sse = sse
                best = (mu_c, Ks_c)

    mu_max, Ks = best
    return float(mu_max), float(Ks), float("nan"), float("nan")


def simulate_X(t_end_h, mu_max, Ks, Yxs, X0, S0, dt_h=0.02):
    """Resuelve:

    dX/dt = μmax X (Yxs*S0 + X0 - X) / (Yxs*S0 + Yxs*Ks + X0 - X)

    Retorna (t_grid, X_grid).
    """

    def rhs(t, X):
        X = float(X)
        num = (Yxs * S0 + X0 - X)
        den = (Yxs * S0 + Yxs * Ks + X0 - X)
        den = den if abs(den) > 1e-12 else 1e-12
        return mu_max * X * (num / den)

    if SCIPY_OK:
        sol = solve_ivp(
            fun=lambda t, y: [rhs(t, y[0])],
            t_span=(0.0, float(t_end_h)),
            y0=[float(X0)],
            method="RK45",
            dense_output=True,
            max_step=max(float(dt_h), 1e-3),
        )
        t_grid = np.linspace(0, float(t_end_h), 300)
        X_grid = sol.sol(t_grid)[0] if sol.sol is not None else np.interp(t_grid, sol.t, sol.y[0])
        return t_grid, X_grid

    # Euler explícito
    n = int(np.ceil(float(t_end_h) / float(dt_h))) + 1
    t_grid = np.linspace(0, float(t_end_h), n)
    X_grid = np.zeros(n, dtype=float)
    X_grid[0] = float(X0)
    for i in range(1, n):
        dXdt = rhs(t_grid[i - 1], X_grid[i - 1])
        X_grid[i] = max(X_grid[i - 1] + dXdt * float(dt_h), 0.0)
    return t_grid, X_grid


# ----------------------------
# Selección de columnas
# ----------------------------
if df_input is None or df_input.empty or df_input.shape[1] < 3:
    st.warning("Ingresa datos (al menos 3 columnas: tiempo, X y S).")
    st.stop()

cols = df_input.columns.tolist()


def guess_index(names, default=0):
    for n in names:
        if n in cols:
            return cols.index(n)
    return default


st.sidebar.subheader("🧾 Columnas")
col_t = st.sidebar.selectbox(
    "Columna de Tiempo (t)",
    cols,
    index=guess_index(["t (h)", "Tiempo (h)", "Tiempo", "t"], 0),
)
col_X = st.sidebar.selectbox(
    "Columna de Biomasa (X)",
    cols,
    index=guess_index(["X (g/L)", "Biomasa_X (g/L)", "Biomasa_X", "X"], 1 if len(cols) > 1 else 0),
)
col_S = st.sidebar.selectbox(
    "Columna de Sustrato (S)",
    cols,
    index=guess_index(["S (g/L)", "Sustrato_S (g/L)", "Sustrato_S", "S"], 2 if len(cols) > 2 else 0),
)

# Predicción
st.sidebar.subheader("🧪 Predicción")
S0_pred = st.sidebar.number_input("S0 (g/L)", value=10.0, min_value=0.0, step=0.1, format="%.4f")
X0_pred = st.sidebar.number_input("X0 (g/L)", value=0.5, min_value=0.0, step=0.1, format="%.4f")
t_pred = st.sidebar.number_input("Tiempo de predicción (h)", value=60.0, min_value=0.0, step=1.0, format="%.2f")
V_pred = st.sidebar.number_input("Volumen (L)", value=100.0, min_value=0.0, step=1.0, format="%.2f")

run_all = st.sidebar.button("🚀 Calcular", use_container_width=True)

# Nota SciPy
if not SCIPY_OK:
    st.sidebar.warning("SciPy no está disponible. Recomendado para ajuste/ODE robustos.")
    st.sidebar.code("pip install scipy", language="bash")


# ----------------------------
# Ejecutar
# ----------------------------
if not run_all:
    st.info("Ajusta los datos/parámetros y presiona **Calcular**.")
    st.stop()

try:
    d = clean_and_sort(df_input, col_t, col_X, col_S)
    if len(d) < 3:
        st.error("Necesitas al menos 3 puntos válidos (t, X, S) para estimar cinética.")
        st.stop()

    tabs = st.tabs(["1) Rendimiento", "2) Monod", "3) Predicción", "📦 Exportar"])

    # ----------------------------
    # TAB 1: Rendimiento
    # ----------------------------
    with tabs[0]:
        st.subheader("1) Rendimiento $Y_{X/S}$")

        X = d["X"].to_numpy(dtype=float)
        S = d["S"].to_numpy(dtype=float)

        delta_X = X - X[0]
        minus_delta_S = S[0] - S

        x_model = sm.add_constant(minus_delta_S)
        model = sm.OLS(delta_X, x_model)
        results = model.fit()

        Yxs = float(results.params[1])
        intercept = float(results.params[0])
        r2 = float(results.rsquared)

        c1, c2 = st.columns([1, 2], gap="large")
        with c1:
            st.metric("$Y_{X/S}$ (gX/gS)", f"{Yxs:.6f}")
            st.metric("Intercepto", f"{intercept:.6f}")
            st.metric("$R^2$", f"{r2:.4f}")

            st.markdown(r"**Ecuación ajustada:**")
            st.latex(
                r"\Delta X = "
                + f"{Yxs:.6f}"
                + r"\,(-\Delta S)"
                + (r" + " if intercept >= 0 else r" - ")
                + f"{abs(intercept):.6f}"
            )

        with c2:
            fig, ax = plt.subplots()
            ax.scatter(minus_delta_S, delta_X, s=80, alpha=0.85, edgecolors="k")
            x_line = np.linspace(float(np.min(minus_delta_S)), float(np.max(minus_delta_S)), 200)
            y_line = intercept + Yxs * x_line
            ax.plot(x_line, y_line, linestyle="--")
            ax.set_xlabel(r"Sustrato consumido: $-\Delta S$ (g/L)")
            ax.set_ylabel(r"Biomasa producida: $\Delta X$ (g/L)")
            ax.set_title(r"Determinación de $Y_{X/S}$")
            ax.grid(True, linestyle=":", alpha=0.6)
            st.pyplot(fig)

        with st.expander("Ver resumen estadístico (statsmodels)"):
            st.text(results.summary())

    # ----------------------------
    # TAB 2: Monod
    # ----------------------------
    with tabs[1]:
        st.subheader("2) Constantes cinéticas Monod ($\mu_{max}$ y $K_s$)")

        intervals = compute_interval_table(d)
        if intervals.empty or len(intervals) < 2:
            st.error("No hay intervalos válidos suficientes para ajustar Monod.")
            st.stop()

        S_avg = intervals["S_prom (g/L)"].to_numpy(dtype=float)
        mu_obs = intervals["μ_obs (1/h)"].to_numpy(dtype=float)

        mask = np.isfinite(S_avg) & np.isfinite(mu_obs) & (S_avg > 0) & (mu_obs >= 0)
        S_avg_fit = S_avg[mask]
        mu_obs_fit = mu_obs[mask]

        mu_max, Ks, mu_max_se, Ks_se = fit_monod(S_avg_fit, mu_obs_fit)

        c3, c4 = st.columns([1, 2], gap="large")
        with c3:
            st.metric("$\mu_{max}$ (1/h)", f"{mu_max:.6f}")
            st.metric("$K_s$ (g/L)", f"{Ks:.6f}")

            st.markdown(
                r"""
**Cómo se calcula $\mu_{obs}$ por intervalo:**

$$
\mu_{obs}=\frac{1}{X_{prom}}\frac{\Delta X}{\Delta t}
$$

Luego se ajusta Monod:

$$
\mu(S)=\mu_{max}\frac{S}{K_s+S}
$$
"""
            )

            if SCIPY_OK:
                st.caption(f"Errores estándar (aprox): SE(μmax)={mu_max_se:.6f}, SE(Ks)={Ks_se:.6f}")
            else:
                st.warning("Sin SciPy: ajuste por rejilla (sin errores estándar).")

        with c4:
            fig2, ax2 = plt.subplots()
            ax2.scatter(S_avg_fit, mu_obs_fit, s=80, alpha=0.85, edgecolors="k", label=r"$\mu_{obs}$")
            S_line = np.linspace(0, float(np.max(S_avg_fit)) * 1.1, 300)
            ax2.plot(S_line, monod_mu(S_line, mu_max, Ks), label="Monod")
            ax2.set_xlabel(r"$S_{prom}$ (g/L)")
            ax2.set_ylabel(r"$\mu$ (1/h)")
            ax2.set_title("Ajuste no lineal: Monod")
            ax2.grid(True, linestyle=":", alpha=0.6)
            ax2.legend()
            st.pyplot(fig2)

        st.write("### Tabla por intervalos")
        st.dataframe(intervals, use_container_width=True)

    # ----------------------------
    # TAB 3: Predicción
    # ----------------------------
    with tabs[2]:
        st.subheader("3) Predicción de biomasa")

        t_grid, X_grid = simulate_X(
            t_end_h=float(t_pred),
            mu_max=float(mu_max),
            Ks=float(Ks),
            Yxs=float(Yxs),
            X0=float(X0_pred),
            S0=float(S0_pred),
            dt_h=0.02,
        )

        X_final = float(X_grid[-1])
        biomasa_total_final = X_final * float(V_pred)
        biomasa_producida = (X_final - float(X0_pred)) * float(V_pred)

        c5, c6, c7 = st.columns(3, gap="large")
        c5.metric("$X(t_{final})$ (g/L)", f"{X_final:.6f}")
        c6.metric("Biomasa total (g)", f"{biomasa_total_final:.3f}")
        c7.metric("Biomasa neta producida (g)", f"{biomasa_producida:.3f}")

        fig3, ax3 = plt.subplots()
        ax3.plot(t_grid, X_grid)
        ax3.set_xlabel("t (h)")
        ax3.set_ylabel("X (g/L)")
        ax3.set_title("Predicción X(t)")
        ax3.grid(True, linestyle=":", alpha=0.6)
        st.pyplot(fig3)

        # S(t) estimado por balance (sanity check)
        if Yxs > 0:
            S_est = float(S0_pred) - (X_grid - float(X0_pred)) / float(Yxs)
            fig4, ax4 = plt.subplots()
            ax4.plot(t_grid, S_est)
            ax4.set_xlabel("t (h)")
            ax4.set_ylabel("S estimado (g/L)")
            ax4.set_title(r"$S(t)=S_0-\frac{X-X_0}{Y_{X/S}}$")
            ax4.grid(True, linestyle=":", alpha=0.6)
            st.pyplot(fig4)

        with st.expander("Ver ecuación usada"):
            st.markdown(
                r"""
Se resolvió:

$$
\frac{dX}{dt}=\mu_{max}X\,\frac{Y_{X/S}S_0+X_0-X}{Y_{X/S}S_0+Y_{X/S}K_s+X_0-X}
$$

con condiciones iniciales $X(0)=X_0$ y parámetros estimados.
"""
            )

    # ----------------------------
    # TAB 4: Exportar
    # ----------------------------
    with tabs[3]:
        st.subheader("📦 Exportar resultados")

        df_clean = d.copy()
        df_intervals = intervals.copy()

        summary = pd.DataFrame(
            {
                "Parametro": [
                    "Yxs (gX/gS)",
                    "Intercepto",
                    "R2",
                    "mu_max (1/h)",
                    "Ks (g/L)",
                    "X_final (g/L)",
                    "Biomasa_total (g)",
                    "Biomasa_neta (g)",
                ],
                "Valor": [Yxs, intercept, r2, mu_max, Ks, X_final, biomasa_total_final, biomasa_producida],
            }
        )

        c8, c9 = st.columns(2, gap="large")
        with c8:
            st.write("**Resumen**")
            st.dataframe(summary, use_container_width=True)

        with c9:
            st.write("**Descargas**")
            st.download_button(
                "⬇️ Descargar datos limpios (CSV)",
                data=df_clean.to_csv(index=False).encode("utf-8"),
                file_name="datos_limpios.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.download_button(
                "⬇️ Descargar intervalos (CSV)",
                data=df_intervals.to_csv(index=False).encode("utf-8"),
                file_name="tabla_intervalos.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.download_button(
                "⬇️ Descargar resumen (CSV)",
                data=summary.to_csv(index=False).encode("utf-8"),
                file_name="resumen_parametros.csv",
                mime="text/csv",
                use_container_width=True,
            )

except Exception as e:
    st.error(f"Error en el cálculo: {e}")
    st.info("Tip: revisa que (t, X, S) sean numéricos, que t esté ordenado y sin duplicados.")
