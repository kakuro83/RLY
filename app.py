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

st.title("🧫 Taller: Rendimiento $Y_{X/S}$, cinética Monod ($\mu_{max}$, $K_s$) y predicción")

st.caption(
    "App para calcular rendimiento biomasa/sustrato, ajustar cinética de Monod y predecir X(t) en un biorreactor."
)

# Bloque de ecuaciones: mejor render (evita backslashes rotos)
st.markdown(
    r"""
### Modelo y ecuaciones

**1) Rendimiento (por regresión lineal):**

$$
Y_{X/S}=\frac{\Delta X}{-\Delta S}
$$

**2) Cinética (por intervalos + ajuste no lineal Monod):**

$$
\mu_{obs}=\frac{1}{X_{prom}}\,\frac{\Delta X}{\Delta t}, \qquad
\mu(S)=\mu_{max}\,\frac{S}{K_s+S}
$$

**3) Predicción de biomasa (ecuación en función de X):**

$$
\frac{dX}{dt}=\mu_{max}X\,\frac{Y_{X/S}S_0+X_0-X}{Y_{X/S}S_0+Y_{X/S}K_s+X_0-X}
$$
""",
    unsafe_allow_html=False,
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
    dS = np.diff(S)

    valid = dt > 0

    t0 = t[:-1][valid]
    t1 = t[1:][valid]
    dt = dt[valid]
    dX = dX[valid]
    dS = dS[valid]

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
    Ks_candidates = np.linspace(max(float(np.min(S_avg)) * 0.05, 1e-6), max(float(np.max(S_avg)) * 2.0, 1e-6), 200)

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
    index=guess_index(["S (g/L)", "Sustrato_S (g/L)
