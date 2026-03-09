import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Optimizador de Cartera",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
  .main { background-color: #0a0e1a; }
  .block-container { padding-top: 1.5rem; }
  h1, h2, h3 { color: #00d4ff; }
  .stTabs [data-baseweb="tab-list"] { gap: 6px; }
  .stTabs [data-baseweb="tab"] {
    background-color: #111827;
    border: 1px solid #1e2d45;
    border-radius: 8px;
    color: #64748b;
    padding: 6px 16px;
  }
  .stTabs [aria-selected="true"] {
    background-color: #00d4ff !important;
    color: #0a0e1a !important;
    font-weight: 700;
  }
  div[data-testid="metric-container"] {
    background: #111827;
    border: 1px solid #1e2d45;
    border-radius: 10px;
    padding: 12px 16px;
  }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = pd.DataFrame(columns=['Ticker','Monto_USD','Target_%','Sector'])
if 'sectors' not in st.session_state:
    st.session_state.sectors = [
        'Tecnología','Comunicaciones','Salud','Financiero',
        'Energía','Consumo','Industriales','Brasil/EM',
        'Minería/Cobre','Defensa','Otro'
    ]
if 'hist_data' not in st.session_state:
    st.session_state.hist_data = None
if 'tickers_loaded' not in st.session_state:
    st.session_state.tickers_loaded = []

# ─────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────
COLORS = px.colors.qualitative.Set2

def fmt_usd(v):
    return f"${v:,.0f}"

def fmt_pct(v):
    return f"{v:.2f}%"

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_data(tickers, period="5y"):
    """Fetch historical closing prices from Yahoo Finance."""
    try:
        raw = yf.download(tickers, period=period, auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw['Close']
        else:
            prices = raw[['Close']] if 'Close' in raw.columns else raw
        prices = prices.dropna(how='all')
        return prices
    except Exception as e:
        st.error(f"Error al descargar datos: {e}")
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def validate_tickers(tickers):
    """Check which tickers exist on Yahoo Finance."""
    valid, invalid = [], []
    for t in tickers:
        try:
            info = yf.Ticker(t).fast_info
            if hasattr(info, 'last_price') and info.last_price and info.last_price > 0:
                valid.append(t)
            else:
                invalid.append(t)
        except:
            invalid.append(t)
    return valid, invalid

def calc_portfolio_weights(df):
    total = df['Monto_USD'].sum()
    df = df.copy()
    df['Peso_Actual_%'] = df['Monto_USD'] / total * 100 if total > 0 else 0
    df['Desvio_%'] = df['Peso_Actual_%'] - df['Target_%']
    df['Target_USD'] = df['Target_%'] / 100 * total
    df['Rebalanceo_USD'] = df['Target_USD'] - df['Monto_USD']
    df['Total'] = total
    return df

def calc_metrics(prices, weights=None, rf=0.02):
    """Calculate return, volatility, Sharpe, CAGR, Beta vs SPY."""
    prices = prices.dropna(how='all')
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()
    if len(prices) < 30:
        raise ValueError("Datos insuficientes para calcular métricas (mínimo 30 días).")
    returns = prices.pct_change().dropna()
    if len(returns) == 0:
        raise ValueError("No hay retornos calculables con los datos disponibles.")
    annual_ret = returns.mean() * 252
    annual_vol = returns.std() * np.sqrt(252)
    sharpe = (annual_ret - rf) / annual_vol

    # CAGR — safe
    first = prices.iloc[0].replace(0, np.nan)
    last  = prices.iloc[-1]
    total_ret = last / first
    years = max(len(prices) / 252, 0.01)
    cagr = total_ret ** (1/years) - 1

    metrics = pd.DataFrame({
        'Retorno Anual %': (annual_ret * 100).round(2),
        'Volatilidad %': (annual_vol * 100).round(2),
        'Sharpe': sharpe.round(3),
        'CAGR %': (cagr * 100).round(2),
    })

    # Beta vs SPY
    try:
        spy_raw = yf.download('SPY', period=f'{max(int(years),1)}y', auto_adjust=True, progress=False)
        if isinstance(spy_raw.columns, pd.MultiIndex):
            spy = spy_raw['Close']['SPY'] if 'SPY' in spy_raw['Close'].columns else spy_raw['Close'].iloc[:,0]
        else:
            spy = spy_raw['Close'] if 'Close' in spy_raw.columns else spy_raw.iloc[:,0]
        spy = spy.squeeze()
        spy_ret = spy.pct_change().dropna()
        betas = {}
        for col in returns.columns:
            s1 = returns[col].dropna()
            s2 = spy_ret.dropna()
            common = s1.index.intersection(s2.index)
            if len(common) > 30:
                cov_matrix = np.cov(s1.loc[common].values, s2.loc[common].values)
                betas[col] = round(cov_matrix[0,1] / cov_matrix[1,1], 2) if cov_matrix[1,1] != 0 else np.nan
            else:
                betas[col] = np.nan
        metrics['Beta SPY'] = pd.Series(betas)
    except Exception as e:
        metrics['Beta SPY'] = np.nan

    # VaR 95%
    var_1d = (-returns.quantile(0.05) * 100).round(2)
    var_10d = (var_1d * np.sqrt(10)).round(2)
    metrics['VaR 1d 95%'] = var_1d
    metrics['VaR 10d 95%'] = var_10d

    return metrics, returns

def portfolio_metrics(weights_arr, returns_df, rf=0.02):
    port_ret = np.dot(weights_arr, returns_df.mean()) * 252
    port_vol = np.sqrt(np.dot(weights_arr, np.dot(returns_df.cov() * 252, weights_arr)))
    sharpe = (port_ret - rf) / port_vol
    return port_ret, port_vol, sharpe

def max_sharpe(returns_df, rf=0.02, min_weight=0.0):
    n = len(returns_df.columns)
    w0 = np.ones(n) / n
    lb = max(min_weight, 0.0)
    bounds = [(lb, 1.0)] * n
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    result = minimize(
        lambda w: -portfolio_metrics(w, returns_df, rf)[2],
        w0, method='SLSQP', bounds=bounds, constraints=constraints
    )
    return result.x if result.success else w0

def min_variance(returns_df, min_weight=0.0):
    n = len(returns_df.columns)
    w0 = np.ones(n) / n
    lb = max(min_weight, 0.0)
    bounds = [(lb, 1.0)] * n
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    result = minimize(
        lambda w: portfolio_metrics(w, returns_df)[1],
        w0, method='SLSQP', bounds=bounds, constraints=constraints
    )
    return result.x if result.success else w0

def target_return_weights(returns_df, target_ret, min_weight=0.0):
    """Minimize volatility subject to hitting a target annual return."""
    n = len(returns_df.columns)
    w0 = np.ones(n) / n
    lb = max(min_weight, 0.0)
    bounds = [(lb, 1.0)] * n
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        {'type': 'eq', 'fun': lambda w: portfolio_metrics(w, returns_df)[0] - target_ret},
    ]
    result = minimize(
        lambda w: portfolio_metrics(w, returns_df)[1],
        w0, method='SLSQP', bounds=bounds, constraints=constraints
    )
    return result.x if result.success else None

def black_litterman(weights, returns_df, views_q, views_conf, rf=0.02, tau=0.05):
    """Simplified Black-Litterman model."""
    mu_eq = returns_df.mean() * 252
    sigma = returns_df.cov() * 252
    pi = mu_eq  # market equilibrium proxy

    n = len(weights)
    posterior_mu = pi.copy()
    for i, ticker in enumerate(returns_df.columns):
        if ticker in views_q:
            q = views_q[ticker]
            conf = views_conf.get(ticker, 0.5)
            posterior_mu[ticker] = (1 - conf) * pi[ticker] + conf * q

    return posterior_mu

def monte_carlo(port_ret, port_vol, total_usd, n_sims=500, n_days=252):
    dt = 1/252
    paths = np.zeros((n_sims, n_days+1))
    paths[:, 0] = total_usd
    z = np.random.standard_normal((n_sims, n_days))
    for d in range(1, n_days+1):
        paths[:, d] = paths[:, d-1] * np.exp(
            (port_ret - 0.5*port_vol**2)*dt + port_vol*np.sqrt(dt)*z[:, d-1]
        )
    return paths

# ─────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuración")

    # Años de historia — slider igual al otro optimizador
    years = st.slider("Años de historia", min_value=1, max_value=20, value=5,
        help="Cantidad de años de datos históricos a analizar")
    period_map = {1:"1y",2:"2y",3:"3y",4:"4y",5:"5y",6:"6y",7:"7y",8:"8y",
                  9:"9y",10:"10y",15:"15y",20:"20y"}
    PERIOD = period_map.get(years, f"{years}y")

    rf_rate = st.number_input("Tasa libre de riesgo (%)", value=2.0, step=0.25,
        help="Tasa anual libre de riesgo (ej: 2%)") / 100

    st.markdown("---")
    st.markdown("### 📐 Opciones avanzadas")

    # Retorno objetivo
    use_target_return = st.checkbox("Usar retorno objetivo",
        help="Optimizar la cartera buscando un retorno anual específico")
    target_return_pct = None
    if use_target_return:
        target_return_pct = st.number_input(
            "Retorno objetivo (%)", min_value=-50.0, max_value=100.0,
            value=10.0, step=0.5,
            help="Retorno anual objetivo para la optimización BL y Monte Carlo"
        )
        st.session_state['target_return'] = target_return_pct / 100
    else:
        st.session_state['target_return'] = None

    # Peso mínimo por activo
    min_weight_pct = st.number_input(
        "Peso mínimo por activo (%)", min_value=0.0, max_value=50.0,
        value=0.0, step=1.0,
        help="Peso mínimo que debe tener cada activo en la cartera optimizada (0 = sin mínimo)"
    )
    st.session_state['min_weight'] = min_weight_pct / 100

    st.markdown("---")
    st.markdown("### 🏷️ Gestión de sectores")
    new_sector = st.text_input("Nuevo sector", placeholder="Ej: REITs")
    if st.button("➕ Agregar sector") and new_sector.strip():
        if new_sector.strip() not in st.session_state.sectors:
            st.session_state.sectors.append(new_sector.strip())
            st.success(f"Sector '{new_sector}' agregado")
        else:
            st.warning("Ya existe ese sector")

    sector_to_del = st.selectbox("Eliminar sector", ["—"] + st.session_state.sectors)
    if st.button("🗑️ Eliminar sector") and sector_to_del != "—":
        st.session_state.sectors.remove(sector_to_del)
        st.success(f"Sector '{sector_to_del}' eliminado")

    st.markdown("---")
    if st.button("🚀 Cargar datos de mercado", type="primary", use_container_width=True):
        tickers = st.session_state.portfolio['Ticker'].tolist()
        if tickers:
            with st.spinner("Descargando datos de Yahoo Finance..."):
                st.session_state.hist_data = fetch_data(tickers, PERIOD)
                st.session_state.tickers_loaded = tickers
            st.success("✅ Datos cargados")
        else:
            st.warning("Primero cargá tu cartera")

    st.markdown("---")
    st.caption("📊 Optimizador de Cartera v4.0\nDatos: Yahoo Finance")

# ─────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────
st.markdown("# 📊 Optimizador de Cartera")
st.markdown("*Rebalanceo · Sectores · Correlación · Métricas · Black-Litterman · Monte Carlo — datos reales de Yahoo Finance*")
st.markdown("---")

# ─────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────
tabs = st.tabs([
    "📁 Mi Cartera",
    "🏷️ Sectores",
    "⚖️ Rebalanceo",
    "🔗 Correlación",
    "📐 Métricas",
    "🔥 Stress Test",
    "🧠 Black-Litterman",
    "🎲 Monte Carlo",
])

# ══════════════════════════════════════════
#  TAB 1 — MI CARTERA
# ══════════════════════════════════════════
with tabs[0]:
    st.subheader("📁 Mi Cartera")

    col1, col2 = st.columns([1,1])
    with col1:
        st.markdown("#### Cargar desde Excel")
        uploaded = st.file_uploader(
            "Subí tu Excel (.xlsx) con columnas: Ticker, Monto_USD, Target_%, Sector (opcional)",
            type=['xlsx','xls','csv']
        )
        if uploaded:
            try:
                if uploaded.name.endswith('.csv'):
                    df_up = pd.read_csv(uploaded)
                else:
                    df_up = pd.read_excel(uploaded)
                df_up.columns = [c.strip() for c in df_up.columns]
                # Normalize column names
                rename_map = {}
                for c in df_up.columns:
                    cl = c.lower().replace(' ','_')
                    if 'ticker' in cl: rename_map[c] = 'Ticker'
                    elif 'monto' in cl or 'usd' in cl: rename_map[c] = 'Monto_USD'
                    elif 'target' in cl: rename_map[c] = 'Target_%'
                    elif 'sector' in cl: rename_map[c] = 'Sector'
                df_up = df_up.rename(columns=rename_map)
                if 'Sector' not in df_up.columns: df_up['Sector'] = ''
                df_up['Ticker'] = df_up['Ticker'].astype(str).str.upper().str.strip()
                df_up = df_up[['Ticker','Monto_USD','Target_%','Sector']].dropna(subset=['Ticker','Monto_USD'])
                # Check duplicates
                dupes = df_up[df_up.duplicated('Ticker', keep=False)]['Ticker'].unique()
                if len(dupes) > 0:
                    st.warning(f"⚠️ Tickers duplicados encontrados: {', '.join(dupes)} — se unificarán sumando montos")
                    df_up = df_up.groupby('Ticker', as_index=False).agg({
                        'Monto_USD':'sum', 'Target_%':'sum', 'Sector':'first'
                    })
                st.session_state.portfolio = df_up
                st.success(f"✅ {len(df_up)} activos cargados")
            except Exception as e:
                st.error(f"Error al leer el archivo: {e}")

    with col2:
        st.markdown("#### Agregar activo manualmente")
        with st.form("add_asset"):
            c1, c2, c3, c4 = st.columns([1.2,1.2,1,1.5])
            new_ticker = c1.text_input("Ticker", placeholder="AAPL").upper().strip()
            new_monto = c2.number_input("Monto USD", min_value=0.0, step=100.0)
            new_target = c3.number_input("Target %", min_value=0.0, max_value=100.0, step=0.5)
            new_sector = c4.selectbox("Sector", [""] + st.session_state.sectors)
            submitted = st.form_submit_button("➕ Agregar", type="primary")

        # Handle outside form to allow st.rerun()
        if submitted and new_ticker:
            import re
            if not re.match(r'^[A-Z0-9.\-]{1,10}$', new_ticker):
                st.error("⚠️ Formato de ticker inválido")
            else:
                # Validate ticker exists on Yahoo Finance
                with st.spinner(f"Verificando {new_ticker}..."):
                    try:
                        info = yf.Ticker(new_ticker).fast_info
                        ticker_ok = hasattr(info, 'last_price') and info.last_price and info.last_price > 0
                    except:
                        ticker_ok = False
                if not ticker_ok:
                    st.error(f"❌ **{new_ticker}** no se encontró en Yahoo Finance — verificá que el ticker sea correcto (ej: TSLA, no TESLA)")
                elif new_ticker in st.session_state.portfolio['Ticker'].values:
                    idx = st.session_state.portfolio[st.session_state.portfolio['Ticker']==new_ticker].index[0]
                    st.session_state.portfolio.loc[idx,'Monto_USD'] += new_monto
                    st.session_state.portfolio.loc[idx,'Target_%'] += new_target
                    st.warning(f"⚠️ {new_ticker} ya existía — se sumó el monto y target")
                    st.rerun()
                else:
                    new_row = pd.DataFrame([{'Ticker':new_ticker,'Monto_USD':new_monto,'Target_%':new_target,'Sector':new_sector}])
                    st.session_state.portfolio = pd.concat([st.session_state.portfolio, new_row], ignore_index=True)
                    st.success(f"✅ {new_ticker} agregado correctamente")
                    st.rerun()

    st.markdown("---")

    if st.button("📋 Cargar ejemplo"):
        st.session_state.portfolio = pd.DataFrame([
            {'Ticker':'GOOGL','Monto_USD':25000,'Target_%':23,'Sector':'Comunicaciones'},
            {'Ticker':'ASML', 'Monto_USD':15000,'Target_%':14,'Sector':'Tecnología'},
            {'Ticker':'FXI',  'Monto_USD':12000,'Target_%':10,'Sector':'Brasil/EM'},
            {'Ticker':'EWZ',  'Monto_USD':10000,'Target_%':10,'Sector':'Brasil/EM'},
            {'Ticker':'XLV',  'Monto_USD':8000, 'Target_%':6, 'Sector':'Salud'},
            {'Ticker':'META', 'Monto_USD':6000, 'Target_%':8, 'Sector':'Comunicaciones'},
            {'Ticker':'MSFT', 'Monto_USD':4000, 'Target_%':5, 'Sector':'Tecnología'},
            {'Ticker':'IBM',  'Monto_USD':4000, 'Target_%':5.4,'Sector':'Tecnología'},
            {'Ticker':'PANW', 'Monto_USD':4000, 'Target_%':5, 'Sector':'Tecnología'},
            {'Ticker':'JPM',  'Monto_USD':3000, 'Target_%':2.8,'Sector':'Financiero'},
            {'Ticker':'V',    'Monto_USD':3000, 'Target_%':4, 'Sector':'Financiero'},
            {'Ticker':'RTX',  'Monto_USD':3000, 'Target_%':3, 'Sector':'Defensa'},
            {'Ticker':'CEG',  'Monto_USD':3000, 'Target_%':3.8,'Sector':'Energía'},
        ])
        st.rerun()

    if not st.session_state.portfolio.empty:
        df = calc_portfolio_weights(st.session_state.portfolio.copy())
        total = df['Total'].iloc[0]
        total_target = df['Target_%'].sum()

        # Metrics row
        m1,m2,m3,m4,m5 = st.columns(5)
        m1.metric("Total cartera", fmt_usd(total))
        m2.metric("Activos", len(df))
        m3.metric("Total Target %", f"{total_target:.2f}%",
                  delta=f"{total_target-100:.2f}% vs 100%" if abs(total_target-100)>0.1 else "✓ OK")
        m4.metric("A comprar", len(df[df['Desvio_%']<-0.5]))
        m5.metric("A vender", len(df[df['Desvio_%']>0.5]))

        if abs(total_target - 100) > 0.1:
            st.warning(f"⚠️ Los targets suman {total_target:.2f}% — deben sumar 100%")

        # Editable table
        st.markdown("#### Editá tu cartera")

        # ✅ Fix 2: selector de orden
        sort_col_map = {
            'Ticker (A→Z)': ('Ticker', True),
            'Monto USD (mayor→menor)': ('Monto_USD', False),
            'Peso Actual % (mayor→menor)': ('Peso_Actual_%', False),
            'Desvío % (mayor→menor)': ('Desvio_%', False),
            'Sector (A→Z)': ('Sector', True),
        }
        sort_choice = st.selectbox("Ordenar por", list(sort_col_map.keys()), index=1, key="sort_portfolio")
        sort_field, sort_asc = sort_col_map[sort_choice]
        df_sorted = df[['Ticker','Monto_USD','Target_%','Sector','Peso_Actual_%','Desvio_%']].round(2).sort_values(sort_field, ascending=sort_asc)

        edited = st.data_editor(
            df_sorted,
            column_config={
                'Ticker': st.column_config.TextColumn('Ticker', width='small'),
                'Monto_USD': st.column_config.NumberColumn('Monto USD', format="$%.0f"),
                'Target_%': st.column_config.NumberColumn('Target %', format="%.1f%%"),
                'Sector': st.column_config.SelectboxColumn('Sector', options=[""] + st.session_state.sectors),
                'Peso_Actual_%': st.column_config.NumberColumn('Peso Actual %', format="%.2f%%", disabled=True),
                'Desvio_%': st.column_config.NumberColumn('Desvío %', format="%.2f%%", disabled=True),
            },
            hide_index=True,
            num_rows="dynamic",
            use_container_width=True,
            key="portfolio_editor"
        )
        if st.button("💾 Guardar cambios"):
            st.session_state.portfolio = edited[['Ticker','Monto_USD','Target_%','Sector']].copy()
            st.session_state.portfolio['Ticker'] = st.session_state.portfolio['Ticker'].str.upper().str.strip()
            st.rerun()

        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Exportar CSV", csv, "mi_cartera.csv", "text/csv")
        with col_exp2:
            import io
            buf = io.BytesIO()
            df.to_excel(buf, index=False)
            st.download_button("⬇️ Exportar Excel", buf.getvalue(), "mi_cartera.xlsx")
    else:
        st.info("Cargá tu cartera usando el formulario o subiendo un Excel")


# ══════════════════════════════════════════
#  TAB 2 — SECTORES
# ══════════════════════════════════════════
with tabs[1]:
    st.subheader("🏷️ Análisis por Sector")
    if st.session_state.portfolio.empty:
        st.info("Cargá tu cartera primero")
    else:
        df = calc_portfolio_weights(st.session_state.portfolio.copy())
        total = df['Total'].iloc[0]

        # Sector target editor
        st.markdown("#### Target % por sector (opcional)")
        if 'sector_targets' not in st.session_state:
            st.session_state.sector_targets = {s:0.0 for s in st.session_state.sectors}

        sector_cols = st.columns(min(4, len(st.session_state.sectors)))
        for i, s in enumerate(st.session_state.sectors):
            with sector_cols[i % len(sector_cols)]:
                val = st.number_input(s, min_value=0.0, max_value=100.0, step=1.0,
                    value=float(st.session_state.sector_targets.get(s,0)),
                    key=f"st_{s}")
                st.session_state.sector_targets[s] = val

        st.markdown("---")

        # Group by sector
        sector_group = df.groupby('Sector').agg(
            Monto_USD=('Monto_USD','sum'),
            Peso_Actual=('Peso_Actual_%','sum'),
            Tickers=('Ticker', lambda x: ', '.join(x))
        ).reset_index().sort_values('Peso_Actual', ascending=False)

        sector_group['Target_%'] = sector_group['Sector'].map(st.session_state.sector_targets).fillna(0)
        sector_group['Desvío_%'] = sector_group['Peso_Actual'] - sector_group['Target_%']
        sector_group['Monto_USD'] = sector_group['Monto_USD'].apply(fmt_usd)
        sector_group['Peso_Actual'] = sector_group['Peso_Actual'].round(2)

        st.dataframe(
            sector_group.rename(columns={
                'Sector':'Sector','Monto_USD':'Monto USD','Peso_Actual':'Peso Actual %',
                'Target_%':'Target %','Desvío_%':'Desvío %','Tickers':'Activos'
            }),
            use_container_width=True, hide_index=True
        )

        # Alerts
        for _, row in sector_group.iterrows():
            if row['Target_%'] > 0 and abs(row['Desvío_%']) > 2:
                icon = "📈" if row['Desvío_%'] > 0 else "📉"
                color = "orange" if row['Desvío_%'] > 0 else "blue"
                st.warning(f"{icon} **{row['Sector']}**: sobrecomprado en {row['Desvío_%']:.2f}%" if row['Desvío_%']>0
                    else f"{icon} **{row['Sector']}**: subcomprado en {abs(row['Desvío_%']):.2f}%")

        # Charts
        col1, col2 = st.columns(2)
        with col1:
            sg = df.groupby('Sector')['Peso_Actual_%'].sum().reset_index().sort_values('Peso_Actual_%', ascending=False)
            fig = px.pie(sg, values='Peso_Actual_%', names='Sector',
                title='Composición por Sector (Actual)',
                color_discrete_sequence=COLORS)
            fig.update_traces(textposition='inside', textinfo='label+percent',
                textfont_size=11, hole=0.3)
            fig.update_layout(paper_bgcolor='#111827', plot_bgcolor='#111827',
                font_color='#e2e8f0', showlegend=True,
                legend=dict(font=dict(size=11, color='#e2e8f0'), bgcolor='rgba(0,0,0,0)'))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            sgt = sector_group.copy()
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(name='Actual %', x=sgt['Sector'], y=sgt['Peso_Actual'],
                marker_color='rgba(0,212,255,0.7)', text=sgt['Peso_Actual'].round(1).astype(str)+'%',
                textposition='outside'))
            fig2.add_trace(go.Bar(name='Target %', x=sgt['Sector'], y=sgt['Target_%'],
                marker_color='rgba(255,107,53,0.6)'))
            fig2.update_layout(barmode='group', title='Actual vs Target por Sector',
                paper_bgcolor='#111827', plot_bgcolor='#111827',
                font_color='#e2e8f0', yaxis_title='%',
                legend=dict(font=dict(size=11, color='#e2e8f0'), bgcolor='rgba(0,0,0,0)'))
            st.plotly_chart(fig2, use_container_width=True)

        # ── Fila 2: Desvío por sector + Pareto ──
        st.markdown("---")
        col3, col4 = st.columns(2)

        with col3:
            sg_dev = sector_group.copy()
            sg_dev['Peso_Actual_num'] = df.groupby('Sector')['Peso_Actual_%'].sum().reindex(sg_dev['Sector']).values
            has_targets = (sg_dev['Target_%'] > 0).any()
            if has_targets:
                sg_dev = sg_dev[sg_dev['Target_%'] > 0].copy()
                sg_dev['Desvío_num'] = sg_dev['Peso_Actual_num'] - sg_dev['Target_%']
            else:
                avg = sg_dev['Peso_Actual_num'].mean()
                sg_dev['Desvío_num'] = sg_dev['Peso_Actual_num'] - avg
            # ✅ Fix 3: ordenado de mayor a menor (mismo criterio que Pareto)
            sg_dev = sg_dev.sort_values('Desvío_num', ascending=False)
            colors_dev = ['rgba(74,222,128,0.8)' if v >= 0 else 'rgba(248,113,113,0.8)'
                for v in sg_dev['Desvío_num']]
            fig3 = go.Figure(go.Bar(
                x=sg_dev['Sector'], y=sg_dev['Desvío_num'].round(2),
                marker_color=colors_dev,
                text=sg_dev['Desvío_num'].round(2).astype(str)+'%',
                textposition='outside'
            ))
            fig3.add_hline(y=0, line_dash='dot', line_color='rgba(255,255,255,0.2)')
            fig3.update_layout(
                title='Desvío por Sector (Actual − Target)',
                paper_bgcolor='#111827', plot_bgcolor='#111827',
                font_color='#e2e8f0', yaxis_title='Desvío %',
                showlegend=False
            )
            st.plotly_chart(fig3, use_container_width=True)

        with col4:
            sg_par = df.groupby('Sector')['Peso_Actual_%'].sum().reset_index()
            sg_par.columns = ['Sector', 'Peso']
            sg_par = sg_par.sort_values('Peso', ascending=False).reset_index(drop=True)
            sg_par['Acumulado'] = sg_par['Peso'].cumsum().round(2)

            fig4 = go.Figure()
            fig4.add_trace(go.Bar(
                x=sg_par['Sector'], y=sg_par['Peso'].round(2),
                name='Peso %',
                marker_color='rgba(0,212,255,0.7)',
                text=sg_par['Peso'].round(1).astype(str)+'%',
                textposition='outside',
                yaxis='y'
            ))
            fig4.add_trace(go.Scatter(
                x=sg_par['Sector'], y=sg_par['Acumulado'],
                name='Acumulado %',
                mode='lines+markers',
                line=dict(color='#ffffff', width=2),
                marker=dict(size=6, color='white'),
                yaxis='y2'
            ))
            fig4.update_layout(
                title='Concentración por Sector (Pareto)',
                paper_bgcolor='#111827', plot_bgcolor='#111827',
                font_color='#e2e8f0',
                yaxis=dict(title='Peso %', gridcolor='rgba(30,45,69,0.6)'),
                yaxis2=dict(title='Acum %', overlaying='y', side='right',
                    range=[0, 105], showgrid=False, ticksuffix='%'),
                legend=dict(font=dict(size=11, color='#e2e8f0'), bgcolor='rgba(0,0,0,0)')
            )
            st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════
#  TAB 3 — REBALANCEO
# ══════════════════════════════════════════
with tabs[2]:
    st.subheader("⚖️ Rebalanceo")
    if st.session_state.portfolio.empty:
        st.info("Cargá tu cartera primero")
    else:
        df = calc_portfolio_weights(st.session_state.portfolio.copy())
        total = df['Total'].iloc[0]

        m1,m2,m3,m4,m5 = st.columns(5)
        m1.metric("Total", fmt_usd(total))
        m2.metric("Activos", len(df))
        m3.metric("A comprar", len(df[df['Desvio_%']<-0.5]), delta="subweight")
        m4.metric("A vender", len(df[df['Desvio_%']>0.5]), delta="overweight")
        vol_mover = df[df['Rebalanceo_USD']>0]['Rebalanceo_USD'].sum()
        m5.metric("Volumen a mover", fmt_usd(vol_mover))

        st.markdown("#### Tabla de rebalanceo (ordenada por acción)")
        df_reb = df.sort_values('Rebalanceo_USD').copy()
        df_reb['Acción'] = df_reb['Rebalanceo_USD'].apply(
            lambda x: '▲ COMPRAR' if x>50 else ('▼ VENDER' if x<-50 else '✓ OK'))

        def color_rebalanceo(val):
            if isinstance(val, (int,float)):
                if val > 50: return 'color: #4ade80; font-weight: bold'
                if val < -50: return 'color: #f87171; font-weight: bold'
            return ''

        display_cols = ['Ticker','Sector','Monto_USD','Target_USD','Rebalanceo_USD',
                        'Peso_Actual_%','Target_%','Desvio_%','Acción']
        st.dataframe(
            df_reb[display_cols].round(2).style.map(color_rebalanceo, subset=['Rebalanceo_USD']),
            use_container_width=True, hide_index=True
        )

        col1, col2 = st.columns(2)
        with col1:
            df_sorted = df.sort_values('Peso_Actual_%', ascending=False)
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Actual %', x=df_sorted['Ticker'], y=df_sorted['Peso_Actual_%'],
                marker_color='rgba(0,212,255,0.75)', text=df_sorted['Peso_Actual_%'].round(1).astype(str)+'%',
                textposition='outside'))
            fig.add_trace(go.Bar(name='Target %', x=df_sorted['Ticker'], y=df_sorted['Target_%'],
                marker_color='rgba(255,107,53,0.65)'))
            fig.update_layout(barmode='group', title='Tenencia Actual vs Target',
                paper_bgcolor='#111827', plot_bgcolor='#111827',
                font_color='#e2e8f0', yaxis_title='%')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = px.pie(df, values='Peso_Actual_%', names='Ticker',
                title='Composición actual',
                color_discrete_sequence=px.colors.qualitative.Set3)
            fig2.update_traces(textposition='inside', textinfo='label+percent',
                textfont_size=10, hole=0.25)
            fig2.update_layout(paper_bgcolor='#111827', plot_bgcolor='#111827',
                font_color='#e2e8f0', showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

        buf = io.BytesIO()
        df_reb[display_cols].round(2).to_excel(buf, index=False)
        st.download_button("⬇️ Exportar rebalanceo Excel", buf.getvalue(), "rebalanceo.xlsx")


# ══════════════════════════════════════════
#  TAB 4 — CORRELACIÓN
# ══════════════════════════════════════════
with tabs[3]:
    st.subheader("🔗 Correlación (datos reales)")
    if st.session_state.portfolio.empty:
        st.info("Cargá tu cartera primero")
    elif st.session_state.hist_data is None:
        st.warning("⬅️ Hacé click en **Cargar datos de mercado** en el panel lateral")
    else:
        prices = st.session_state.hist_data
        available = [t for t in st.session_state.portfolio['Ticker'].tolist() if t in prices.columns]
        if len(available) < 2:
            st.warning("Se necesitan al menos 2 tickers con datos disponibles")
        else:
            prices_clean = prices[available].dropna()
            returns = prices_clean.pct_change().dropna()
            corr = returns.corr().round(3)

            # Alert high correlations
            high = []
            for i in range(len(corr.columns)):
                for j in range(i+1, len(corr.columns)):
                    val = corr.iloc[i,j]
                    if val > 0.80:
                        high.append(f"{corr.columns[i]}–{corr.columns[j]}: **{val:.2f}**")
            if high:
                st.error(f"⚠️ Alta correlación (>0.80): {' | '.join(high)} — considerá reducir exposición")
            else:
                st.success("✅ Diversificación correcta — ninguna correlación supera 0.80")

            # Heatmap
            fig = px.imshow(corr, text_auto=True, aspect='auto',
                color_continuous_scale='RdYlGn_r',
                title=f'Matriz de Correlación ({years} años de historia)',
                zmin=-1, zmax=1)
            fig.update_layout(paper_bgcolor='#111827', plot_bgcolor='#111827',
                font_color='#e2e8f0', height=500)
            fig.update_traces(textfont_size=11)
            st.plotly_chart(fig, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                avg_corr = corr.apply(lambda x: x[x.index!=x.name].mean()).sort_values(ascending=False)
                colors_corr = ['rgba(248,113,113,.8)' if v>0.6 else 'rgba(251,191,36,.8)' if v>0.4
                    else 'rgba(74,222,128,.8)' for v in avg_corr]
                fig2 = go.Figure(go.Bar(x=avg_corr.index, y=avg_corr.values,
                    marker_color=colors_corr,
                    text=avg_corr.round(2).values, textposition='outside'))
                fig2.update_layout(title='Correlación promedio por activo',
                    paper_bgcolor='#111827', plot_bgcolor='#111827',
                    font_color='#e2e8f0', yaxis=dict(range=[0,1]))
                st.plotly_chart(fig2, use_container_width=True)

            with col2:
                # Rolling correlation for top pair
                pairs = [(corr.columns[i], corr.columns[j], corr.iloc[i,j])
                    for i in range(len(corr.columns)) for j in range(i+1,len(corr.columns))]
                pairs.sort(key=lambda x: abs(x[2]), reverse=True)
                if pairs:
                    t1, t2, _ = pairs[0]
                    rolling = returns[t1].rolling(60).corr(returns[t2]).dropna()
                    fig3 = go.Figure(go.Scatter(x=rolling.index, y=rolling.values,
                        fill='tozeroy', line=dict(color='#00d4ff', width=2)))
                    fig3.add_hline(y=0.8, line_dash='dash', line_color='#f87171',
                        annotation_text='Alto (0.80)')
                    fig3.update_layout(title=f'Correlación móvil 60d: {t1} vs {t2}',
                        paper_bgcolor='#111827', plot_bgcolor='#111827',
                        font_color='#e2e8f0', yaxis_title='Correlación')
                    st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════
#  TAB 5 — MÉTRICAS
# ══════════════════════════════════════════
with tabs[4]:
    st.subheader("📐 Métricas reales")
    if st.session_state.portfolio.empty:
        st.info("Cargá tu cartera primero")
    elif st.session_state.hist_data is None:
        st.warning("⬅️ Hacé click en **Cargar datos de mercado** en el panel lateral")
    else:
        prices = st.session_state.hist_data
        df_port = calc_portfolio_weights(st.session_state.portfolio.copy())
        available = [t for t in df_port['Ticker'].tolist() if t in prices.columns]
        prices_clean = prices[available].dropna()

        with st.spinner("Calculando métricas..."):
            try:
                metrics, returns = calc_metrics(prices_clean, rf=rf_rate)
            except Exception as e:
                st.error(f"❌ No se pudieron calcular las métricas: {e}\n\nAsegurate de que los tickers tengan datos suficientes y de haber hecho click en **Cargar datos de mercado**.")
                st.stop()

        # Portfolio-level
        weights_actual = np.array([
            df_port[df_port['Ticker']==t]['Peso_Actual_%'].values[0]/100
            for t in available if t in df_port['Ticker'].values
        ])
        if len(weights_actual) == len(available):
            port_ret, port_vol, port_sharpe = portfolio_metrics(weights_actual, returns, rf_rate)
            port_var1d = (-np.dot(weights_actual, returns.quantile(0.05))) * 100
            beta_values = metrics['Beta SPY'].dropna()
            port_beta = float(np.dot(
                [w for t,w in zip(available, weights_actual) if t in metrics.index and not np.isnan(metrics.loc[t,'Beta SPY'])],
                beta_values.reindex([t for t in available if t in beta_values.index]).dropna().values
            )) if len(beta_values) > 0 else np.nan

            m1,m2,m3,m4,m5,m6 = st.columns(6)
            m1.metric("Retorno Anual", f"{port_ret*100:.1f}%")
            m2.metric("Volatilidad", f"{port_vol*100:.1f}%")
            m3.metric("Sharpe", f"{port_sharpe:.2f}")
            m4.metric("Beta vs SPY", f"{port_beta:.2f}" if not np.isnan(port_beta) else "—")
            m5.metric("VaR 1d 95%", f"{port_var1d:.2f}%")
            m6.metric("VaR 10d 95%", f"{port_var1d*np.sqrt(10):.2f}%")
            st.session_state['port_stats'] = {'ret':port_ret,'vol':port_vol,'sharpe':port_sharpe,'beta':port_beta if not np.isnan(port_beta) else 1.0}

        # Table
        st.markdown("#### Métricas por activo")
        metrics_display = metrics.copy()
        metrics_display.index.name = 'Ticker'
        st.dataframe(metrics_display, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            sharpe_sorted = metrics['Sharpe'].sort_values(ascending=True)
            colors_s = ['rgba(74,222,128,.8)' if v>1 else 'rgba(0,212,255,.7)' if v>0.5
                else 'rgba(248,113,113,.8)' for v in sharpe_sorted]
            fig = go.Figure(go.Bar(y=sharpe_sorted.index, x=sharpe_sorted.values,
                orientation='h', marker_color=colors_s,
                text=sharpe_sorted.round(2).values, textposition='outside'))
            fig.update_layout(title='Sharpe Ratio por activo',
                paper_bgcolor='#111827', plot_bgcolor='#111827', font_color='#e2e8f0')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Risk-Return scatter
            fig2 = go.Figure()
            for i, ticker in enumerate(metrics.index):
                fig2.add_trace(go.Scatter(
                    x=[metrics.loc[ticker,'Volatilidad %']],
                    y=[metrics.loc[ticker,'Retorno Anual %']],
                    mode='markers+text', name=ticker,
                    text=[ticker], textposition='top center',
                    marker=dict(size=12, color=COLORS[i % len(COLORS)])
                ))
            fig2.update_layout(title='Mapa Riesgo vs Retorno',
                paper_bgcolor='#111827', plot_bgcolor='#111827',
                font_color='#e2e8f0', showlegend=False,
                xaxis_title='Volatilidad %', yaxis_title='Retorno Anual %')
            st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════
#  TAB 6 — STRESS TEST
# ══════════════════════════════════════════
with tabs[5]:
    st.subheader("🔥 Stress Test")
    if st.session_state.portfolio.empty:
        st.info("Cargá tu cartera primero")
    elif st.session_state.hist_data is None:
        st.warning("⬅️ Cargá datos de mercado primero")
    else:
        port_stats = st.session_state.get('port_stats', {'ret':0.18,'vol':0.22,'beta':1.1})
        df_port = calc_portfolio_weights(st.session_state.portfolio.copy())
        total = df_port['Total'].iloc[0]
        beta = port_stats.get('beta', 1.1)

        st.markdown(f"**Beta de tu cartera vs SPY:** {beta:.2f}")
        st.markdown("---")

        scenarios = [
            ("Corrección leve", -0.05, "2022 drawdown"),
            ("Corrección moderada", -0.10, "Corrección típica"),
            ("Crash severo", -0.20, "Bear market"),
            ("Crisis 2008 (Lehman)", -0.38, "Histórico"),
            ("COVID-19 Crash (Feb-Mar 2020)", -0.34, "Histórico"),
            ("Flash Crash hipotético", -0.50, "Extremo"),
        ]

        st.markdown("#### Escenarios hipotéticos")
        data_stress = []
        for label, spy_drop, tipo in scenarios:
            impact = spy_drop * beta
            usd_loss = impact * total
            data_stress.append({
                'Escenario': label,
                'Tipo': tipo,
                'SPY cae': f"{spy_drop*100:.0f}%",
                'Impacto cartera': f"{impact*100:.2f}%",
                'Pérdida USD': fmt_usd(abs(usd_loss)),
                '_impact_val': impact
            })

        df_stress = pd.DataFrame(data_stress)

        def color_stress(val):
            if isinstance(val, str) and '%' in val and val.startswith('-'):
                try:
                    v = float(val.replace('%',''))
                    if v < -30: return 'background-color: rgba(220,38,38,0.4); color: #fca5a5'
                    if v < -15: return 'background-color: rgba(234,88,12,0.3); color: #fdba74'
                    return 'background-color: rgba(202,138,4,0.2); color: #fde68a'
                except: pass
            return ''

        st.dataframe(
            df_stress[['Escenario','Tipo','SPY cae','Impacto cartera','Pérdida USD']]\
                .style.map(color_stress, subset=['Impacto cartera']),
            use_container_width=True, hide_index=True
        )

        # Chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df_stress['Escenario'],
            y=[float(r['_impact_val'])*100 for _,r in df_stress.iterrows()],
            marker_color=[f'rgba(220,38,38,{min(0.9,abs(r["_impact_val"])*2+0.2)})' for _,r in df_stress.iterrows()],
            text=[r['Impacto cartera'] for _,r in df_stress.iterrows()],
            textposition='outside'
        ))
        fig.update_layout(title='Impacto estimado por escenario',
            paper_bgcolor='#111827', plot_bgcolor='#111827',
            font_color='#e2e8f0', yaxis_title='% impacto', xaxis_tickangle=-25)
        st.plotly_chart(fig, use_container_width=True)

        # Historical drawdown from real data
        if st.session_state.hist_data is not None:
            st.markdown("#### Drawdown histórico real de tu cartera")
            prices = st.session_state.hist_data
            df_p = calc_portfolio_weights(st.session_state.portfolio.copy())
            available = [t for t in df_p['Ticker'].tolist() if t in prices.columns]
            if available:
                w = np.array([df_p[df_p['Ticker']==t]['Peso_Actual_%'].values[0]/100
                    for t in available])
                port_prices = (prices[available] * w).sum(axis=1)
                port_prices = port_prices / port_prices.iloc[0] * 100
                rolling_max = port_prices.cummax()
                drawdown = (port_prices - rolling_max) / rolling_max * 100

                fig2 = go.Figure(go.Scatter(x=drawdown.index, y=drawdown.values,
                    fill='tozeroy', line=dict(color='#f87171', width=1.5),
                    fillcolor='rgba(248,113,113,0.15)'))
                fig2.update_layout(title='Drawdown histórico real',
                    paper_bgcolor='#111827', plot_bgcolor='#111827',
                    font_color='#e2e8f0', yaxis_title='Drawdown %')
                st.plotly_chart(fig2, use_container_width=True)
                st.info(f"📉 Máximo drawdown histórico: **{drawdown.min():.2f}%**")


# ══════════════════════════════════════════
#  TAB 7 — BLACK-LITTERMAN
# ══════════════════════════════════════════
with tabs[6]:
    st.subheader("🧠 Black-Litterman")
    if st.session_state.portfolio.empty:
        st.info("Cargá tu cartera primero")
    elif st.session_state.hist_data is None:
        st.warning("⬅️ Cargá datos de mercado primero")
    else:
        prices = st.session_state.hist_data
        df_port = calc_portfolio_weights(st.session_state.portfolio.copy())
        available = [t for t in df_port['Ticker'].tolist() if t in prices.columns]
        prices_clean = prices[available].dropna()
        returns = prices_clean.pct_change().dropna()

        st.markdown("#### Ingresá tus views (expectativas de retorno)")
        st.caption("q % anual = tu expectativa de retorno anual para ese activo | Confianza 0–1 = cuánto confiás en ese view")

        views_q, views_conf = {}, {}
        cols = st.columns(3)
        for i, ticker in enumerate(available):
            with cols[i % 3]:
                st.markdown(f"**{ticker}**")
                c1, c2 = st.columns(2)
                views_q[ticker] = c1.number_input(f"q % {ticker}", value=15.0, step=1.0, key=f"q_{ticker}") / 100
                views_conf[ticker] = c2.number_input(f"conf {ticker}", value=0.7, min_value=0.0, max_value=1.0, step=0.05, key=f"c_{ticker}")

        if st.button("🧠 Correr Black-Litterman", type="primary"):
            w_actual = np.array([df_port[df_port['Ticker']==t]['Peso_Actual_%'].values[0]/100
                for t in available])

            posterior_mu = black_litterman(w_actual, returns, views_q, views_conf, rf=rf_rate)
            prior_mu = returns.mean() * 252

            # Read sidebar settings
            min_w = st.session_state.get('min_weight', 0.0)
            target_ret = st.session_state.get('target_return', None)

            # Optimal BL weights
            w_bl = max_sharpe(returns, rf=rf_rate, min_weight=min_w)
            w_minvar = min_variance(returns, min_weight=min_w)

            # Target return portfolio (if enabled)
            w_target = None
            if target_ret is not None:
                w_target = target_return_weights(returns, target_ret, min_weight=min_w)
                if w_target is None:
                    st.warning(f"⚠️ No se encontró solución para retorno objetivo {target_ret*100:.1f}% — puede estar fuera del rango alcanzable con estos activos.")

            # Results table
            bl_df = pd.DataFrame({
                'Ticker': available,
                'Prior (eq.) %': (prior_mu.values * 100).round(2),
                'Posterior BL %': (posterior_mu.values * 100).round(2),
                'Delta %': ((posterior_mu.values - prior_mu.values) * 100).round(2),
                'Peso BL %': (w_bl * 100).round(2),
                'Peso Actual %': (w_actual * 100).round(2),
            }).sort_values('Delta %', ascending=False)

            st.dataframe(bl_df, use_container_width=True, hide_index=True)

            col1, col2 = st.columns(2)
            with col1:
                colors_bl = ['rgba(74,222,128,.8)' if v>=0 else 'rgba(248,113,113,.8)'
                    for v in bl_df['Delta %']]
                fig = go.Figure(go.Bar(x=bl_df['Ticker'], y=bl_df['Delta %'],
                    marker_color=colors_bl, text=bl_df['Delta %'].astype(str)+'%',
                    textposition='outside'))
                fig.update_layout(title='Delta BL (Posterior − Prior)',
                    paper_bgcolor='#111827', plot_bgcolor='#111827',
                    font_color='#e2e8f0', yaxis_title='puntos % anual')
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(name='Actual %', x=bl_df['Ticker'],
                    y=bl_df['Peso Actual %'], marker_color='rgba(0,212,255,.7)'))
                fig2.add_trace(go.Bar(name='BL Optimo %', x=bl_df['Ticker'],
                    y=bl_df['Peso BL %'], marker_color='rgba(255,107,53,.7)'))
                fig2.update_layout(barmode='group', title='Pesos: Actual vs BL Óptimo',
                    paper_bgcolor='#111827', plot_bgcolor='#111827',
                    font_color='#e2e8f0', yaxis_title='%')
                st.plotly_chart(fig2, use_container_width=True)

            # Frontier scatter
            port_actual = portfolio_metrics(w_actual, returns, rf_rate)
            port_bl = portfolio_metrics(w_bl, returns, rf_rate)
            port_mv = portfolio_metrics(w_minvar, returns, rf_rate)

            scatter_ports = [
                ('Actual',      port_actual, '#00d4ff', 'x'),
                ('BL Óptimo',   port_bl,     '#f87171', 'square'),
                ('Min Varianza',port_mv,     '#4ade80', 'circle'),
            ]
            if w_target is not None:
                port_tgt = portfolio_metrics(w_target, returns, rf_rate)
                scatter_ports.append((f'Ret.Obj {target_ret*100:.0f}%', port_tgt, '#fbbf24', 'diamond'))

            fig3 = go.Figure()
            for label, stats, color, symbol in scatter_ports:
                fig3.add_trace(go.Scatter(x=[stats[1]*100], y=[stats[0]*100],
                    mode='markers+text', name=f"{label} (Sharpe:{stats[2]:.2f})",
                    text=[label], textposition='top center',
                    marker=dict(size=16, color=color, symbol=symbol,
                        line=dict(width=2, color=color))))
            fig3.update_layout(title='Frontera Eficiente: Actual vs BL vs MinVar',
                paper_bgcolor='#111827', plot_bgcolor='#111827', font_color='#e2e8f0',
                xaxis_title='Volatilidad %', yaxis_title='Retorno %')
            st.plotly_chart(fig3, use_container_width=True)

            # Store for MC — prefer target return portfolio if set
            best = port_tgt if (w_target is not None) else port_bl
            st.session_state['bl_stats'] = {
                'ret_bl': best[0], 'vol_bl': best[1]
            }


# ══════════════════════════════════════════
#  TAB 8 — MONTE CARLO
# ══════════════════════════════════════════
with tabs[7]:
    st.subheader("🎲 Monte Carlo")
    if st.session_state.portfolio.empty:
        st.info("Cargá tu cartera primero")
    elif st.session_state.hist_data is None:
        st.warning("⬅️ Cargá datos de mercado primero")
    else:
        df_port = calc_portfolio_weights(st.session_state.portfolio.copy())
        total = df_port['Total'].iloc[0]
        port_stats = st.session_state.get('port_stats', {'ret':0.18,'vol':0.22})
        bl_stats = st.session_state.get('bl_stats', None)

        col1, col2, col3 = st.columns(3)
        n_sims = col1.number_input("N° simulaciones", value=500, min_value=100, max_value=2000, step=100)
        n_days = col2.number_input("Días de trading", value=252, min_value=60, max_value=756, step=21)
        show_paths = col3.number_input("Trayectorias a mostrar", value=60, min_value=10, max_value=200, step=10)

        if st.button("🎲 Correr simulación", type="primary"):
            mu_act = port_stats['ret']
            vol_act = port_stats['vol']
            mu_bl = bl_stats['ret_bl'] if bl_stats else mu_act * 1.15
            vol_bl = bl_stats['vol_bl'] if bl_stats else vol_act * 0.88

            with st.spinner("Simulando..."):
                paths_act = monte_carlo(mu_act, vol_act, total, int(n_sims), int(n_days))
                paths_bl  = monte_carlo(mu_bl,  vol_bl,  total, int(n_sims), int(n_days))

            mean_act = paths_act.mean(axis=0)
            mean_bl  = paths_bl.mean(axis=0)
            days_x = list(range(int(n_days)+1))

            # MC chart
            fig = go.Figure()
            step = max(1, int(n_sims)//int(show_paths))
            for i in range(0, int(n_sims), step):
                fig.add_trace(go.Scatter(x=days_x, y=paths_act[i], mode='lines',
                    line=dict(color='rgba(0,212,255,0.06)', width=1), showlegend=False))
                fig.add_trace(go.Scatter(x=days_x, y=paths_bl[i], mode='lines',
                    line=dict(color='rgba(255,107,53,0.06)', width=1), showlegend=False))
            fig.add_trace(go.Scatter(x=days_x, y=mean_act, mode='lines', name='Media Actual',
                line=dict(color='#00d4ff', width=3)))
            fig.add_trace(go.Scatter(x=days_x, y=mean_bl, mode='lines', name='Media BL Optimizada',
                line=dict(color='#ff6b35', width=3)))
            fig.update_layout(
                title=f'Proyección Monte Carlo ({int(n_days)} días) — Capital inicial: {fmt_usd(total)}',
                paper_bgcolor='#111827', plot_bgcolor='#111827', font_color='#e2e8f0',
                xaxis_title='Días de Trading', yaxis_title='Valor USD',
                yaxis=dict(tickformat='$,.0f'))
            st.plotly_chart(fig, use_container_width=True)

            final_act = paths_act[:, -1]
            final_bl  = paths_bl[:, -1]
            med_act = final_act.mean()
            med_bl  = final_bl.mean()
            gain = (med_bl - med_act) / med_act * 100

            m1,m2,m3,m4,m5,m6 = st.columns(6)
            m1.metric("Capital inicial", fmt_usd(total))
            m2.metric("Media Actual (1 año)", fmt_usd(med_act))
            m3.metric("Media BL Optimizada", fmt_usd(med_bl), delta=f"+{gain:.1f}%")
            m4.metric("Peor 5% Actual", fmt_usd(np.percentile(final_act,5)))
            m5.metric("Peor 5% BL", fmt_usd(np.percentile(final_bl,5)))
            m6.metric("Mejor 95% BL", fmt_usd(np.percentile(final_bl,95)))

            col1, col2 = st.columns(2)
            with col1:
                all_vals = np.concatenate([final_act, final_bl])
                bins = np.linspace(all_vals.min(), all_vals.max(), 35)
                hist_act, _ = np.histogram(final_act, bins=bins)
                hist_bl,  _ = np.histogram(final_bl,  bins=bins)
                bin_centers = (bins[:-1] + bins[1:]) / 2

                fig2 = go.Figure()
                fig2.add_trace(go.Bar(x=bin_centers, y=hist_act,
                    name=f'Actual (Media: {fmt_usd(med_act)})',
                    marker_color='rgba(0,212,255,0.55)'))
                fig2.add_trace(go.Bar(x=bin_centers, y=hist_bl,
                    name=f'BL Optim (Media: {fmt_usd(med_bl)})',
                    marker_color='rgba(255,107,53,0.55)'))
                fig2.update_layout(barmode='overlay', title='Distribución de valor final',
                    paper_bgcolor='#111827', plot_bgcolor='#111827', font_color='#e2e8f0',
                    xaxis=dict(tickformat='$,.0f'), yaxis_title='Frecuencia')
                st.plotly_chart(fig2, use_container_width=True)

            with col2:
                pcts = [5,10,25,50,75,90,95]
                fig3 = go.Figure()
                fig3.add_trace(go.Bar(x=[f'P{p}' for p in pcts],
                    y=[np.percentile(final_act,p) for p in pcts],
                    name='Actual', marker_color='rgba(0,212,255,0.7)'))
                fig3.add_trace(go.Bar(x=[f'P{p}' for p in pcts],
                    y=[np.percentile(final_bl,p) for p in pcts],
                    name='BL Optim', marker_color='rgba(255,107,53,0.7)'))
                fig3.update_layout(barmode='group', title='Percentiles de resultado',
                    paper_bgcolor='#111827', plot_bgcolor='#111827', font_color='#e2e8f0',
                    yaxis=dict(tickformat='$,.0f'))
                st.plotly_chart(fig3, use_container_width=True)
