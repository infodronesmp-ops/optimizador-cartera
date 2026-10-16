import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.optimize import minimize
import warnings
import json
import os
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
#  PERSISTENCIA EN DISCO
# ─────────────────────────────────────────
DATA_FILE = "cartera_data.json"

def load_persistent():
    """Carga datos guardados en disco. Si no existe el archivo, devuelve defaults."""
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, 'r') as f:
                data = json.load(f)
            return data
        except:
            pass
    return None

def save_persistent(portfolio_df, instruments_df, sectors_list, sector_targets_dict=None):
    """Guarda cartera, instrumentos, sectores y targets de sector en disco."""
    try:
        data = {
            'portfolio': portfolio_df.to_dict(orient='records'),
            'instruments': instruments_df.to_dict(orient='records'),
            'sectors': sectors_list,
            'sector_targets': sector_targets_dict or {},
        }
        # Cartera total importada de Balanz (liquidez, renta fija y variable)
        _bz = st.session_state.get('balanz_data') if hasattr(st, 'session_state') else None
        if _bz is not None and len(_bz):
            data['balanz_data'] = _bz.to_dict(orient='records')
            data['balanz_tipo_cambio'] = float(st.session_state.get('balanz_tipo_cambio', 0) or 0)
            data['balanz_tc_fuente'] = st.session_state.get('balanz_tc_fuente', '')
        with open(DATA_FILE, 'w') as f:
            json.dump(data, f, ensure_ascii=False)
        return True
    except Exception as e:
        st.error(f"Error al guardar: {e}")
        return False

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

# Cartera preseteada por defecto — el usuario puede modificarla y guardar cambios
DEFAULT_PORTFOLIO = pd.DataFrame([
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

DEFAULT_SECTORS = [
    'Tecnología','Comunicaciones','Salud','Financiero',
    'Energía','Consumo','Industriales','Brasil/EM',
    'Minería/Cobre','Defensa','Otro'
]

# ── Inicializar session state cargando desde disco ──
if 'app_loaded' not in st.session_state:
    saved = load_persistent()
    if saved:
        st.session_state.portfolio       = pd.DataFrame(saved.get('portfolio', DEFAULT_PORTFOLIO.to_dict('records')))
        st.session_state.instruments     = pd.DataFrame(saved.get('instruments', DEFAULT_PORTFOLIO[['Ticker','Sector']].to_dict('records')))
        st.session_state.sectors         = saved.get('sectors', DEFAULT_SECTORS.copy())
        st.session_state.sector_targets  = saved.get('sector_targets', {})
        if saved.get('balanz_data'):
            st.session_state.balanz_data       = pd.DataFrame(saved['balanz_data'])
            st.session_state.balanz_tipo_cambio = saved.get('balanz_tipo_cambio', 1400.0)
            st.session_state.balanz_tc_fuente   = saved.get('balanz_tc_fuente', '')
    else:
        st.session_state.portfolio       = DEFAULT_PORTFOLIO.copy()
        st.session_state.instruments     = DEFAULT_PORTFOLIO[['Ticker','Sector']].copy().reset_index(drop=True)
        st.session_state.sectors         = DEFAULT_SECTORS.copy()
        st.session_state.sector_targets  = {}
    st.session_state.hist_data      = None
    st.session_state.tickers_loaded  = []
    st.session_state.app_loaded      = True
if 'bench_data' not in st.session_state:
    st.session_state.bench_data = None
if 'balanz_data' not in st.session_state:
    st.session_state.balanz_data = None          # DataFrame importado de Balanz
if 'balanz_usd_tickers' not in st.session_state:
    st.session_state.balanz_usd_tickers = []     # Tickers con V.Actual en USD
if 'balanz_tipo_cambio' not in st.session_state:
    st.session_state.balanz_tipo_cambio = 1400.0 # Tipo de cambio default

def sync_instrument(ticker, sector=""):
    """Agrega un ticker al catálogo si no existe, y guarda en disco."""
    if ticker and ticker not in st.session_state.instruments['Ticker'].values:
        new_row = pd.DataFrame([{'Ticker': ticker, 'Sector': sector}])
        st.session_state.instruments = pd.concat(
            [st.session_state.instruments, new_row], ignore_index=True
        )
        save_persistent(st.session_state.portfolio, st.session_state.instruments, st.session_state.sectors, st.session_state.get("sector_targets",{}))

# ─────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────
COLORS = px.colors.qualitative.Set2

def fmt_usd(v):
    return f"u$s {v:,.0f}"

def fmt_pct(v):
    return f"{v:.2f}%"

# ─────────────────────────────────────────
#  TIPO DE CAMBIO (fuente objetiva)
# ─────────────────────────────────────────
TC_FUENTES = {
    "MEP (automático)": ("bolsa", "Dólar MEP"),
    "CCL (automático)": ("contadoconliqui", "Dólar CCL"),
}

@st.cache_data(ttl=900, show_spinner=False)
def obtener_tc(casa):
    """Cotización de dolarapi.com (agrega datos de mercado). Devuelve dict o None."""
    import requests
    try:
        r = requests.get(f"https://dolarapi.com/v1/dolares/{casa}", timeout=8)
        r.raise_for_status()
        d = r.json()
        venta = float(d.get('venta') or 0)
        if venta <= 0:
            return None
        fecha = pd.to_datetime(d.get('fechaActualizacion'), utc=True, errors='coerce')
        if pd.notna(fecha):
            fecha = fecha.tz_convert('America/Argentina/Buenos_Aires')
        return {'venta': venta, 'compra': float(d.get('compra') or 0), 'fecha': fecha}
    except Exception:
        return None

import re as _re
_RE_USD = _re.compile(r'(u\s*\$\s*[sd]?|us\s*\$|usd)', _re.I)

# Tipos de instrumento cuyo V. Actual se carga EN PESOS (todo lo demás se toma en DÓLARES)


def leer_valor(raw):
    """Lee un V. Actual. Devuelve (numero, moneda_del_texto) donde moneda_del_texto es
    'USD' (trae u$s/usd), 'ARS' (trae $ solo) o None (número sin símbolo). (None, None) si no se puede leer."""
    if raw is None:
        return None, None
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return (None, None) if pd.isna(raw) else (float(raw), None)
    s = str(raw).strip()
    if not s or s in ('-', '#N/A', 'None', 'nan'):
        return None, None
    if _RE_USD.search(s):
        moneda = 'USD'
    elif '$' in s:
        moneda = 'ARS'
    else:
        moneda = None
    num = _re.sub(r'[^0-9,.\-]', '', s)          # saca letras, símbolos y espacios
    if not num or num in ('-', '.', ','):
        return None, None
    try:
        return float(num.replace('.', '').replace(',', '.')), moneda
    except ValueError:
        return None, None

# ─────────────────────────────────────────
#  CAPA DE DATOS
# ─────────────────────────────────────────
# Ticker de Balanz / local → ticker de Yahoo Finance.
# Acciones argentinas con ADR en NYSE: se usa el ADR (precio real en USD).
# Acciones argentinas sin ADR: ticker .BA, convertido a USD con CCL implícito.
TICKER_MAP_YF = {
    'YPFD': 'YPF', 'PAMP': 'PAM', 'TGSU2': 'TGS', 'CEPU': 'CEPU', 'GGAL': 'GGAL',
    'BMA': 'BMA', 'BBAR': 'BBAR', 'SUPV': 'SUPV', 'EDN': 'EDN', 'LOMA': 'LOMA',
    'IRSA': 'IRS', 'CRES': 'CRESY', 'TECO2': 'TEO', 'VIST': 'VIST',
    'METR': 'METR.BA', 'TGNO4': 'TGNO4.BA',
    'BRKB': 'BRK-B', 'BRK.B': 'BRK-B', 'BTC': 'BTC-USD',
}

def yf_ticker(t):
    t = str(t).strip()
    return TICKER_MAP_YF.get(t.upper(), t)

BENCHMARK = 'SPY'

def ticker_yf_balanz(ticker, instrumento):
    """Ticker de Balanz/Excel → ticker de Yahoo, según el tipo de instrumento."""
    t = str(ticker).strip()
    inst = str(instrumento or '').strip()
    if inst == 'Acciones ARG':
        return TICKER_MAP_YF.get(t, f"{t}.BA")      # con ADR → ADR; sin ADR → BYMA (se convierte con CCL)
    if inst == 'Bitcoin':
        return 'BTC-USD'
    return yf_ticker(t)                              # CEDEAR → subyacente en USD

def _calcular_ccl(px_):
    """CCL implícito diario a partir de un par local/ADR. Devuelve (serie, fuente) o (None, None)."""
    pares = [('YPFD.BA', 'YPF', 1.0, 'YPF'), ('GGAL.BA', 'GGAL', 10.0, 'GGAL')]
    for local, adr, ratio, nombre in pares:
        if local in px_.columns and adr in px_.columns:
            c = px_[local].ffill(limit=3) * ratio / px_[adr].ffill(limit=3)
            c = c.replace([np.inf, -np.inf], np.nan)
            if c.notna().sum() > 60:
                # filtrar saltos espurios por feriados no coincidentes (desvío >10% vs mediana de 5 días)
                med = c.rolling(5, center=True, min_periods=1).median()
                c[(c / med - 1).abs() > 0.10] = np.nan
                return c.ffill(), nombre
    return None, None

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_data(tickers, period="5y"):
    """Descarga precios de cierre ajustados de Yahoo Finance, todos expresados en USD
    y alineados a los días hábiles del mercado de EEUU.
    Devuelve (precios, benchmark_SPY, notas)."""
    tickers = list(dict.fromkeys(yf_ticker(t) for t in tickers if t and str(t).strip()))
    if not tickers:
        return None, None, ["No hay tickers válidos para descargar."]
    notas = []
    ars_tickers = [t for t in tickers if t.upper().endswith('.BA')]
    extra = [BENCHMARK]
    if ars_tickers:
        extra += ['YPF', 'YPFD.BA', 'GGAL', 'GGAL.BA', 'ARS=X']
    descarga = list(dict.fromkeys(tickers + extra))
    try:
        raw = yf.download(descarga, period=period, auto_adjust=True, progress=False)
    except Exception as e:
        return None, None, [f"Error al descargar datos: {e}"]
    if raw is None or len(raw) == 0:
        return None, None, ["Yahoo Finance no devolvió datos."]
    px_ = raw['Close'] if isinstance(raw.columns, pd.MultiIndex) else raw[['Close']].rename(columns={'Close': descarga[0]})
    px_ = px_.dropna(how='all')

    if BENCHMARK not in px_.columns or px_[BENCHMARK].dropna().empty:
        return None, None, ["No se pudo descargar el benchmark (SPY)."]
    dias = px_[BENCHMARK].dropna().index   # calendario de EEUU (saca fines de semana de BTC)

    if ars_tickers:
        ccl, fuente = _calcular_ccl(px_)
        if ccl is None and 'ARS=X' in px_.columns:
            ccl, fuente = px_['ARS=X'].ffill(), None
            notas.append("⚠️ No se pudo calcular el CCL implícito; las acciones argentinas sin ADR se convirtieron con el dólar oficial (ARS=X). Sus retornos en USD pueden estar distorsionados.")
        if ccl is None:
            notas.append(f"⚠️ Sin tipo de cambio disponible: se excluyen {', '.join(ars_tickers)}.")
            px_ = px_.drop(columns=ars_tickers, errors='ignore')
        else:
            for t in ars_tickers:
                if t in px_.columns:
                    px_[t] = px_[t].ffill(limit=3) / ccl
            if fuente:
                notas.append(f"💱 Acciones argentinas sin ADR ({', '.join(ars_tickers)}) convertidas a USD con CCL implícito diario (par {fuente}).")

    cols = [t for t in tickers if t in px_.columns and px_[t].dropna().size > 0]
    faltan = [t for t in tickers if t not in cols]
    if faltan:
        notas.append(f"⚠️ Sin datos en Yahoo Finance: {', '.join(faltan)}.")
    precios = px_[cols].reindex(dias)
    bench = px_[BENCHMARK].reindex(dias)
    return precios, bench, notas

def preparar_precios(prices, tickers, min_obs=60):
    """Prepara una matriz de precios en VENTANA COMÚN (todos los activos con datos reales).
    No rellena hacia atrás: un activo nuevo no inventa historia con retornos cero."""
    disp = [t for t in dict.fromkeys(tickers) if t in prices.columns]
    p = prices[disp].ffill(limit=5)
    ok = [t for t in disp if p[t].notna().sum() >= min_obs]
    info = {'excluidos': [t for t in disp if t not in ok], 'inicio': None, 'limitantes': [],
            'primeros': {}, 'cortos': [], 'min_anios': 0}
    if not ok:
        return p[ok], info
    # Activos con menos historia que el mínimo elegido: quedan fuera del análisis de cartera
    min_anios = float(st.session_state.get('min_hist_anios', 0) or 0)
    if min_anios > 0:
        corte = p.index[-1] - pd.DateOffset(days=int(min_anios * 365.25))
        cortos = [t for t in ok if p[t].first_valid_index() > corte + pd.Timedelta(days=15)]
        if len(cortos) < len(ok) - 1:        # siempre dejar al menos 2 activos
            info['cortos'] = cortos
            info['min_anios'] = min_anios
            ok = [t for t in ok if t not in cortos]
    p = p[ok]
    primeros = p.apply(lambda s: s.first_valid_index())
    inicio = primeros.max()
    umbral = p.index[0] + pd.Timedelta(days=30)
    info['inicio'] = inicio
    info['primeros'] = primeros.to_dict()
    info['limitantes'] = sorted([t for t, d in primeros.items() if d > umbral],
                                key=lambda t: primeros[t], reverse=True)
    p = p.loc[inicio:].ffill().dropna(axis=1, how='any')
    return p, info

def mostrar_aviso_ventana(info):
    if info.get('cortos'):
        _pf = st.session_state.portfolio
        _tot = _pf['Monto_USD'].sum()
        _pct = _pf[_pf['Ticker'].isin(info['cortos'])]['Monto_USD'].sum() / _tot * 100 if _tot > 0 else 0
        st.info(f"⏳ Fuera del análisis de cartera por tener menos de {info['min_anios']:g} años de historia: "
                f"{', '.join(info['cortos'])} (representan {_pct:.1f}% de la renta variable). "
                f"Podés cambiar ese mínimo en el panel lateral.")
    if info.get('excluidos'):
        st.warning(f"⚠️ Sin historia suficiente (excluidos de este cálculo): {', '.join(info['excluidos'])}")
    if info.get('limitantes') and info.get('inicio') is not None:
        ini = info['inicio']
        anios = (pd.Timestamp.today() - ini).days / 365.25
        lim = ", ".join(f"{t} (desde {info['primeros'][t]:%m/%Y})" for t in info['limitantes'][:5])
        st.info(f"📅 Ventana común de análisis: desde **{ini:%d/%m/%Y}** ({anios:.1f} años). "
                f"La limitan los activos con historia más corta: {lim}. "
                f"Las métricas individuales de cada activo (solapa Métricas) usan su historia completa.")

def pesos_para(df_port, tickers):
    """Pesos actuales (suman 1) para la lista de tickers dada."""
    g = df_port.groupby('Ticker')['Peso_Actual_%'].sum()
    w = np.array([float(g.get(t, 0.0)) for t in tickers])
    s = w.sum()
    return w / s if s > 0 else np.ones(len(tickers)) / max(len(tickers), 1)

def serie_cartera(prices, w):
    """Valor de la cartera (base 1) manteniendo constantes los pesos dados."""
    r = prices.pct_change().fillna(0).values @ np.asarray(w)
    return pd.Series(np.cumprod(1 + r), index=prices.index)

def calc_beta(precios_a, precios_b, freq='Semanal'):
    """Beta = Cov(a, b) / Var(b) con retornos en USD sobre fechas comunes.
    Semanal (viernes a viernes) es más robusto para activos poco líquidos."""
    if precios_a is None or precios_b is None:
        return np.nan
    df = pd.concat([precios_a, precios_b], axis=1, keys=['a', 'b']).dropna()
    if str(freq).startswith('Semanal'):
        df = df.resample('W-FRI').last().dropna()
        min_n = 26
    else:
        min_n = 60
    r = df.pct_change().dropna()
    if len(r) < min_n or r['b'].var() == 0:
        return np.nan
    return float(r['a'].cov(r['b']) / r['b'].var())

@st.cache_data(ttl=3600, show_spinner=False)
def metricas_serie(serie, bench, rf=0.02, beta_freq='Semanal'):
    """Métricas completas de una serie de valor (cartera, SPY u óptima)."""
    s = serie.dropna()
    r = s.pct_change().dropna()
    anios = max(len(r) / 252, 0.01)
    ret = r.mean() * 252
    vol = r.std() * np.sqrt(252)
    cagr = (s.iloc[-1] / s.iloc[0]) ** (1 / anios) - 1
    down = np.sqrt((np.minimum(r, 0) ** 2).mean()) * np.sqrt(252)
    mdd = (s / s.cummax() - 1).min()
    var95 = r.quantile(0.05)
    beta = calc_beta(s, bench, beta_freq) if bench is not None else np.nan
    alfa = np.nan
    if bench is not None and pd.notna(beta):
        rb = bench.reindex(s.index).ffill().pct_change().dropna()
        alfa = ret - (rf + beta * (rb.mean() * 252 - rf))
    return {
        'CAGR': cagr, 'Retorno anual': ret, 'Volatilidad': vol,
        'Sharpe': (ret - rf) / vol if vol > 0 else np.nan,
        'Sortino': (ret - rf) / down if down > 0 else np.nan,
        'Calmar': cagr / abs(mdd) if mdd < 0 else np.nan,
        'Beta': beta, 'Alfa': alfa, 'Máx Drawdown': mdd,
        'VaR 1d 95%': -var95, 'CVaR 1d 95%': -r[r <= var95].mean(),
    }

def semaforo(v, verde, amarillo, mayor_mejor=True):
    if pd.isna(v):
        return '⚪'
    if mayor_mejor:
        return '🟢' if v >= verde else ('🟡' if v >= amarillo else '🔴')
    return '🟢' if v <= verde else ('🟡' if v <= amarillo else '🔴')

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

def calc_metrics(prices, bench=None, rf=0.02, beta_freq='Semanal'):
    """Métricas por activo, cada uno con SU historia completa disponible."""
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()
    filas = {}
    for col in prices.columns:
        s = prices[col].dropna()
        if len(s) < 60:
            continue
        r = s.pct_change().dropna()
        anios = max(len(s) / 252, 0.01)
        ret_a = r.mean() * 252
        vol_a = r.std() * np.sqrt(252)
        downside = np.sqrt((np.minimum(r, 0) ** 2).mean()) * np.sqrt(252)
        var1 = -r.quantile(0.05) * 100
        filas[col] = {
            'Historia desde': s.index[0].strftime('%m/%Y'),
            'Retorno Anual %': round(ret_a * 100, 2),
            'CAGR %': round(((s.iloc[-1] / s.iloc[0]) ** (1 / anios) - 1) * 100, 2),
            'Volatilidad %': round(vol_a * 100, 2),
            'Sharpe': round((ret_a - rf) / vol_a, 3) if vol_a > 0 else np.nan,
            'Sortino': round((ret_a - rf) / downside, 3) if downside > 0 else np.nan,
            'Máx Drawdown %': round((s / s.cummax() - 1).min() * 100, 2),
            'Beta SPY': round(calc_beta(s, bench, beta_freq), 2) if bench is not None else np.nan,
            'VaR 1d 95%': round(var1, 2),
            'VaR 10d 95%': round(var1 * np.sqrt(10), 2),
        }
    if not filas:
        raise ValueError("Datos insuficientes para calcular métricas (mínimo 60 días).")
    metrics = pd.DataFrame(filas).T
    num = [c for c in metrics.columns if c != 'Historia desde']
    metrics[num] = metrics[num].astype(float)
    return metrics, prices.pct_change()

def portfolio_metrics(weights_arr, returns_df, rf=0.02):
    # Drop columns with insufficient data, realign weights
    clean = returns_df.dropna(axis=1, thresh=max(30, len(returns_df)//2))
    clean = clean.dropna()
    if clean.empty or len(clean.columns) == 0:
        return np.nan, np.nan, np.nan
    # Realign weights to surviving columns
    if len(weights_arr) != len(clean.columns):
        idx = [i for i, c in enumerate(returns_df.columns) if c in clean.columns]
        weights_arr = np.array([weights_arr[i] for i in idx])
        w_sum = weights_arr.sum()
        if w_sum > 0:
            weights_arr = weights_arr / w_sum
        else:
            weights_arr = np.ones(len(clean.columns)) / len(clean.columns)
    port_ret = np.dot(weights_arr, clean.mean()) * 252
    cov = clean.cov() * 252
    port_vol = np.sqrt(np.dot(weights_arr, np.dot(cov, weights_arr)))
    if port_vol == 0 or np.isnan(port_vol):
        return np.nan, np.nan, np.nan
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
# ─────────────────────────────────────────
#  EXPORTAR RESUMEN (Excel)
# ─────────────────────────────────────────
def _ret_freq(precios, freq):
    p = precios.resample('W-FRI').last() if str(freq).startswith('Semanal') else precios
    return p.pct_change().dropna(how='all')

def generar_resumen_excel(years, rf, beta_freq):
    """Arma un Excel con todo lo calculado, para revisar o compartir."""
    import io
    hist = st.session_state.hist_data
    bench = st.session_state.bench_data
    df_port = calc_portfolio_weights(st.session_state.portfolio.copy())
    tickers = [t for t in df_port['Ticker'] if t in hist.columns]
    hoy = pd.Timestamp.today()

    # ── Métricas con historia completa de cada activo ──
    met_full, _ = calc_metrics(hist[tickers].ffill(limit=5), bench, rf=rf, beta_freq=beta_freq)

    # ── Ventana común (lo que usa la cartera) ──
    pc, info = preparar_precios(hist, tickers)
    en_ventana = list(pc.columns)
    w = pesos_para(df_port, en_ventana)
    serie = serie_cartera(pc, w)
    r_d = serie.pct_change().dropna()
    ret_p, vol_p, sh_p = portfolio_metrics(w, pc.pct_change().dropna(), rf)
    beta_reg = calc_beta(serie, bench, beta_freq)
    mdd = (serie / serie.cummax() - 1).min()

    # retornos a la frecuencia de la beta, para correlación y volatilidad comparables
    r_f = _ret_freq(pd.concat([pc, bench.rename('__SPY__')], axis=1).loc[pc.index[0]:].dropna(subset=['__SPY__']), beta_freq)
    per_anio = 52 if str(beta_freq).startswith('Semanal') else 252
    spy_vol = r_f['__SPY__'].std() * np.sqrt(per_anio)

    cov = pc.pct_change().dropna().cov().values * 252
    var_p = float(w @ cov @ w)
    rc = (w * (cov @ w) / var_p * 100) if var_p > 0 else np.full(len(w), np.nan)

    filas = []
    sector = df_port.set_index('Ticker')['Sector'].to_dict()
    monto = df_port.groupby('Ticker')['Monto_USD'].sum().to_dict()
    peso_rv = df_port.groupby('Ticker')['Peso_Actual_%'].sum().to_dict()
    for t in df_port['Ticker'].unique():
        fila = {'Ticker': t, 'Sector': sector.get(t, ''), 'Monto u$s': round(monto.get(t, 0), 2),
                'Peso en RV %': round(peso_rv.get(t, 0), 2)}
        if t in met_full.index:
            m = met_full.loc[t]
            fila.update({'Historia desde': m['Historia desde'],
                         'Beta (historia completa)': m['Beta SPY'],
                         'Volatilidad % (hist. completa)': m['Volatilidad %'],
                         'Retorno anual % (hist. completa)': m['Retorno Anual %'],
                         'Sharpe (hist. completa)': m['Sharpe'],
                         'Máx Drawdown %': m['Máx Drawdown %']})
        if t in en_ventana:
            i = en_ventana.index(t)
            b = calc_beta(pc[t], bench, beta_freq)
            corr = r_f[t].corr(r_f['__SPY__']) if t in r_f else np.nan
            vol_v = r_f[t].std() * np.sqrt(per_anio) if t in r_f else np.nan
            fila.update({'En análisis de cartera': 'Sí',
                         'Peso en análisis %': round(w[i] * 100, 2),
                         'Beta (ventana común)': round(b, 3),
                         'Correlación con SPY': round(corr, 3),
                         'Volatilidad % (ventana común)': round(vol_v * 100, 2),
                         'Aporte a la beta': round(w[i] * b, 4) if not np.isnan(b) else np.nan,
                         'Aporte al riesgo %': round(rc[i], 2)})
        else:
            if t not in hist.columns or hist[t].notna().sum() < 60:
                motivo = 'Sin datos suficientes en Yahoo'
            else:
                motivo = 'Historia corta'
            fila.update({'En análisis de cartera': f'No ({motivo})'})
        filas.append(fila)
    df_act = pd.DataFrame(filas).sort_values('Monto u$s', ascending=False)

    beta_pond = df_act['Aporte a la beta'].sum(min_count=1) if 'Aporte a la beta' in df_act else np.nan
    fuera = df_act[df_act['En análisis de cartera'] != 'Sí']
    resumen = [
        ('Fecha del resumen', hoy.strftime('%d/%m/%Y %H:%M')),
        ('— CONFIGURACIÓN —', ''),
        ('Años de historia', years),
        ('Tasa libre de riesgo %', rf * 100),
        ('Frecuencia de la beta', beta_freq),
        ('Historia mínima para análisis de cartera (años)', st.session_state.get('min_hist_anios', 0)),
        ('Benchmark', 'SPY'),
        ('Tipo de cambio usado', f"{st.session_state.get('balanz_tc_fuente', '')} ${st.session_state.get('balanz_tipo_cambio', 0):,.2f}"),
        ('— RENTA VARIABLE —', ''),
        ('Total renta variable u$s', round(df_port['Monto_USD'].sum(), 2)),
        ('Activos en renta variable', df_port['Ticker'].nunique()),
        ('Activos en análisis de cartera', len(en_ventana)),
        ('Peso fuera del análisis %', round(fuera['Peso en RV %'].sum(), 2)),
        ('Fuera del análisis', ', '.join(fuera['Ticker'])),
        ('Ventana común desde', info['inicio'].strftime('%d/%m/%Y') if info.get('inicio') is not None else ''),
        ('— CARTERA (ventana común, pesos actuales) —', ''),
        ('Retorno anual %', round(ret_p * 100, 2)),
        ('Volatilidad anual %', round(vol_p * 100, 2)),
        ('Sharpe', round(sh_p, 3)),
        ('Máx Drawdown %', round(mdd * 100, 2)),
        ('VaR 1d 95% (histórico) %', round(-r_d.quantile(0.05) * 100, 2)),
        ('Beta por regresión (serie de la cartera vs SPY)', round(beta_reg, 3)),
        ('Beta ponderada (suma de peso × beta de cada activo)', round(beta_pond, 3) if pd.notna(beta_pond) else ''),
        ('Volatilidad SPY en la ventana %', round(spy_vol * 100, 2)),
    ]
    df_res = pd.DataFrame(resumen, columns=['Concepto', 'Valor'])

    # ── Cartera total (lo importado de Balanz) ──
    hojas_total = {}
    bz = st.session_state.get('balanz_data')
    if bz is not None and len(bz):
        tc = float(st.session_state.get('balanz_tipo_cambio', 1) or 1)
        bz = bz.copy()
        bz['u$s'] = np.where(bz['Es_USD_Real'], bz['V_Actual_Num'], bz['V_Actual_Num'] / tc).round(2)
        bz['Leído en'] = np.where(bz['Es_USD_Real'], 'USD', 'ARS')
        tot = bz['u$s'].sum()
        bz['% cartera total'] = (bz['u$s'] / tot * 100).round(2)
        hojas_total['Cartera total'] = bz[['Ticker', 'Descripcion', 'Instrumento', 'Renta', 'Moneda', 'Pais',
                                           'Sector_Macro', 'Leído en', 'V_Actual_Num', 'u$s', '% cartera total']]
        agr = []
        for col in ['Renta', 'Moneda', 'Instrumento', 'Pais', 'Sector_Macro']:
            g = bz.groupby(col)['u$s'].sum().sort_values(ascending=False)
            for k, v in g.items():
                agr.append({'Agrupación': col, 'Grupo': k, 'u$s': round(v, 2), '%': round(v / tot * 100, 2)})
        hojas_total['Agrupaciones'] = pd.DataFrame(agr)

    # precios semanales en USD, para poder verificar cálculos
    px_sem = pd.concat([hist[tickers], bench.rename('SPY (benchmark)')], axis=1).resample('W-FRI').last()
    px_sem.index = px_sem.index.strftime('%Y-%m-%d')

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as xw:
        df_res.to_excel(xw, sheet_name='Resumen', index=False)
        df_act.to_excel(xw, sheet_name='Renta variable', index=False)
        for nombre, d in hojas_total.items():
            d.to_excel(xw, sheet_name=nombre, index=False)
        px_sem.to_excel(xw, sheet_name='Precios semanales USD')
        for ws in xw.book.worksheets:
            for col in ws.columns:
                largo = max(len(str(c.value)) if c.value is not None else 0 for c in col[:200])
                ws.column_dimensions[col[0].column_letter].width = min(max(10, largo + 2), 55)
    return buf.getvalue()

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

    st.session_state['min_hist_anios'] = st.slider(
        "Historia mínima para el análisis de cartera (años)", min_value=0.0, max_value=10.0,
        value=2.0, step=0.5,
        help="Los activos más nuevos que esto quedan fuera de los cálculos de cartera (correlación, "
             "VaR, frontera, rendimiento) para no recortar la ventana de todos. "
             "Igual aparecen en Métricas con su propia historia. 0 = incluir todos.")

    beta_freq = st.selectbox("Frecuencia para calcular Beta", ["Semanal", "Diaria"], index=0,
        help="Semanal (viernes a viernes) es el estándar para activos poco líquidos como "
             "las acciones argentinas sin ADR: evita que los días sin operaciones subestimen la beta.")

    st.markdown("---")
    if st.button("🚀 Cargar datos de mercado", type="primary", use_container_width=True):
        # Normalizar tickers a formato Yahoo (ej: BRKB → BRK-B) y analizar solo posiciones con monto > 0
        _port = st.session_state.portfolio.copy()
        _port['Ticker'] = _port['Ticker'].map(yf_ticker)
        st.session_state.portfolio = _port
        all_tickers = _port.loc[_port['Monto_USD'] > 0, 'Ticker'].tolist()
        # Filtrar tickers que YF no puede resolver:
        # - Bonos argentinos (GD, AL, AE, BP prefijos + números)
        # - FCIs (BCMMUSDA y similares largos)
        # - Bitcoin (BTC solo)
        # - Efectivo/Cash
        import re
        def es_ticker_yf_valido(t):
            t = str(t).strip()
            if t in ('BCMMUSDA', 'Dolares Cash', 'Pesos Cash', 'Plazo fijo'):
                return False
            # Bonos argentinos: GD35, GD38, GD41, AE38, AL30, BPOA8, etc.
            if re.match(r'^(GD|AL|AE|BP|TX|TV|DICP|CUAP)[0-9]', t):
                return False
            # Tickers muy largos sin punto (FCIs) — excepto BTC-USD
            if t == 'BTC-USD':
                return True
            if len(t) > 6 and '.' not in t and not t.endswith('.BA') and '-' not in t:
                return False
            return True

        tickers_yf = [t for t in all_tickers if es_ticker_yf_valido(t)]
        excluidos  = [t for t in all_tickers if not es_ticker_yf_valido(t)]

        if tickers_yf:
            with st.spinner(f"Descargando {len(tickers_yf)} tickers de Yahoo Finance..."):
                _precios, _bench, _notas = fetch_data(tickers_yf, PERIOD)
            st.session_state.data_notas = _notas
            if _precios is None:
                st.session_state.hist_data = None
                st.session_state.bench_data = None
                st.error(" ".join(_notas))
            else:
                st.session_state.hist_data = _precios
                st.session_state.bench_data = _bench
                st.session_state.tickers_loaded = list(_precios.columns)
                st.success(f"✅ Datos cargados — {_precios.shape[1]} tickers (en USD, calendario EEUU)")
                for _n in _notas:
                    st.info(_n)
            if excluidos:
                st.info(f"ℹ️ Excluidos del análisis histórico (no disponibles en YF): {', '.join(excluidos)}")
        else:
            st.warning("Primero cargá tu cartera")

    if st.session_state.get('hist_data') is not None:
        st.markdown("---")
        st.markdown("### 📤 Exportar resumen")
        if st.button("Preparar resumen", use_container_width=True,
                     help="Genera un Excel con la configuración, las métricas de la cartera, cada activo "
                          "(beta, correlación, aporte al riesgo), la cartera total y los precios usados."):
            with st.spinner("Armando el resumen..."):
                try:
                    st.session_state.resumen_xlsx = generar_resumen_excel(years, rf_rate, beta_freq)
                except Exception as e:
                    st.session_state.resumen_xlsx = None
                    st.error(f"No se pudo armar el resumen: {e}")
        if st.session_state.get('resumen_xlsx'):
            st.download_button("⬇️ Descargar resumen (Excel)", data=st.session_state.resumen_xlsx,
                file_name=f"resumen_cartera_{pd.Timestamp.today():%Y-%m-%d}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True)

    st.markdown("---")
    st.caption("📊 Optimizador de Cartera v4.4\nDatos: Yahoo Finance")

# ─────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────
st.markdown("# 📊 Optimizador de Cartera")
st.markdown("*Cartera total · Renta variable · Riesgo · Optimización — datos reales de Yahoo Finance*")
st.markdown("---")

# ─────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────
T = st.tabs([
    "🏠 Cartera total",
    "🧩 Renta variable",
    "📐 Riesgo",
    "🎯 Optimización",
    "📥 Importar",
    "⚙️ Datos",
    "📖 Guía",
])
with T[1]:
    _rv = st.tabs(["🏷️ Por sector", "🎯 Targets", "⚖️ Rebalanceo"])
with T[2]:
    _rk = st.tabs(["📐 Métricas", "🔗 Correlación", "📈 Rendimiento", "⚠️ VaR", "🔥 Stress test"])
with T[3]:
    _op = st.tabs(["🎚️ Simulador de beta", "🌐 Frontera eficiente", "🎲 Monte Carlo"])
with T[5]:
    _dt = st.tabs(["📁 Cartera (edición manual)", "📋 Catálogo de instrumentos", "🏷️ Sectores"])

# Mapeo de las secciones originales a su nueva ubicación
tabs = {
    0: T[0],                       # Tablero macro → Cartera total
    1: T[4],                       # Importar
    2: _dt[1],                     # Catálogo de instrumentos
    3: _dt[0],                     # Mi cartera (edición manual)
    4: _rv[0],                     # Sectores
    5: _rv[2],                     # Rebalanceo
    6: _rv[1],                     # Estrategia / targets
    7: _rk[1],                     # Correlación
    8: _rk[0],                     # Métricas
    9: _rk[2],                     # Rendimiento
    10: _rk[3],                    # VaR
    11: _rk[4],                    # Stress test
    12: _op[1],                    # Frontera eficiente
    14: _op[2],                    # Monte Carlo
    15: _op[0],                    # Simulador de beta por sector
}


# ══════════════════════════════════════════
#  TAB 0 — TABLERO MACRO
# ══════════════════════════════════════════
with tabs[0]:
    st.subheader("📊 Tablero Macro")

    if st.session_state.balanz_data is None:
        st.info("📥 Primero importá tu cartera desde la solapa **📥 Importar**")
    else:
        df_b = st.session_state.balanz_data.copy()
        total_pesos = df_b['V_Actual_Pesos'].sum()

        if total_pesos == 0:
            st.error("Sin datos válidos")
        else:
            # ── KPIs principales ──
            st.markdown("### 💼 Cartera total")
            tc = st.session_state.balanz_tipo_cambio
            total_usd = total_pesos / tc
            k1,k2,k3 = st.columns(3)
            k1.metric("Total en Pesos", f"${total_pesos:,.0f}")
            k2.metric("Total en USD (aprox)", f"u$s {total_usd:,.0f}")
            k3.metric("Tipo de cambio usado", f"${tc:,.0f}")

            st.markdown("---")

            # ─────────────────────────────
            # BLOQUE 1 — MONEDA
            # ─────────────────────────────
            st.markdown("### 💱 Por Moneda")
            mon_group = df_b.groupby('Moneda_Display')['V_Actual_Pesos'].sum()
            mon_pct   = (mon_group / total_pesos * 100).round(1)

            col_m1, col_m2 = st.columns([1,1])
            with col_m1:
                fig_mon = go.Figure(go.Pie(
                    labels=mon_group.index, values=mon_group.values,
                    hole=0.45,
                    marker_colors=['#00d4ff','#f87171'],
                    textinfo='label+percent',
                    textfont=dict(color='#e2e8f0', size=13)
                ))
                fig_mon.update_layout(
                    paper_bgcolor='#111827', font_color='#e2e8f0',
                    title_font_color='#00d4ff', title_font_size=14,
                    title='Distribución por moneda',
                    showlegend=False, height=300, margin=dict(t=40,b=10)
                )
                st.plotly_chart(fig_mon, use_container_width=True)

            with col_m2:
                for mon, pct in mon_pct.items():
                    val = mon_group[mon]
                    st.metric(f"{'💵' if 'USD' in mon or 'Dólar' in mon else '🇦🇷'} {mon}",
                              f"{pct:.1f}%", f"${val:,.0f}")

            st.markdown("---")

            # ─────────────────────────────
            # BLOQUE 2 — TIPO DE RENTA con desglose moneda
            # ─────────────────────────────
            st.markdown("### 📈 Por Tipo de Renta")

            renta_types = ['Variable', 'Fija', 'Liquidez']
            renta_colors = {'Variable':'#00d4ff', 'Fija':'#4ade80', 'Liquidez':'#fbbf24'}

            # Normalize renta values
            def norm_renta(r):
                if r is None: return 'Otro'
                r = str(r).strip()
                if 'ariable' in r: return 'Variable'
                if 'ija' in r or 'ija' in r: return 'Fija'
                if 'iquidez' in r or 'iquid' in r or 'Plazo Fijo' in r: return 'Liquidez'
                return 'Otro'

            df_b['Renta_Norm'] = df_b['Renta'].apply(norm_renta)
            renta_group = df_b.groupby('Renta_Norm')['V_Actual_Pesos'].sum()
            renta_pct   = (renta_group / total_pesos * 100).round(1)

            col_r1, col_r2 = st.columns([1.2, 1])
            with col_r1:
                fig_renta = go.Figure(go.Bar(
                    x=list(renta_pct.index),
                    y=list(renta_pct.values),
                    marker_color=[renta_colors.get(r,'#94a3b8') for r in renta_pct.index],
                    text=[f"{v:.1f}%" for v in renta_pct.values],
                    textposition='outside', textfont=dict(color='#e2e8f0')
                ))
                fig_renta.update_layout(
                    paper_bgcolor='#111827', plot_bgcolor='#111827',
                    font_color='#e2e8f0', title_font_color='#00d4ff', title_font_size=14,
                    title='% sobre cartera total', yaxis_title='%',
                    height=300, margin=dict(t=40,b=10), showlegend=False
                )
                st.plotly_chart(fig_renta, use_container_width=True)

            with col_r2:
                for renta in renta_group.index:
                    val     = renta_group[renta]
                    pct_tot = renta_pct[renta]
                    df_r    = df_b[df_b['Renta_Norm']==renta]

                    # Desglose por moneda dentro de este tipo de renta
                    mon_dentro = df_r.groupby('Moneda_Display')['V_Actual_Pesos'].sum()
                    total_r    = val

                    st.markdown(f"**{renta_colors.get(renta,'⬜')} {renta} — {pct_tot:.1f}% del total**")
                    for mon, mval in mon_dentro.items():
                        pct_dentro = mval / total_r * 100 if total_r > 0 else 0
                        pct_global = mval / total_pesos * 100
                        icon = '💵' if 'USD' in mon or 'Dólar' in mon else '🇦🇷'
                        st.markdown(f"&nbsp;&nbsp;&nbsp;{icon} {mon}: **{pct_dentro:.1f}%** de {renta} ({pct_global:.1f}% del total)")
                    st.markdown("---")

            # ─────────────────────────────
            # BLOQUE 3 — SECTOR MACRO (solo renta variable)
            # ─────────────────────────────
            st.markdown("### 🏭 Sectores MACRO (Renta Variable)")
            df_rv = df_b[df_b['Renta_Norm']=='Variable'].copy()
            if df_rv.empty:
                st.info("Sin instrumentos de renta variable")
            else:
                total_rv = df_rv['V_Actual_Pesos'].sum()
                sec_group = df_rv.groupby('Sector_Macro')['V_Actual_Pesos'].sum().sort_values(ascending=False)
                sec_pct_rv    = (sec_group / total_rv * 100).round(1)
                sec_pct_total = (sec_group / total_pesos * 100).round(1)

                col_s1, col_s2 = st.columns([1.2,1])
                with col_s1:
                    fig_sec = go.Figure(go.Bar(
                        y=list(sec_pct_rv.index),
                        x=list(sec_pct_rv.values),
                        orientation='h',
                        marker_color='rgba(0,212,255,0.7)',
                        text=[f"{v:.1f}%" for v in sec_pct_rv.values],
                        textposition='outside', textfont=dict(color='#e2e8f0')
                    ))
                    fig_sec.update_layout(
                        paper_bgcolor='#111827', plot_bgcolor='#111827',
                        font_color='#e2e8f0', title_font_color='#00d4ff', title_font_size=14,
                        title='% sobre Renta Variable', height=350,
                        margin=dict(t=40,b=10), showlegend=False
                    )
                    st.plotly_chart(fig_sec, use_container_width=True)

                with col_s2:
                    for sec in sec_group.index:
                        pct_rv  = sec_pct_rv[sec]
                        pct_tot = sec_pct_total[sec]
                        st.markdown(f"**{sec}**: {pct_rv:.1f}% de RV ({pct_tot:.1f}% del total)")

            st.markdown("---")

            # ─────────────────────────────
            # BLOQUE 4 — TABLA DETALLE
            # ─────────────────────────────
            st.markdown("### 📋 Detalle por instrumento")
            df_show = df_b[['Ticker','Descripcion','Renta_Norm','Sector_Macro','Sector_Detalle',
                            'Moneda_Display','V_Actual_Pesos']].copy()
            df_show['% Cartera'] = (df_show['V_Actual_Pesos'] / total_pesos * 100).round(2)
            df_show = df_show.sort_values('V_Actual_Pesos', ascending=False)
            df_show.columns = ['Ticker','Descripción','Renta','Sector MACRO','Sector Detalle',
                               'Moneda','V. Actual ($)','% Cartera']
            st.dataframe(df_show, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════
#  TAB 1 — IMPORTAR BALANZ
# ══════════════════════════════════════════
with tabs[1]:
    st.subheader("📥 Importar desde Balanz")
    st.markdown("Copiá y pegá la info desde Balanz a tu Excel y subilo acá. La app lee la solapa **'Actualizar de Balanz'** automáticamente.")

    col_up, col_tc = st.columns([2,1])
    with col_up:
        uploaded_balanz = st.file_uploader(
            "Subí tu archivo Excel de Balanz (.xlsm o .xlsx)",
            type=['xlsm','xlsx','xls'],
            key="balanz_uploader"
        )
    with col_tc:
        tc_tipo = st.selectbox("Tipo de cambio", list(TC_FUENTES.keys()) + ["Manual"], key="tc_tipo",
            help="Se usa para pasar a pesos los instrumentos valuados en dólares y para expresar la cartera en USD. "
                 "MEP es el dólar que obtenés operando en tu cuenta de Balanz. Se toma el valor de VENTA.")
        tc_input = None
        if tc_tipo != "Manual":
            _casa, _nombre = TC_FUENTES[tc_tipo]
            _tc = obtener_tc(_casa)
            if _tc:
                tc_input = _tc['venta']
                _f = f"{_tc['fecha']:%d/%m/%Y %H:%M}" if pd.notna(_tc['fecha']) else "s/d"
                st.metric(f"{_nombre} (venta)", f"${tc_input:,.2f}")
                st.caption(f"Fuente: dolarapi.com · actualizado {_f}")
            else:
                st.warning("No se pudo obtener la cotización automática. Ingresala a mano.")
        if tc_input is None:
            tc_input = st.number_input(
                "Tipo de cambio manual ($ por u$s)",
                min_value=1.0, value=float(st.session_state.balanz_tipo_cambio),
                step=1.0, help="Usalo si querés valuar con el TC de una fecha puntual (ej: la del Excel).")
        st.session_state.balanz_tipo_cambio = float(tc_input)
        st.session_state.balanz_tc_fuente = tc_tipo

    if uploaded_balanz:
        try:
            import openpyxl
            wb = openpyxl.load_workbook(uploaded_balanz, read_only=True, keep_vba=True, data_only=True)

            # Find the Balanz sheet
            sheet_name = None
            for name in wb.sheetnames:
                if 'balanz' in name.lower() or 'actualiz' in name.lower():
                    sheet_name = name
                    break
            if not sheet_name:
                st.error("No se encontró la solapa 'Actualizar de Balanz' en el archivo")
            else:
                ws = wb[sheet_name]
                rows = list(ws.iter_rows(values_only=True))

                # Find header row
                header_row = None
                for i, row in enumerate(rows[:10]):
                    if row and 'Ticker' in str(row):
                        header_row = i
                        break

                if header_row is None:
                    st.error("No se encontró el encabezado con 'Ticker' en la solapa")
                else:
                    headers = [str(h).strip() if h else '' for h in rows[header_row]]

                    # Map columns
                    col_map = {}
                    for i, h in enumerate(headers):
                        hl = h.lower()
                        if 'ticker' in hl: col_map['Ticker'] = i
                        elif 'target' in hl and 'sector' in hl: col_map['Target_Sector'] = i  # ANTES que sector macro
                        elif 'v. actual' in hl or 'v.actual' in hl or ('actual' in hl and 'valor' not in hl):
                            col_map.setdefault('V_Actual', i)   # la primera columna "V. Actual" (la de Balanz) manda
                        elif 'descripci' in hl: col_map['Descripcion'] = i
                        elif hl == 'instrumento': col_map['Instrumento'] = i
                        elif 'moneda' in hl: col_map['Moneda'] = i
                        elif 'sector macro' in hl or 'sector_macro' in hl or ('sector' in hl and 'macro' in hl): col_map['Sector_Macro'] = i
                        elif 'sector' in hl and 'macro' not in hl and 'target' not in hl: col_map['Sector_Detalle'] = i
                        elif 'renta' in hl and 'rendimiento' not in hl: col_map['Renta'] = i
                        elif 'pa' in hl and 'is' in hl: col_map['Pais'] = i

                    # Also check for "V. Actual" specifically (col index 8 based on our reading)
                    if 'V_Actual' not in col_map:
                        for i, h in enumerate(headers):
                            if 'actual' in h.lower():
                                col_map['V_Actual'] = i
                                break

                    st.info(f"📋 Solapa encontrada: **{sheet_name}** | Columnas mapeadas: {list(col_map.keys())}")

                    # ── Qué tipos de instrumento vienen en PESOS (todo lo demás: DÓLARES) ──
                    _idx_inst = col_map.get('Instrumento', 16)
                    tipos_archivo = sorted({str(r[_idx_inst]).strip() for r in rows[header_row+1:]
                                            if r and len(r) > _idx_inst and r[_idx_inst] not in (None, '')
                                            and str(r[_idx_inst]).strip() not in ('#N/A', 'None', 'nan')})
                    # Por defecto: todo tipo que diga "banco" va en pesos
                    if 'tipos_en_pesos' in st.session_state:
                        _default_pesos = [t for t in st.session_state.tipos_en_pesos if t in tipos_archivo]
                    else:
                        _default_pesos = [t for t in tipos_archivo if 'banco' in t.lower()]
                    tipos_en_pesos = st.multiselect(
                        "💱 Tipos de instrumento cargados en PESOS (todo lo demás se toma en dólares)",
                        options=tipos_archivo,
                        default=_default_pesos,
                        help="Por defecto todos los montos están en dólares. Elegí acá solo los tipos que cargás "
                             "en pesos (ej: FCI Bancos, Liquidez Bancos). Los tipos salen de la columna "
                             "'Instrumento' de tu Excel.")
                    st.session_state.tipos_en_pesos = tipos_en_pesos

                    # Parse data rows
                    data_rows = []
                    filas_ilegibles = []
                    for row in rows[header_row+1:]:
                        if not row or not any(v is not None for v in row):
                            continue
                        ticker = str(row[col_map.get('Ticker', 0)] or '').strip()
                        if not ticker or ticker in ['#N/A','None','nan','']:
                            continue
                        # Filas de títulos/totales que vienen al pegar desde Balanz
                        if ticker.lower() in ('total', 'totales', 'pesos', 'dolares', 'dólares', 'ticker'):
                            continue

                        # V.Actual: detectar si viene como texto "u$s XXXX" (en dólares reales)
                        # o como número (ya en pesos, Balanz lo convirtió)
                        v_actual_raw = row[col_map.get('V_Actual', 8)]
                        v_actual_str = str(v_actual_raw).strip()
                        v_actual_num, moneda_texto = leer_valor(v_actual_raw)
                        if v_actual_num is None:
                            if v_actual_str not in ('', 'None', '-', 'nan', '#N/A'):
                                filas_ilegibles.append(f"{ticker} ('{v_actual_str}')")
                            continue
                        tipo_inst = str(row[_idx_inst] or '').strip() if len(row) > _idx_inst else ''
                        # Regla: por defecto DÓLARES. Pesos si el tipo está marcado como "en pesos".
                        # Si el valor trae explícito u$s o $, manda eso.
                        if moneda_texto:
                            es_usd_real = (moneda_texto == 'USD')
                        else:
                            es_usd_real = tipo_inst not in tipos_en_pesos
                        if v_actual_num <= 0:
                            continue

                        # Moneda del instrumento (naturaleza) directo del Excel
                        moneda_excel = str(row[col_map.get('Moneda', 22)] or '').strip()

                        # Target sector desde columna Excel
                        try:
                            target_sector_val = float(row[col_map.get('Target_Sector', 26)] or 0)
                        except:
                            target_sector_val = 0.0

                        data_rows.append({
                            'Ticker':         ticker,
                            'Descripcion':    str(row[col_map.get('Descripcion',1)] or '').strip(),
                            'V_Actual_Num':   v_actual_num,
                            'Es_USD_Real':    es_usd_real,
                            'Moneda':         moneda_excel,
                            'Sector_Macro':   str(row[col_map.get('Sector_Macro',25)] or '').strip(),
                            'Sector_Detalle': str(row[col_map.get('Sector_Detalle',24)] or '').strip(),
                            'Renta':          str(row[col_map.get('Renta',18)] or '').strip(),
                            'Pais':           str(row[col_map.get('Pais',23)] or '').strip(),
                            'Instrumento':    tipo_inst,
                            'Target_Sector':  target_sector_val,  # % como decimal (0.15 = 15%)
                        })

                    if filas_ilegibles:
                        st.error("❌ No pude leer el valor de estos instrumentos, **no se están contando**. "
                                 "Revisá cómo están escritos en la columna V. Actual: " + ", ".join(filas_ilegibles))
                    if not data_rows:
                        st.error("No se encontraron datos válidos en la solapa")
                    else:
                        df_balanz = pd.DataFrame(data_rows)

                        # Convertir a pesos: si Es_USD_Real → × tipo de cambio; si no → ya está en pesos
                        df_balanz['V_Actual_Pesos'] = df_balanz.apply(
                            lambda r: r['V_Actual_Num'] * tc_input if r['Es_USD_Real'] else r['V_Actual_Num'],
                            axis=1
                        )
                        # Moneda_Display: la naturaleza del instrumento (columna Moneda del Excel)
                        df_balanz['Moneda_Display'] = df_balanz['Moneda'].apply(
                            lambda m: 'Dólares (USD)' if 'olar' in str(m) else 'Pesos (ARS)'
                        )

                        usd_reales = df_balanz[df_balanz['Es_USD_Real']]['Ticker'].tolist()
                        total = df_balanz['V_Actual_Pesos'].sum()

                        total_usd = total / tc_input
                        # Control de cordura: un valor leído en dólares demasiado grande probablemente esté en pesos
                        _sospechosos = df_balanz[df_balanz['Es_USD_Real'] & (df_balanz['V_Actual_Num'] > 1_000_000)]
                        if not _sospechosos.empty:
                            st.error("🚨 Estos valores se leyeron en **dólares** pero superan u$s 1.000.000 cada uno. "
                                     "¿No estarán en pesos? Revisá si pegaste la vista en pesos de Balanz o si el tipo "
                                     "debería estar marcado como 'en pesos': "
                                     + ", ".join(f"{r.Ticker} ({r.V_Actual_Num:,.0f})" for r in _sospechosos.itertuples()))
                        st.success(f"✅ {len(df_balanz)} instrumentos leídos — Total: **${total:,.0f}** "
                                   f"≈ **u$s {total_usd:,.0f}**")
                        if usd_reales:
                            st.info(f"💵 Instrumentos valuados en dólares (convertidos con {st.session_state.get('balanz_tc_fuente','TC')}: ${tc_input:,.2f}): **{', '.join(usd_reales)}**")

                        # Preview table
                        df_prev = df_balanz[['Ticker','Descripcion','Instrumento','Es_USD_Real','V_Actual_Num','V_Actual_Pesos']].copy()
                        df_prev['Es_USD_Real'] = df_prev['Es_USD_Real'].map({True: 'USD', False: 'ARS'})
                        df_prev['V.Actual en USD'] = (df_prev['V_Actual_Pesos'] / tc_input).round(0)
                        df_prev.columns = ['Ticker','Descripción','Tipo','Leído en','Valor leído','V.Actual en Pesos','V.Actual en USD']
                        st.caption("**Leído en** muestra en qué moneda interpretó la app cada valor: por defecto dólares; "
                                   "pesos solo para los tipos elegidos arriba.")
                        st.dataframe(df_prev, use_container_width=True, hide_index=True)

                        st.markdown("---")
                        if st.button("✅ Confirmar e importar cartera", type="primary"):
                            st.session_state.balanz_data        = df_balanz
                            st.session_state.balanz_usd_tickers = usd_reales

                            # ── Leer targets de sector desde "Sector Macro - Target" ──
                            sector_targets_excel = {}
                            try:
                                wb_targets = openpyxl.load_workbook(uploaded_balanz, read_only=True, keep_vba=True, data_only=True)
                                if 'Sector Macro - Target' in wb_targets.sheetnames:
                                    ws_t = wb_targets['Sector Macro - Target']
                                    for trow in ws_t.iter_rows(values_only=True):
                                        if trow[1] and trow[2] is not None:
                                            sector_name = str(trow[1]).strip()
                                            target_val  = float(trow[2])
                                            if sector_name and sector_name != 'Sector Macro' and sector_name != 'Total':
                                                sector_targets_excel[sector_name] = target_val
                            except:
                                pass

                            # Tipos que NO van al análisis cuantitativo (YF)
                            EXCLUIR_YF = {'Bonos en U$', 'FCI Money Market', 'Plazo Fijo',
                                          'U$ efectivo', '$ Efectivo'}
                            # Tipos que NO van al peso de cartera (excluir de RV)
                            EXCLUIR_PESO = {'Bonos en U$', 'FCI Money Market', 'Plazo Fijo',
                                            'U$ efectivo', '$ Efectivo'}

                            # ── Solo renta variable para cartera de análisis ──
                            df_rv = df_balanz[df_balanz['Renta'].str.strip() == 'Variable'].copy()
                            # Excluir bonos/FCIs del cálculo de pesos pero mantener Bitcoin
                            df_rv_peso = df_rv[~df_rv['Instrumento'].isin(EXCLUIR_PESO)]
                            total_rv_pesos = df_rv_peso['V_Actual_Pesos'].sum()

                            portfolio_rows = []
                            inst_rows      = []
                            for _, irow in df_rv_peso.iterrows():
                                instrumento = str(irow.get('Instrumento', '')).strip()
                                ticker_orig = irow['Ticker']

                                # Mapeo explícito: Balanz ticker → YF ticker
                                # ADRs que cotizan en NYSE en USD (sin .BA)
                                TICKER_MAP = {
                                    'YPFD':  'YPF',      # ADR NYSE
                                    'PAMP':  'PAM',      # ADR NYSE
                                    'TGSU2': 'TGS',      # ADR NYSE
                                    'BMA':   'BMA',      # ADR NYSE
                                    'GGAL':  'GGAL',     # ADR NYSE
                                    'CEPU':  'CEPU',     # ADR NYSE
                                    'EDN':   'EDN',      # ADR NYSE
                                    'VIST':  'VIST',     # ADR NYSE
                                    # Solo panel BYMA en ARS → necesitan conversión TC
                                    'METR':  'METR.BA',
                                    'TGNO4': 'TGNO4.BA',
                                }
                                # Tickers que cotizan en ARS → hay que convertir retornos a USD
                                TICKERS_ARS = {'METR.BA', 'TGNO4.BA'}

                                ticker_yf = ticker_yf_balanz(ticker_orig, instrumento)

                                monto_usd  = round(irow['V_Actual_Pesos'] / tc_input, 2)
                                target_pct = round(irow['V_Actual_Pesos'] / total_rv_pesos * 100, 2) if total_rv_pesos > 0 else 0
                                sector = str(irow['Sector_Macro']).strip()
                                if not sector or sector in ['0','None','nan']:
                                    sector = str(irow['Sector_Detalle']).strip()
                                if not sector or sector in ['0','None','nan']:
                                    sector = 'Otro'

                                portfolio_rows.append({
                                    'Ticker':    ticker_yf,
                                    'Monto_USD': monto_usd,
                                    'Target_%':  target_pct,
                                    'Sector':    sector,
                                })
                                inst_rows.append({'Ticker': ticker_yf, 'Sector': sector})

                            # Un mismo ticker en dos cuentas (ej: Balanz + Wallbit) → una sola posición sumada
                            _pf = pd.DataFrame(portfolio_rows)
                            if not _pf.empty:
                                _pf = (_pf.groupby('Ticker', as_index=False, sort=False)
                                          .agg({'Monto_USD': 'sum', 'Target_%': 'sum', 'Sector': 'first'}))
                                _pf[['Monto_USD', 'Target_%']] = _pf[['Monto_USD', 'Target_%']].round(2)
                            st.session_state.portfolio    = _pf
                            st.session_state.instruments  = pd.DataFrame(inst_rows).drop_duplicates('Ticker')

                            # ── Poblar Sectores (solo de renta variable) ──
                            sectores_limpios = []
                            for s in df_rv['Sector_Macro'].dropna().unique():
                                s = str(s).strip()
                                if s and s not in ['0','None','nan']:
                                    sectores_limpios.append(s)
                            for _, irow in df_rv.iterrows():
                                if str(irow['Sector_Macro']).strip() in ['0','None','nan','']:
                                    sd = str(irow['Sector_Detalle']).strip()
                                    if sd and sd not in ['0','None','nan',''] and sd not in sectores_limpios:
                                        sectores_limpios.append(sd)
                            if 'Otro' not in sectores_limpios:
                                sectores_limpios.append('Otro')

                            # ── Leer targets de sector desde solapa "Sector Macro - Target" ──
                            sector_targets_excel = {}
                            try:
                                uploaded_balanz.seek(0)
                                wb_t = openpyxl.load_workbook(uploaded_balanz, read_only=True, keep_vba=True, data_only=True)
                                if 'Sector Macro - Target' in wb_t.sheetnames:
                                    for trow in wb_t['Sector Macro - Target'].iter_rows(values_only=True):
                                        if trow[1] and trow[2] is not None:
                                            sname = str(trow[1]).strip()
                                            if sname and sname not in ('Sector Macro', 'Total'):
                                                sector_targets_excel[sname] = float(trow[2])
                                                # Agregar al listado de sectores si no está
                                                if sname not in sectores_limpios:
                                                    sectores_limpios.append(sname)
                            except Exception as e_t:
                                st.warning(f"No se pudo leer 'Sector Macro - Target': {e_t}")

                            st.session_state.sectors = sectores_limpios

                            if sector_targets_excel:
                                st.session_state.sector_targets = sector_targets_excel
                                st.info(f"✅ Targets de sector importados desde Excel: {len(sector_targets_excel)} sectores")

                            save_persistent(
                                st.session_state.portfolio,
                                st.session_state.instruments,
                                st.session_state.sectors,
                                st.session_state.sector_targets
                            )

                            total = df_balanz['V_Actual_Pesos'].sum()
                            st.success(f"✅ Cartera importada — {len(df_balanz)} instrumentos — Total: ${total:,.0f}")
                            st.info("👆 Andá al **📊 Tablero Macro** para el análisis, o a **⚖️ Rebalanceo** para ver los ajustes")
                            st.rerun()

        except Exception as e:
            st.error(f"Error al leer el archivo: {e}")
            import traceback
            st.code(traceback.format_exc())

    elif st.session_state.balanz_data is not None:
        df_b = st.session_state.balanz_data
        st.success(f"✅ Cartera importada — {len(df_b)} instrumentos — Total: ${df_b['V_Actual_Pesos'].sum():,.0f}")
        st.markdown(f"**Instrumentos con V.Actual en USD:** {', '.join(st.session_state.balanz_usd_tickers) or 'ninguno'}")
        if st.button("🗑️ Limpiar datos importados"):
            st.session_state.balanz_data = None
            st.rerun()


# ══════════════════════════════════════════
#  TAB 0 — INSTRUMENTOS (catálogo maestro)
# ══════════════════════════════════════════
with tabs[2]:
    st.subheader("📋 Catálogo de Instrumentos")
    st.markdown("Acá cargás tus instrumentos **una sola vez**. Quedan guardados y disponibles para armar tu cartera.")

    # ── Agregar instrumento al catálogo ──
    st.markdown("#### Agregar instrumento al catálogo")
    with st.form("add_instrument_form"):
        ci1, ci2, ci3 = st.columns([1.5, 2, 1])
        inst_ticker = ci1.text_input("Ticker", placeholder="AAPL").upper().strip()
        inst_sector = ci2.selectbox("Sector", [""] + st.session_state.sectors)
        inst_submit = ci3.form_submit_button("➕ Agregar", type="primary")

    if inst_submit and inst_ticker:
        import re
        if not re.match(r'^[A-Z0-9.\-]{1,10}$', inst_ticker):
            st.error("⚠️ Formato de ticker inválido")
        elif inst_ticker in st.session_state.instruments['Ticker'].values:
            st.warning(f"⚠️ **{inst_ticker}** ya está en el catálogo")
        else:
            with st.spinner(f"Verificando {inst_ticker} en Yahoo Finance..."):
                try:
                    info = yf.Ticker(inst_ticker).fast_info
                    ticker_ok = hasattr(info, 'last_price') and info.last_price and info.last_price > 0
                except:
                    ticker_ok = False
            if not ticker_ok:
                st.error(f"❌ **{inst_ticker}** no se encontró en Yahoo Finance")
            else:
                new_inst = pd.DataFrame([{'Ticker': inst_ticker, 'Sector': inst_sector}])
                st.session_state.instruments = pd.concat(
                    [st.session_state.instruments, new_inst], ignore_index=True
                )
                save_persistent(st.session_state.portfolio, st.session_state.instruments, st.session_state.sectors, st.session_state.get("sector_targets",{}))
                st.success(f"✅ **{inst_ticker}** agregado al catálogo")
                st.rerun()

    st.markdown("---")

    # ── Tabla del catálogo ──
    st.markdown(f"#### Mis instrumentos ({len(st.session_state.instruments)} cargados)")

    inst_edited = st.data_editor(
        st.session_state.instruments.copy(),
        column_config={
            'Ticker': st.column_config.TextColumn('Ticker', width='small', disabled=True),
            'Sector': st.column_config.SelectboxColumn('Sector', options=[""] + st.session_state.sectors, width='medium'),
        },
        hide_index=True,
        num_rows="dynamic",
        use_container_width=True,
        key="instruments_editor"
    )

    col_save, col_restore = st.columns([1,1])
    with col_save:
        if st.button("💾 Guardar cambios del catálogo", type="primary"):
            inst_edited_clean = inst_edited.dropna(subset=['Ticker'])
            inst_edited_clean['Ticker'] = inst_edited_clean['Ticker'].astype(str).str.upper().str.strip()
            inst_edited_clean = inst_edited_clean[inst_edited_clean['Ticker'].str.match(r'^[A-Z0-9.\-]{1,10}$', na=False)]
            st.session_state.instruments = inst_edited_clean.reset_index(drop=True)
            save_persistent(st.session_state.portfolio, st.session_state.instruments, st.session_state.sectors, st.session_state.get("sector_targets",{}))
            st.success("✅ Catálogo actualizado")
            st.rerun()
    with col_restore:
        if st.button("🔄 Restaurar catálogo original"):
            st.session_state.instruments = DEFAULT_PORTFOLIO[['Ticker','Sector']].copy().reset_index(drop=True)
            save_persistent(st.session_state.portfolio, st.session_state.instruments, st.session_state.sectors, st.session_state.get("sector_targets",{}))
            st.success("✅ Catálogo restaurado")
            st.rerun()

    st.info("💡 Los instrumentos del catálogo están disponibles en **Mi Cartera** para seleccionarlos con un click.")

# ══════════════════════════════════════════
#  TAB 1 — MI CARTERA
# ══════════════════════════════════════════
with tabs[3]:
    st.subheader("📁 Mi Cartera")

    # ── Agregar desde catálogo ──
    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown("#### Agregar desde catálogo de instrumentos")
        # Instruments not yet in portfolio
        in_portfolio = set(st.session_state.portfolio['Ticker'].values)
        available_inst = [
            t for t in st.session_state.instruments['Ticker'].values
            if t not in in_portfolio
        ]
        if available_inst:
            with st.form("add_from_catalog"):
                fc1, fc2, fc3 = st.columns([1.5, 1.2, 1])
                sel_ticker = fc1.selectbox("Instrumento", available_inst)
                sel_monto  = fc2.number_input("Monto USD", min_value=0.0, step=100.0, value=1000.0)
                sel_target = fc3.number_input("Target %", min_value=0.0, max_value=100.0, step=0.5)
                cat_submit = st.form_submit_button("➕ Agregar a mi cartera", type="primary")
            if cat_submit and sel_ticker:
                sel_sector = st.session_state.instruments[
                    st.session_state.instruments['Ticker']==sel_ticker
                ]['Sector'].values[0]
                new_row = pd.DataFrame([{
                    'Ticker': sel_ticker, 'Monto_USD': sel_monto,
                    'Target_%': sel_target, 'Sector': sel_sector
                }])
                st.session_state.portfolio = pd.concat(
                    [st.session_state.portfolio, new_row], ignore_index=True
                )
                save_persistent(st.session_state.portfolio, st.session_state.instruments, st.session_state.sectors, st.session_state.get("sector_targets",{}))
                st.success(f"✅ {sel_ticker} agregado a tu cartera")
                st.rerun()
        else:
            st.info("Todos los instrumentos del catálogo ya están en tu cartera.")

    with col2:
        st.markdown("#### Ticker nuevo (no está en catálogo)")
        with st.form("add_new_ticker"):
            n1, n2, n3 = st.columns([1.2, 1, 1])
            new_ticker = n1.text_input("Ticker", placeholder="AAPL").upper().strip()
            new_monto  = n2.number_input("Monto USD", min_value=0.0, step=100.0, value=1000.0)
            new_target = n3.number_input("Target %", min_value=0.0, max_value=100.0, step=0.5)
            new_sector = st.selectbox("Sector (se guardará en catálogo)", [""] + st.session_state.sectors)
            new_submit = st.form_submit_button("➕ Agregar y guardar en catálogo", type="primary")

        if new_submit and new_ticker:
            import re
            if not re.match(r'^[A-Z0-9.\-]{1,10}$', new_ticker):
                st.error("⚠️ Formato de ticker inválido")
            elif new_ticker in st.session_state.portfolio['Ticker'].values:
                st.warning(f"⚠️ {new_ticker} ya está en tu cartera")
            else:
                with st.spinner(f"Verificando {new_ticker} en Yahoo Finance..."):
                    try:
                        info = yf.Ticker(new_ticker).fast_info
                        ticker_ok = hasattr(info, 'last_price') and info.last_price and info.last_price > 0
                    except:
                        ticker_ok = False
                if not ticker_ok:
                    st.error(f"❌ **{new_ticker}** no se encontró en Yahoo Finance")
                else:
                    # Add to portfolio
                    new_row = pd.DataFrame([{
                        'Ticker': new_ticker, 'Monto_USD': new_monto,
                        'Target_%': new_target, 'Sector': new_sector
                    }])
                    st.session_state.portfolio = pd.concat(
                        [st.session_state.portfolio, new_row], ignore_index=True
                    )
                    # Sync to instruments catalog
                    sync_instrument(new_ticker, new_sector)
                    save_persistent(st.session_state.portfolio, st.session_state.instruments, st.session_state.sectors, st.session_state.get("sector_targets",{}))
                    st.success(f"✅ {new_ticker} agregado a tu cartera y al catálogo")
                    st.rerun()

    st.markdown("---")

    if st.button("🔄 Restaurar cartera de ejemplo"):
        st.session_state.portfolio = DEFAULT_PORTFOLIO.copy()
        save_persistent(st.session_state.portfolio, st.session_state.instruments, st.session_state.sectors, st.session_state.get("sector_targets",{}))
        st.rerun()

    if not st.session_state.portfolio.empty:
        df = calc_portfolio_weights(st.session_state.portfolio.copy())
        total = df['Total'].iloc[0]
        total_target = df['Target_%'].sum()

        # Metrics row
        m1,m2,m3,m4,m5 = st.columns(5)
        m1.metric("Total Renta Variable", fmt_usd(total),
                  help="Esta solapa trabaja solo con la renta variable (acciones, CEDEARs, BTC). "
                       "La cartera total con liquidez y renta fija se ve en el Tablero Macro.")
        m2.metric("Activos", len(df))
        m3.metric("Total Target %", f"{total_target:.2f}%",
                  delta=f"{total_target-100:.2f}% vs 100%" if abs(total_target-100)>0.1 else "✓ OK")
        m4.metric("A comprar", len(df[df['Desvio_%']<-0.5]))
        m5.metric("A vender", len(df[df['Desvio_%']>0.5]))

        if abs(total_target - 100) > 0.1:
            st.warning(f"⚠️ Los targets suman {total_target:.2f}% — deben sumar 100%")

        # Editable table
        st.markdown("#### Editá tu cartera")

        # Sort controls
        sort_col_map = {
            'Monto USD': 'Monto_USD',
            'Peso Actual %': 'Peso_Actual_%',
            'Target %': 'Target_%',
            'Desvío %': 'Desvio_%',
            'Ticker': 'Ticker',
            'Sector': 'Sector',
        }
        sc1, sc2 = st.columns([3,1])
        sort_choice = sc1.selectbox("Ordenar por", list(sort_col_map.keys()), index=0, key="sort_portfolio")
        sort_asc = sc2.selectbox("Orden", ["↓ Mayor→Menor", "↑ Menor→Mayor"], key="sort_dir") == "↑ Menor→Mayor"
        sort_field = sort_col_map[sort_choice]
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
        st.warning("⚠️ **Acordate de guardar** — cualquier cambio en la tabla recién se aplica al hacer click en el botón de abajo.")
        if st.button("💾 Guardar cambios en mi cartera", type="primary"):
            edited_clean = edited[['Ticker','Monto_USD','Target_%','Sector']].copy()
            edited_clean['Ticker'] = edited_clean['Ticker'].astype(str).str.upper().str.strip()
            edited_clean = edited_clean[edited_clean['Ticker'].str.match(r'^[A-Z0-9.\-]{1,10}$', na=False)]
            # Validate any NEW tickers not already in portfolio
            old_tickers = set(st.session_state.portfolio['Ticker'].values)
            new_tickers = [t for t in edited_clean['Ticker'].values if t not in old_tickers]
            invalid = []
            if new_tickers:
                with st.spinner(f"Validando tickers nuevos: {', '.join(new_tickers)}..."):
                    for t in new_tickers:
                        try:
                            info = yf.Ticker(t).fast_info
                            if not (hasattr(info, 'last_price') and info.last_price and info.last_price > 0):
                                invalid.append(t)
                        except:
                            invalid.append(t)
            if invalid:
                st.error(f"❌ Tickers no encontrados en Yahoo Finance: **{', '.join(invalid)}** — corregí o eliminá esas filas antes de guardar.")
            else:
                # Sync any new tickers to instruments catalog
                for _, row in edited_clean.iterrows():
                    sync_instrument(row['Ticker'], row.get('Sector',''))
                st.session_state.portfolio = edited_clean
                save_persistent(st.session_state.portfolio, st.session_state.instruments, st.session_state.sectors, st.session_state.get("sector_targets",{}))
                st.success("✅ Cambios guardados — nuevos tickers sincronizados al catálogo")
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
with tabs[4]:
    st.subheader("🏷️ Análisis por Sector")
    if st.session_state.portfolio.empty:
        st.info("Cargá tu cartera primero")
    else:
        df = calc_portfolio_weights(st.session_state.portfolio.copy())
        total = df['Total'].iloc[0]

        st.caption("Los targets por sector se editan en la solapa 🎯 Targets.")
        if 'sector_targets' not in st.session_state:
            st.session_state.sector_targets = {}

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

        # Tabla con columnas en orden lógico y Desvío visible
        df_table = sector_group[['Sector','Monto_USD','Peso_Actual','Target_%','Desvío_%','Tickers']].copy()
        df_table.columns = ['Sector','Monto USD','Peso Actual %','Target %','Desvío %','Activos']
        st.dataframe(
            df_table,
            column_config={
                'Desvío %': st.column_config.NumberColumn('Desvío %', format="%.2f%%"),
                'Peso Actual %': st.column_config.NumberColumn('Peso Actual %', format="%.2f%%"),
                'Target %': st.column_config.NumberColumn('Target %', format="%.2f%%"),
            },
            use_container_width=True, hide_index=True
        )

        # Alerts — incluye sectores con target=0 pero peso real>0 (sobrecomprado sin target)
        for _, row in sector_group.iterrows():
            desv = row['Desvío_%']
            tgt  = row['Target_%']
            if abs(desv) > 1:
                if desv > 0 and tgt == 0:
                    st.warning(f"⚠️ **{row['Sector']}**: tenés {row['Peso_Actual']:.1f}% invertido pero el target es 0% — considerá si querés mantenerlo")
                elif desv > 2:
                    st.warning(f"📈 **{row['Sector']}**: sobrecomprado en {desv:.2f}% (actual {row['Peso_Actual']:.1f}% vs target {tgt:.1f}%)")
                elif desv < -2 and tgt > 0:
                    st.info(f"📉 **{row['Sector']}**: subcomprado en {abs(desv):.2f}% (actual {row['Peso_Actual']:.1f}% vs target {tgt:.1f}%)")

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
                font_color='#e2e8f0', title_font_color='#00d4ff', title_font_size=14, showlegend=True,
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
                font_color='#e2e8f0', title_font_color='#00d4ff', title_font_size=14, yaxis_title='%',
                legend=dict(font=dict(size=11, color='#e2e8f0'), bgcolor='rgba(0,0,0,0)'))
            st.plotly_chart(fig2, use_container_width=True)

        # ── Fila 2: Desvío por sector + Pareto ──
        st.markdown("---")
        col3, col4 = st.columns(2)

        with col3:
            sg_dev = sector_group.copy()
            sg_dev['Peso_Actual_num'] = df.groupby('Sector')['Peso_Actual_%'].sum().reindex(sg_dev['Sector']).values
            sg_dev['Desvío_num'] = sg_dev['Peso_Actual_num'] - sg_dev['Target_%']
            # Mostrar todos los sectores con peso real > 0 O con target > 0
            sg_dev = sg_dev[(sg_dev['Peso_Actual_num'] > 0) | (sg_dev['Target_%'] > 0)].copy()
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
                font_color='#e2e8f0', title_font_color='#00d4ff', title_font_size=14, yaxis_title='Desvío %',
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
                font_color='#e2e8f0', title_font_color='#00d4ff', title_font_size=14,
                yaxis=dict(title='Peso %', gridcolor='rgba(30,45,69,0.6)'),
                yaxis2=dict(title='Acum %', overlaying='y', side='right',
                    range=[0, 105], showgrid=False, ticksuffix='%'),
                legend=dict(font=dict(size=11, color='#e2e8f0'), bgcolor='rgba(0,0,0,0)')
            )
            st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════
#  TAB 3 — REBALANCEO
# ══════════════════════════════════════════
with tabs[5]:
    st.subheader("⚖️ Rebalanceo")
    if st.session_state.portfolio.empty:
        st.info("Cargá tu cartera primero")
    else:
        df = calc_portfolio_weights(st.session_state.portfolio.copy())
        total = df['Total'].iloc[0]

        m1,m2,m3,m4,m5 = st.columns(5)
        m1.metric("Total Renta Variable", fmt_usd(total))
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
                font_color='#e2e8f0', title_font_color='#00d4ff', title_font_size=14, yaxis_title='%', legend=dict(font=dict(size=11, color='#e2e8f0'), bgcolor='rgba(0,0,0,0)'))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = px.pie(df, values='Peso_Actual_%', names='Ticker',
                title='Composición actual',
                color_discrete_sequence=px.colors.qualitative.Set3)
            fig2.update_traces(textposition='inside', textinfo='label+percent',
                textfont_size=10, hole=0.25)
            fig2.update_layout(paper_bgcolor='#111827', plot_bgcolor='#111827',
                font_color='#e2e8f0', title_font_color='#00d4ff', title_font_size=14, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

        buf = io.BytesIO()
        df_reb[display_cols].round(2).to_excel(buf, index=False)
        st.download_button("⬇️ Exportar rebalanceo Excel", buf.getvalue(), "rebalanceo.xlsx")


# ══════════════════════════════════════════
#  TAB 4 — ESTRATEGIA
# ══════════════════════════════════════════
with tabs[6]:
    st.subheader("🎯 Estrategia de Cartera")
    st.markdown("Definí tu **política de inversión**: cuánto % querés en cada sector, y cómo distribuirlo entre los instrumentos.")

    if st.session_state.portfolio.empty:
        st.info("Importá tu cartera primero (solapa 📥 Importar)")
    else:
        df_port = calc_portfolio_weights(st.session_state.portfolio.copy())
        total   = df_port['Total'].iloc[0]

        # ── Inicializar sector_targets si no existe ──
        if 'sector_targets' not in st.session_state:
            st.session_state.sector_targets = {}

        # Sectores presentes en la cartera
        sectores_cartera = sorted(df_port['Sector'].fillna('Sin sector').unique().tolist())

        # Inicializar targets faltantes desde suma de instrumentos actuales
        for sec in sectores_cartera:
            if sec not in st.session_state.sector_targets:
                pesos_sec = df_port[df_port['Sector']==sec]['Target_%'].sum()
                st.session_state.sector_targets[sec] = round(float(pesos_sec), 1)

        # ─────────────────────────────────────────
        # PASO 1 — Targets por sector
        # ─────────────────────────────────────────
        st.markdown("### 1️⃣ Target por sector")
        st.markdown("Editá el **% objetivo** de cada sector. Al cambiar un sector, los instrumentos se redistribuyen proporcionalmente.")

        total_sector_target = sum(st.session_state.sector_targets.get(s, 0) for s in sectores_cartera)
        if abs(total_sector_target - 100) > 0.1:
            st.warning(f"⚠️ Los targets de sector suman **{total_sector_target:.1f}%** — deben sumar 100%")
        else:
            st.success(f"✅ Targets de sector suman 100%")

        # Sector target editor — una fila por sector
        sec_rows = []
        for sec in sectores_cartera:
            tickers_sec = df_port[df_port['Sector']==sec]['Ticker'].tolist()
            peso_real   = df_port[df_port['Sector']==sec]['Peso_Actual_%'].sum()
            target_sec  = st.session_state.sector_targets.get(sec, 0.0)
            sec_rows.append({
                'Sector': sec,
                'Target Sector %': round(target_sec, 1),
                'Peso Real %': round(peso_real, 2),
                'Diferencia %': round(target_sec - peso_real, 2),
                'Instrumentos': ', '.join(tickers_sec),
            })

        df_sec = pd.DataFrame(sec_rows)

        def color_diff(val):
            try:
                v = float(val)
                if v > 1:  return 'background-color: rgba(74,222,128,0.2); color: #4ade80'
                if v < -1: return 'background-color: rgba(248,113,113,0.2); color: #f87171'
                return 'background-color: rgba(251,191,36,0.15); color: #fbbf24'
            except: return ''

        st.dataframe(
            df_sec.style.map(color_diff, subset=['Diferencia %']),
            use_container_width=True, hide_index=True
        )

        # Sector sliders
        st.markdown("#### Ajustá los targets de sector")
        cols_sec = st.columns(min(4, len(sectores_cartera)))
        changed_sectors = {}
        for i, sec in enumerate(sectores_cartera):
            with cols_sec[i % len(cols_sec)]:
                new_val = st.number_input(
                    f"{sec}", min_value=0.0, max_value=100.0, step=0.5,
                    value=float(st.session_state.sector_targets.get(sec, 0.0)),
                    key=f"sec_target_{sec}"
                )
                changed_sectors[sec] = new_val

        if st.button("💾 Guardar targets de sector y redistribuir", type="primary"):
            # Read values directly from widget keys (most reliable)
            for sec in sectores_cartera:
                key = f"sec_target_{sec}"
                if key in st.session_state:
                    changed_sectors[sec] = float(st.session_state[key])
            st.session_state.sector_targets = changed_sectors

            # Redistribute instrument targets proportionally within each sector
            new_portfolio = st.session_state.portfolio.copy()
            for sec, new_target_sec in changed_sectors.items():
                mask = new_portfolio['Sector'] == sec
                tickers_in_sec = new_portfolio[mask]['Ticker'].tolist()
                if not tickers_in_sec:
                    continue
                # Current instrument targets within sector
                old_targets = new_portfolio[mask]['Target_%'].values.astype(float)
                old_sum = old_targets.sum()
                if old_sum > 0:
                    # Proportional redistribution
                    new_targets = old_targets / old_sum * new_target_sec
                else:
                    # Equal distribution if no existing targets
                    new_targets = np.full(len(old_targets), new_target_sec / len(old_targets))
                new_portfolio.loc[mask, 'Target_%'] = np.round(new_targets, 2)

            st.session_state.portfolio = new_portfolio
            save_persistent(
                st.session_state.portfolio,
                st.session_state.instruments,
                st.session_state.sectors,
                st.session_state.sector_targets
            )
            st.success("✅ Targets guardados y redistribuidos proporcionalmente")
            st.rerun()

        st.markdown("---")

        # ─────────────────────────────────────────
        # PASO 2 — Tabla maestra sector + instrumento
        # ─────────────────────────────────────────
        st.markdown("### 2️⃣ Tabla maestra — Sector → Instrumento")
        st.markdown("Podés ajustar el **target de cada instrumento** directamente. La app te avisa si la suma del sector no coincide con el target del sector.")

        # Build master table
        master_rows = []
        for sec in sectores_cartera:
            df_sec_inst = df_port[df_port['Sector']==sec].copy()
            target_sec  = st.session_state.sector_targets.get(sec, 0.0)
            sum_inst    = df_sec_inst['Target_%'].sum()
            ok          = abs(sum_inst - target_sec) < 0.2

            for _, row in df_sec_inst.iterrows():
                master_rows.append({
                    'Sector': sec,
                    'Target Sector %': round(target_sec, 1),
                    'Ticker': row['Ticker'],
                    'Target Instrumento %': round(row['Target_%'], 2),
                    'Peso Real %': round(row['Peso_Actual_%'], 2),
                    'Desvío %': round(row['Target_%'] - row['Peso_Actual_%'], 2),
                    '✓': '✅' if ok else '⚠️',
                })
            # Subtotal row — muestra: target sector / suma targets inst / peso real
            peso_real_sec = df_sec_inst['Peso_Actual_%'].sum()
            master_rows.append({
                'Sector': f'— Subtotal {sec}',
                'Target Sector %': round(target_sec, 1),
                'Ticker': f'({len(df_sec_inst)} instrumentos)',
                'Target Instrumento %': round(sum_inst, 2),
                'Peso Real %': round(peso_real_sec, 2),
                'Desvío %': round(target_sec - peso_real_sec, 2),
                '✓': '✅' if ok else f'⚠️ targets suman {sum_inst:.1f}% ≠ {target_sec:.1f}%',
            })

        df_master = pd.DataFrame(master_rows)

        # Editable — only Target Instrumento % is editable
        edited_master = st.data_editor(
            df_master,
            column_config={
                'Sector':                  st.column_config.TextColumn('Sector', disabled=True),
                'Target Sector %':         st.column_config.NumberColumn('Target Sector %', format="%.1f%%", disabled=True),
                'Ticker':                  st.column_config.TextColumn('Ticker', disabled=True),
                'Target Instrumento %':    st.column_config.NumberColumn('Target Instrumento %', format="%.2f%%"),
                'Peso Real %':             st.column_config.NumberColumn('Peso Real %', format="%.2f%%", disabled=True),
                'Desvío %':                st.column_config.NumberColumn('Desvío %', format="%.2f%%", disabled=True),
                '✓':                       st.column_config.TextColumn('Estado', disabled=True),
            },
            hide_index=True,
            use_container_width=True,
            key="master_strategy_editor"
        )

        st.warning("⚠️ **Acordate de guardar** los cambios de targets de instrumento con el botón de abajo.")

        if st.button("💾 Guardar targets de instrumento"):
            # Apply edited instrument targets back to portfolio (skip subtotal rows)
            new_port = st.session_state.portfolio.copy()
            for _, row in edited_master.iterrows():
                ticker = str(row.get('Ticker','')).strip()
                if not ticker or ticker == '':
                    continue  # skip subtotal rows
                if ticker in new_port['Ticker'].values:
                    new_port.loc[new_port['Ticker']==ticker, 'Target_%'] = float(row['Target Instrumento %'])

            # Validate sector sums vs sector targets
            warnings_list = []
            for sec in sectores_cartera:
                target_sec = st.session_state.sector_targets.get(sec, 0.0)
                sum_inst   = new_port[new_port['Sector']==sec]['Target_%'].sum()
                if abs(sum_inst - target_sec) > 0.5:
                    warnings_list.append(f"**{sec}**: instrumentos suman {sum_inst:.1f}% pero target sector es {target_sec:.1f}%")

            if warnings_list:
                for w in warnings_list:
                    st.warning(f"⚠️ {w}")

            st.session_state.portfolio = new_port
            save_persistent(
                st.session_state.portfolio,
                st.session_state.instruments,
                st.session_state.sectors,
                st.session_state.sector_targets
            )
            st.success("✅ Targets de instrumento guardados")
            st.rerun()

        with st.expander("📖 ¿Cómo usar esta pestaña?"):
            st.markdown("""
- **Paso 1**: Definí el % objetivo de cada sector. Al guardar, los instrumentos se redistribuyen proporcionalmente.
- **Paso 2**: Ajustá fino el target de cada instrumento dentro del sector. La columna ✓ te indica si la suma de instrumentos coincide con el target del sector.
- **Verde ✅**: la suma de instrumentos = target del sector. Todo alineado.
- **Amarillo ⚠️**: hay diferencia — los instrumentos no suman el target del sector.
- Los cambios se guardan en disco y persisten entre sesiones.
            """)


# ══════════════════════════════════════════
#  TAB 4 — CORRELACIÓN
# ══════════════════════════════════════════
with tabs[7]:
    st.subheader("🔗 Correlación (datos reales)")
    if st.session_state.portfolio.empty:
        st.info("Cargá tu cartera primero")
    elif st.session_state.hist_data is None:
        st.warning("⬅️ Hacé click en **Cargar datos de mercado** en el panel lateral")
    else:
        prices = st.session_state.hist_data
        # Use tickers_loaded (YF format) intersected with prices columns
        _port_tickers = st.session_state.portfolio['Ticker'].tolist()
        _loaded = st.session_state.tickers_loaded if st.session_state.tickers_loaded else _port_tickers
        available = [t for t in _loaded if t in prices.columns]
        if len(available) < 2:
            st.warning("Se necesitan al menos 2 tickers con datos disponibles")
        else:
            prices_clean, _info = preparar_precios(prices, available)
            mostrar_aviso_ventana(_info)
            available = list(prices_clean.columns)
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
                font_color='#e2e8f0', title_font_color='#00d4ff', title_font_size=14, height=500)
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
                    font_color='#e2e8f0', title_font_color='#00d4ff', title_font_size=14, yaxis=dict(range=[0,1]))
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
                        font_color='#e2e8f0', title_font_color='#00d4ff', title_font_size=14, yaxis_title='Correlación')
                    st.plotly_chart(fig3, use_container_width=True)

        with st.expander("📖 ¿Cómo interpretar la correlación?"):
            st.markdown("""
- La correlación mide cuánto se mueven juntos dos activos. Va de **-1 a +1**.
- **+1**: se mueven exactamente igual (no diversifican).
- **0**: movimientos independientes (buena diversificación).
- **-1**: se mueven opuesto (cobertura perfecta).
- ⚠️ Correlación **>0.80** es una alerta: esos activos se comportan casi igual y no agregan diversificación real.
- El gráfico de correlación móvil muestra si la relación entre dos activos cambia con el tiempo — lo ideal es que sea estable y baja.
            """)
with tabs[8]:
    st.subheader("📐 Métricas reales")
    if st.session_state.portfolio.empty:
        st.info("Cargá tu cartera primero")
    elif st.session_state.hist_data is None:
        st.warning("⬅️ Hacé click en **Cargar datos de mercado** en el panel lateral")
    else:
        prices = st.session_state.hist_data
        df_port = calc_portfolio_weights(st.session_state.portfolio.copy())
        available = [t for t in (st.session_state.tickers_loaded or df_port['Ticker'].tolist()) if t in prices.columns]
        prices_clean, _info = preparar_precios(prices, available)
        mostrar_aviso_ventana(_info)
        available = list(prices_clean.columns)
        if not available:
            st.error("❌ Ningún ticker tiene datos suficientes. Probá aumentar los años de historia en el panel lateral.")
            st.stop()
        bench = st.session_state.get('bench_data')

        with st.spinner("Calculando métricas..."):
            try:
                # Métricas individuales: cada activo con su historia completa
                metrics, _ = calc_metrics(prices[available].ffill(limit=5), bench, rf=rf_rate, beta_freq=beta_freq)
            except Exception as e:
                st.error(f"❌ No se pudieron calcular las métricas: {e}\n\nAsegurate de que los tickers tengan datos suficientes y de haber hecho click en **Cargar datos de mercado**.")
                st.stop()

        # ── Métricas de la cartera (ventana común, pesos actuales) ──
        returns = prices_clean.pct_change().dropna()
        weights_actual = pesos_para(df_port, available)
        port_ret, port_vol, port_sharpe = portfolio_metrics(weights_actual, returns, rf_rate)
        port_serie = serie_cartera(prices_clean, weights_actual)
        port_r = port_serie.pct_change().dropna()
        port_var1d = -port_r.quantile(0.05) * 100          # VaR histórico real de la cartera (incluye diversificación)
        port_beta = calc_beta(port_serie, bench, beta_freq) if bench is not None else np.nan
        port_mdd = (port_serie / port_serie.cummax() - 1).min() * 100
        _down = np.sqrt((np.minimum(port_r, 0) ** 2).mean()) * np.sqrt(252)
        port_sortino = (port_ret - rf_rate) / _down if _down > 0 else np.nan

        mc = metricas_serie(port_serie, bench, rf_rate, beta_freq)

        # ── Resumen ejecutivo con semáforos ──
        st.markdown("#### Resumen de tu renta variable")
        filas_sem = [
            ("CAGR (crecimiento anual compuesto)", f"{mc['CAGR']*100:.1f}%", semaforo(mc['CAGR'], 0.08, 0.04)),
            ("Volatilidad anual", f"{mc['Volatilidad']*100:.1f}%", semaforo(mc['Volatilidad'], 0.15, 0.25, False)),
            ("Sharpe", f"{mc['Sharpe']:.2f}", semaforo(mc['Sharpe'], 1.0, 0.5)),
            ("Sortino", f"{mc['Sortino']:.2f}", semaforo(mc['Sortino'], 1.5, 0.8)),
            ("Calmar", f"{mc['Calmar']:.2f}", semaforo(mc['Calmar'], 0.5, 0.25)),
            ("Máx Drawdown", f"{mc['Máx Drawdown']*100:.1f}%", semaforo(-mc['Máx Drawdown'], 0.20, 0.35, False)),
            (f"Beta vs SPY ({beta_freq.lower()})", f"{mc['Beta']:.2f}" if pd.notna(mc['Beta']) else "—", "⚪"),
            ("Alfa de Jensen (anual)", f"{mc['Alfa']*100:+.1f}%" if pd.notna(mc['Alfa']) else "—", semaforo(mc['Alfa'], 0.0, -0.03)),
            ("VaR 1d 95%", f"{mc['VaR 1d 95%']*100:.2f}%", "⚪"),
            ("CVaR 1d 95%", f"{mc['CVaR 1d 95%']*100:.2f}%", "⚪"),
        ]
        cs1, cs2 = st.columns(2)
        for i, (lab, val, sem) in enumerate(filas_sem):
            (cs1 if i % 2 == 0 else cs2).markdown(f"{sem} **{lab}:** {val}")
        st.caption("🟢 bueno · 🟡 aceptable · 🔴 a revisar · ⚪ informativo. Umbrales orientativos de uso habitual, no reglas.")

        # ── Comparación: tu cartera vs SPY vs carteras óptimas ──
        st.markdown("#### Tu cartera vs SPY y carteras óptimas")
        comp = {'Tu cartera': mc}
        if bench is not None:
            comp['SPY'] = metricas_serie(bench.reindex(prices_clean.index).ffill(), bench, rf_rate, beta_freq)
        try:
            w_ms = max_sharpe(returns, rf=rf_rate)
            w_mv = min_variance(returns)
            comp['Máx Sharpe (teórica)'] = metricas_serie(serie_cartera(prices_clean, w_ms), bench, rf_rate, beta_freq)
            comp['Mín Varianza (teórica)'] = metricas_serie(serie_cartera(prices_clean, w_mv), bench, rf_rate, beta_freq)
        except Exception:
            pass
        PCT = {'CAGR', 'Retorno anual', 'Volatilidad', 'Alfa', 'Máx Drawdown', 'VaR 1d 95%', 'CVaR 1d 95%'}
        df_comp = pd.DataFrame(comp)
        df_show = df_comp.copy().astype(object)
        for k in df_comp.index:
            df_show.loc[k] = [("—" if pd.isna(v) else (f"{v*100:.1f}%" if k in PCT else f"{v:.2f}")) for v in df_comp.loc[k]]
        st.dataframe(df_show, use_container_width=True)
        st.caption("Las carteras óptimas son teóricas, armadas desde cero mirando el pasado: sirven como referencia, "
                   "no como receta (suelen concentrarse en pocos activos que anduvieron bien).")

        # ── Beta móvil 12 meses ──
        if bench is not None:
            st.markdown("#### Beta móvil (12 meses)")
            _df = pd.concat([port_serie, bench], axis=1, keys=['p', 'b']).dropna()
            if str(beta_freq).startswith('Semanal'):
                _r = _df.resample('W-FRI').last().pct_change().dropna(); _win = 52
            else:
                _r = _df.pct_change().dropna(); _win = 252
            if len(_r) > _win + 5:
                rb = (_r['p'].rolling(_win).cov(_r['b']) / _r['b'].rolling(_win).var()).dropna()
                fig_rb = go.Figure()
                fig_rb.add_trace(go.Scatter(x=rb.index, y=rb.values, line=dict(color='#00d4ff', width=2), name='Beta 12m'))
                fig_rb.add_hline(y=1, line_dash='dot', line_color='#94a3b8', annotation_text='Beta 1 (igual al SPY)')
                if pd.notna(mc['Beta']):
                    fig_rb.add_hline(y=mc['Beta'], line_dash='dash', line_color='#fbbf24', annotation_text='Beta del período')
                fig_rb.update_layout(height=320, paper_bgcolor='#111827', plot_bgcolor='#111827',
                                     font_color='#e2e8f0', showlegend=False, yaxis_title='Beta')
                st.plotly_chart(fig_rb, use_container_width=True)
                st.caption(f"Beta de tu cartera calculada sobre los 12 meses previos a cada fecha (rango: "
                           f"{rb.min():.2f} a {rb.max():.2f}). Si se mueve mucho, la beta del período completo es un promedio que esconde etapas distintas.")
            else:
                st.caption("Hace falta más de un año de historia común para calcular la beta móvil.")

        st.session_state['port_stats'] = {'ret': port_ret, 'vol': port_vol, 'sharpe': port_sharpe,
                                          'beta': port_beta if not np.isnan(port_beta) else 1.0}

        # ── Contribución al riesgo ──
        st.markdown("#### Contribución al riesgo de la cartera")
        st.caption("Qué parte de la volatilidad total aporta cada activo. Un activo puede pesar poco y aportar mucho riesgo (o al revés) según su volatilidad y su correlación con el resto.")
        _cov = returns.cov().values * 252
        _w = weights_actual
        _var_p = float(_w @ _cov @ _w)
        if _var_p > 0:
            _rc = _w * (_cov @ _w) / _var_p * 100
            df_rc = pd.DataFrame({'Ticker': available, 'Peso %': _w * 100, 'Aporte al riesgo %': _rc})
            df_rc['Riesgo / Peso'] = np.where(df_rc['Peso %'] > 0, df_rc['Aporte al riesgo %'] / df_rc['Peso %'], np.nan)
            df_rc = df_rc.sort_values('Aporte al riesgo %', ascending=False).round(2)
            c_rc1, c_rc2 = st.columns([3, 2])
            with c_rc1:
                _top = df_rc.head(20).iloc[::-1]
                fig_rc = go.Figure()
                fig_rc.add_trace(go.Bar(y=_top['Ticker'], x=_top['Peso %'], name='Peso %', orientation='h',
                                        marker_color='rgba(0,212,255,0.7)'))
                fig_rc.add_trace(go.Bar(y=_top['Ticker'], x=_top['Aporte al riesgo %'], name='Aporte al riesgo %',
                                        orientation='h', marker_color='rgba(248,113,113,0.8)'))
                fig_rc.update_layout(barmode='group', height=max(350, 22 * len(_top)),
                    title='Peso vs aporte al riesgo (top 20)',
                    paper_bgcolor='#111827', plot_bgcolor='#111827', font_color='#e2e8f0',
                    title_font_color='#00d4ff', title_font_size=14,
                    legend=dict(font=dict(size=11, color='#e2e8f0'), bgcolor='rgba(0,0,0,0)'))
                st.plotly_chart(fig_rc, use_container_width=True)
            with c_rc2:
                st.dataframe(df_rc, use_container_width=True, hide_index=True,
                    column_config={
                        'Peso %': st.column_config.NumberColumn(format="%.2f%%"),
                        'Aporte al riesgo %': st.column_config.NumberColumn(format="%.2f%%"),
                        'Riesgo / Peso': st.column_config.NumberColumn(format="%.2fx",
                            help=">1: aporta más riesgo que su peso. <1: diversifica."),
                    })

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
                paper_bgcolor='#111827', plot_bgcolor='#111827', font_color='#e2e8f0', title_font_color='#00d4ff', title_font_size=14)
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
                font_color='#e2e8f0', title_font_color='#00d4ff', title_font_size=14, showlegend=False,
                xaxis_title='Volatilidad %', yaxis_title='Retorno Anual %')
            st.plotly_chart(fig2, use_container_width=True)

        with st.expander("📖 ¿Cómo interpretar estas métricas?"):
            st.markdown("""
- **Retorno Anual %**: Rendimiento anualizado promedio en el período seleccionado.
- **Volatilidad %**: Desviación estándar anualizada — mide el riesgo total del activo.
- **Sharpe**: Retorno ajustado por riesgo. >1 es bueno, >2 es excelente. Menor a 0.5 indica que el riesgo no se está compensando.
- **CAGR %**: Tasa de crecimiento anual compuesta — el retorno real si hubieras mantenido el activo todo el período.
- **Beta SPY**: Sensibilidad al mercado. Beta=1 se mueve igual que el mercado. Beta>1 amplifica movimientos. Beta<1 es más defensivo.
- **VaR 1d 95%**: Pérdida máxima esperada en un día normal (95% de confianza). Ejemplo: VaR=2% significa que solo hay 5% de chances de perder más de 2% en un día. El de la cartera se calcula sobre su serie real, así que ya incluye el efecto de la diversificación.
- **Sortino**: Como el Sharpe, pero solo penaliza la volatilidad a la baja. Más útil cuando un activo tiene subas fuertes que inflan su volatilidad.
- **Máx Drawdown %**: La peor caída desde un máximo hasta un mínimo en el período. Es lo que realmente se siente en el bolsillo.
- **Historia desde**: Cada activo se mide con toda su historia disponible dentro de los años elegidos. La cartera, en cambio, se mide en la ventana común a todos.
- **Beta**: Todos los activos (CEDEARs, ADRs argentinos, acciones locales convertidas a USD con CCL y Bitcoin) se miden igual: retornos en USD contra SPY, mismas fechas. Por eso son comparables entre sí.
            """)


# ══════════════════════════════════════════
#  TAB 6 — RENDIMIENTO
# ══════════════════════════════════════════
with tabs[9]:
    st.subheader("📈 Rendimiento acumulado")
    if st.session_state.portfolio.empty:
        st.info("Cargá tu cartera primero")
    elif st.session_state.hist_data is None:
        st.warning("⬅️ Cargá datos de mercado primero")
    else:
        prices = st.session_state.hist_data
        df_port = calc_portfolio_weights(st.session_state.portfolio.copy())
        available = [t for t in (st.session_state.tickers_loaded or df_port['Ticker'].tolist()) if t in prices.columns]
        prices_clean, _info = preparar_precios(prices, available)
        mostrar_aviso_ventana(_info)
        available = list(prices_clean.columns)

        weights_actual = np.array([float(df_port[df_port['Ticker']==t]['Peso_Actual_%'].values[0]) if len(df_port[df_port['Ticker']==t]) > 0 else 0.0 for t in available])
        _w2_sum = weights_actual.sum()
        weights_actual = (weights_actual / _w2_sum) if _w2_sum > 0 else np.ones(len(available)) / len(available)

        # Portfolio combined price
        port_prices = serie_cartera(prices_clean, weights_actual)
        port_cum = (port_prices / port_prices.iloc[0] - 1) * 100

        # Download benchmarks
        with st.spinner("Descargando SPY y QQQ..."):
            try:
                bench_raw = yf.download(['SPY','QQQ'], period=PERIOD, auto_adjust=True, progress=False)
                if isinstance(bench_raw.columns, pd.MultiIndex):
                    bench = bench_raw['Close']
                else:
                    bench = bench_raw
                bench = bench.reindex(port_cum.index, method='ffill').dropna()
                spy_cum  = (bench['SPY']  / bench['SPY'].iloc[0]  - 1) * 100 if 'SPY'  in bench.columns else None
                qqq_cum  = (bench['QQQ']  / bench['QQQ'].iloc[0]  - 1) * 100 if 'QQQ'  in bench.columns else None
            except:
                spy_cum, qqq_cum = None, None

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=port_cum.index, y=port_cum.values,
            name='Mi Cartera', line=dict(color='#00d4ff', width=2.5)))
        if spy_cum is not None:
            fig.add_trace(go.Scatter(x=spy_cum.index, y=spy_cum.values,
                name='SPY', line=dict(color='#f87171', width=1.5, dash='dash')))
        if qqq_cum is not None:
            fig.add_trace(go.Scatter(x=qqq_cum.index, y=qqq_cum.values,
                name='QQQ', line=dict(color='#fbbf24', width=1.5, dash='dot')))

        # Individual tickers
        show_tickers = st.checkbox("Mostrar activos individuales", value=False)
        if show_tickers:
            for i, t in enumerate(available):
                cum = (prices_clean[t] / prices_clean[t].iloc[0] - 1) * 100
                fig.add_trace(go.Scatter(x=cum.index, y=cum.values,
                    name=t, line=dict(width=1, dash='dot'),
                    opacity=0.6))

        fig.add_hline(y=0, line_dash='dot', line_color='rgba(255,255,255,0.2)')
        fig.update_layout(
            title='Rendimiento Acumulado (%)',
            paper_bgcolor='#111827', plot_bgcolor='#111827',
            font_color='#e2e8f0', title_font_color='#00d4ff', title_font_size=14, yaxis_title='Retorno %',
            legend=dict(font=dict(size=11, color='#e2e8f0'), bgcolor='rgba(0,0,0,0)')
        )
        st.plotly_chart(fig, use_container_width=True)

        # CAGR comparison bar
        st.markdown("#### CAGR comparativo")
        cagr_items = {'Mi Cartera': float((port_prices.iloc[-1]/port_prices.iloc[0])**(252/len(port_prices))-1)*100}
        if spy_cum is not None:
            cagr_items['SPY'] = float((bench['SPY'].iloc[-1]/bench['SPY'].iloc[0])**(252/len(bench))-1)*100
        if qqq_cum is not None:
            cagr_items['QQQ'] = float((bench['QQQ'].iloc[-1]/bench['QQQ'].iloc[0])**(252/len(bench))-1)*100

        fig2 = go.Figure(go.Bar(
            x=list(cagr_items.keys()), y=list(cagr_items.values()),
            marker_color=['#00d4ff','#f87171','#fbbf24'][:len(cagr_items)],
            text=[f"{v:.1f}%" for v in cagr_items.values()], textposition='outside'
        ))
        fig2.update_layout(title='CAGR anualizado',
            paper_bgcolor='#111827', plot_bgcolor='#111827',
            font_color='#e2e8f0', title_font_color='#00d4ff', title_font_size=14, yaxis_title='%', legend=dict(font=dict(size=11, color='#e2e8f0'), bgcolor='rgba(0,0,0,0)'))
        st.plotly_chart(fig2, use_container_width=True)

        with st.expander("📖 ¿Cómo interpretar este gráfico?"):
            st.markdown("""
- El gráfico muestra el crecimiento acumulado de $1 invertido al inicio del período.
- **Línea azul** = tu cartera con los pesos actuales.
- **Línea roja** = SPY (S&P 500 — benchmark del mercado americano).
- **Línea amarilla** = QQQ (Nasdaq 100 — benchmark tecnológico).
- **CAGR** = Tasa de Crecimiento Anual Compuesta — es el retorno anual equivalente si hubieras mantenido la inversión todo el período.
            """)


# ══════════════════════════════════════════
#  TAB 7 — VaR
# ══════════════════════════════════════════
with tabs[10]:
    st.subheader("⚠️ Value at Risk (VaR) — 95% de confianza")
    if st.session_state.portfolio.empty:
        st.info("Cargá tu cartera primero")
    elif st.session_state.hist_data is None:
        st.warning("⬅️ Cargá datos de mercado primero")
    else:
        prices = st.session_state.hist_data
        df_port = calc_portfolio_weights(st.session_state.portfolio.copy())
        available = [t for t in (st.session_state.tickers_loaded or df_port['Ticker'].tolist()) if t in prices.columns]
        prices_clean, _info = preparar_precios(prices, available)
        mostrar_aviso_ventana(_info)
        available = list(prices_clean.columns)
        returns_all = prices_clean.pct_change().dropna()

        raw_weights = []
        for t in available:
            match = df_port[df_port['Ticker']==t]['Peso_Actual_%']
            raw_weights.append(float(match.values[0]) if len(match) > 0 else 0.0)
        raw_weights = np.array(raw_weights)
        total_w = raw_weights.sum()
        weights_actual = raw_weights / total_w if total_w > 0 else raw_weights  # renormalizar

        port_ret_series = returns_all.dot(weights_actual)
        var_1d = float(-port_ret_series.quantile(0.05) * 100)
        var_10d = var_1d * np.sqrt(10)
        cvar = float(-port_ret_series[port_ret_series <= port_ret_series.quantile(0.05)].mean() * 100)

        total = df_port['Total'].iloc[0]
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("VaR 1d 95%", f"{var_1d:.2f}%", delta=f"-{var_1d/100*total:,.0f} USD")
        m2.metric("VaR 10d 95%", f"{var_10d:.2f}%", delta=f"-{var_10d/100*total:,.0f} USD")
        m3.metric("CVaR 1d 95%", f"{cvar:.2f}%", delta=f"-{cvar/100*total:,.0f} USD")
        m4.metric("Capital en riesgo (1d)", fmt_usd(var_1d/100*total))

        # Histogram
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=port_ret_series.values * 100,
            nbinsx=60,
            name='Retornos diarios',
            marker_color='rgba(0,212,255,0.6)',
            opacity=0.8
        ))
        fig.add_vline(x=-var_1d, line_dash='dash', line_color='#f87171', line_width=2,
            annotation_text=f'VaR 95% = -{var_1d:.2f}%',
            annotation_font_color='#f87171', annotation_position='top left')
        fig.add_vline(x=-cvar, line_dash='dot', line_color='#fbbf24', line_width=2,
            annotation_text=f'CVaR = -{cvar:.2f}%',
            annotation_font_color='#fbbf24', annotation_position='bottom left')
        fig.update_layout(
            title='Distribución de Retornos Diarios de la Cartera',
            paper_bgcolor='#111827', plot_bgcolor='#111827',
            font_color='#e2e8f0', title_font_color='#00d4ff', title_font_size=14, xaxis_title='Retorno Diario %', yaxis_title='Frecuencia',
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

        # VaR by ticker
        st.markdown("#### VaR por activo")
        var_data = []
        for t in available:
            r = returns_all[t]
            v1 = float(-r.quantile(0.05)*100)
            var_data.append({'Ticker':t, 'VaR 1d %':round(v1,2), 'VaR 10d %':round(v1*np.sqrt(10),2),
                'CVaR 1d %':round(float(-r[r<=r.quantile(0.05)].mean()*100),2)})
        st.dataframe(pd.DataFrame(var_data).sort_values('VaR 1d %', ascending=False),
            use_container_width=True, hide_index=True)

        with st.expander("📖 ¿Cómo interpretar el VaR?"):
            st.markdown("""
- **VaR 1d 95%**: Pérdida máxima esperada en un **día normal**. Con 95% de confianza, no se espera perder más de ese % en un solo día.
- **VaR 10d 95%**: Mismo concepto pero proyectado a 10 días hábiles (se escala por √10).
- **CVaR (Expected Shortfall)**: Promedio de las pérdidas que **sí superan** el VaR — mide qué tan malo puede ser el peor 5% de los días.
- La línea roja en el histograma marca el VaR. Todo lo que queda a la izquierda de esa línea es el 5% de peores días históricos.
- Ejemplo: VaR=2.5% significa que solo 1 de cada 20 días se perdió más de 2.5%.
            """)



# ══════════════════════════════════════════
#  TAB 6 — STRESS TEST
# ══════════════════════════════════════════
with tabs[11]:
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
            font_color='#e2e8f0', title_font_color='#00d4ff', title_font_size=14, yaxis_title='% impacto', xaxis_tickangle=-25)
        st.plotly_chart(fig, use_container_width=True)

        # Historical drawdown from real data
        if st.session_state.hist_data is not None:
            st.markdown("#### Drawdown histórico real de tu cartera")
            prices = st.session_state.hist_data
            df_p = calc_portfolio_weights(st.session_state.portfolio.copy())
            available = [t for t in (st.session_state.tickers_loaded or df_p['Ticker'].tolist()) if t in prices.columns]
            if available:
                _pp, _ = preparar_precios(prices, available)
                w = pesos_para(df_p, list(_pp.columns))
                port_prices = serie_cartera(_pp, w)
                port_prices = port_prices / port_prices.iloc[0] * 100
                rolling_max = port_prices.cummax()
                drawdown = (port_prices - rolling_max) / rolling_max * 100

                fig2 = go.Figure(go.Scatter(x=drawdown.index, y=drawdown.values,
                    fill='tozeroy', line=dict(color='#f87171', width=1.5),
                    fillcolor='rgba(248,113,113,0.15)'))
                fig2.update_layout(title='Drawdown histórico real',
                    paper_bgcolor='#111827', plot_bgcolor='#111827',
                    font_color='#e2e8f0', title_font_color='#00d4ff', title_font_size=14, yaxis_title='Drawdown %')
                st.plotly_chart(fig2, use_container_width=True)
                st.info(f"📉 Máximo drawdown histórico: **{drawdown.min():.2f}%**")

        with st.expander("📖 ¿Cómo interpretar el Stress Test?"):
            st.markdown("""
- Cada fila muestra la **caída estimada de tu cartera** si el SPY (mercado) baja ese porcentaje.
- El cálculo usa la **Beta** de tu cartera: si Beta=1.2 y el SPY cae 10%, tu cartera caería aproximadamente 12%.
- 🟡 Pérdida leve (<15%) · 🟠 Pérdida moderada (15-30%) · 🔴 Pérdida severa (>30%)
- Los escenarios históricos (2008, COVID) reflejan caídas reales del mercado en esos eventos.
- El **Drawdown histórico** muestra la caída máxima real que tuvo tu cartera en el período analizado.
- ⚠️ Estos son estimados — la relación real puede variar según correlaciones y condiciones de mercado.
            """)


# ══════════════════════════════════════════
#  TAB 9 — FRONTERA EFICIENTE
# ══════════════════════════════════════════
with tabs[12]:
    st.subheader("🌐 Frontera Eficiente de Markowitz")
    if st.session_state.portfolio.empty:
        st.info("Cargá tu cartera primero")
    elif st.session_state.hist_data is None:
        st.warning("⬅️ Cargá datos de mercado primero")
    else:
        prices = st.session_state.hist_data
        df_port = calc_portfolio_weights(st.session_state.portfolio.copy())
        available = [t for t in (st.session_state.tickers_loaded or df_port['Ticker'].tolist()) if t in prices.columns]
        if len(available) < 2:
            st.warning("Se necesitan al menos 2 activos con datos.")
        else:
            prices_clean, _info = preparar_precios(prices, available)
            mostrar_aviso_ventana(_info)
            returns_fe = prices_clean.pct_change().dropna()
            available = list(prices_clean.columns)  # update available after cleaning
            min_w = st.number_input("Peso mínimo por activo (%)", min_value=0.0, max_value=float(int(10000 / len(available)) / 100),
                value=min(float(st.session_state.get('min_weight', 0.0)) * 100, float(int(10000 / len(available)) / 100)), step=0.1,
                help="Obliga a que cada activo tenga al menos este peso en las carteras óptimas. 0 = sin mínimo.") / 100
            st.session_state['min_weight'] = min_w

            with st.spinner("Calculando frontera eficiente (puede tardar ~20 segundos)..."):
                # Random portfolios
                n = len(available)
                n_sim = 3000
                rand_rets, rand_vols, rand_sharpes, rand_weights = [], [], [], []
                np.random.seed(42)
                for _ in range(n_sim):
                    w = np.random.dirichlet(np.ones(n))
                    r, v, s = portfolio_metrics(w, returns_fe, rf_rate)
                    rand_rets.append(r*100); rand_vols.append(v*100); rand_sharpes.append(s)

                # Key portfolios
                w_ms  = max_sharpe(returns_fe, rf_rate, min_w)
                w_mv  = min_variance(returns_fe, min_w)
                w_act = np.array([float(df_port[df_port['Ticker']==t]['Peso_Actual_%'].values[0]) if len(df_port[df_port['Ticker']==t]) > 0 else 0.0 for t in available])
                _w3_sum = w_act.sum()
                # Renormalizar para que sumen 1.0 (independiente de cuánto pesa RV en la cartera total)
                w_act = (w_act / _w3_sum) if _w3_sum > 0 else np.ones(len(available)) / len(available)

                ms_r, ms_v, ms_s = portfolio_metrics(w_ms,  returns_fe, rf_rate)
                mv_r, mv_v, mv_s = portfolio_metrics(w_mv,  returns_fe, rf_rate)
                ac_r, ac_v, ac_s = portfolio_metrics(w_act, returns_fe, rf_rate)

            fig = go.Figure()

            # Random cloud
            fig.add_trace(go.Scatter(
                x=rand_vols, y=rand_rets,
                mode='markers',
                marker=dict(size=4, color=rand_sharpes, colorscale='Viridis',
                    showscale=True, colorbar=dict(title='Sharpe', tickfont=dict(color='#e2e8f0')),
                    opacity=0.5),
                name='Portfolios aleatorios', hovertemplate='Vol: %{x:.1f}%<br>Ret: %{y:.1f}%<extra></extra>'
            ))

            # Key points
            for label, rv, rr, rs, color, sym in [
                ('Max Sharpe',    ms_v*100, ms_r*100, ms_s, '#f87171', 'star'),
                ('Min Varianza',  mv_v*100, mv_r*100, mv_s, '#4ade80', 'circle'),
                ('Tu Cartera',    ac_v*100, ac_r*100, ac_s, '#00d4ff', 'x'),
            ]:
                fig.add_trace(go.Scatter(
                    x=[rv], y=[rr], mode='markers+text',
                    name=f'{label} (Sharpe: {rs:.2f})',
                    text=[label], textposition='top center',
                    marker=dict(size=18, color=color, symbol=sym, line=dict(width=2, color=color))
                ))

            fig.update_layout(
                title='Espacio de Portfolios — Frontera Eficiente de Markowitz',
                paper_bgcolor='#111827', plot_bgcolor='#111827', font_color='#e2e8f0', title_font_color='#00d4ff', title_font_size=14,
                xaxis_title='Volatilidad Anual %', yaxis_title='Retorno Anual %',
                legend=dict(font=dict(size=11, color='#e2e8f0'), bgcolor='rgba(0,0,0,0)'),
                height=550
            )
            st.plotly_chart(fig, use_container_width=True)

            # Composition of key portfolios
            st.markdown("#### Composición de portfolios óptimos")
            col1, col2, col3 = st.columns(3)
            for col, label, w, r, v, s, color in [
                (col1, '🔴 Max Sharpe',   w_ms,  ms_r, ms_v, ms_s, '#f87171'),
                (col2, '🟢 Min Varianza', w_mv,  mv_r, mv_v, mv_s, '#4ade80'),
                (col3, '🔵 Tu Cartera',   w_act, ac_r, ac_v, ac_s, '#00d4ff'),
            ]:
                with col:
                    st.markdown(f"**{label}**")
                    st.metric("Retorno Anual", f"{r*100:.1f}%")
                    st.metric("Volatilidad", f"{v*100:.1f}%")
                    st.metric("Sharpe", f"{s:.2f}")
                    df_comp = pd.DataFrame({'Ticker': available, 'Peso %': (w*100).round(1)})\
                        .sort_values('Peso %', ascending=False)
                    # Convert hex color to rgba for transparency
                    color_map = {'#f87171':'rgba(248,113,113,0.7)','#4ade80':'rgba(74,222,128,0.7)','#00d4ff':'rgba(0,212,255,0.7)'}
                    bar_color = color_map.get(color, 'rgba(0,212,255,0.7)')
                    fig_comp = go.Figure(go.Bar(
                        x=df_comp['Ticker'], y=df_comp['Peso %'],
                        marker_color=bar_color,
                        text=df_comp['Peso %'].astype(str)+'%', textposition='outside'
                    ))
                    fig_comp.update_layout(paper_bgcolor='#111827', plot_bgcolor='#111827',
                        font_color='#e2e8f0', title_font_color='#00d4ff', title_font_size=14, height=250, margin=dict(t=10,b=10),
                        yaxis_title='%', showlegend=False)
                    st.plotly_chart(fig_comp, use_container_width=True)

            with st.expander("📖 ¿Cómo interpretar la Frontera Eficiente?"):
                st.markdown("""
- Cada punto del gráfico representa un **portfolio posible** con distintas combinaciones de pesos.
- El **color** de cada punto indica su Sharpe Ratio (más amarillo = mejor relación riesgo/retorno).
- La **curva superior izquierda** de puntos forma la Frontera Eficiente — son los portfolios que maximizan retorno para cada nivel de riesgo.
- **Max Sharpe** ⭐: el portfolio con la mejor relación retorno/riesgo.
- **Min Varianza** 🟢: el portfolio más conservador — menor volatilidad posible.
- **Tu Cartera** 🔵: dónde está tu distribución actual en este espacio.
- Si tu cartera está lejos de la frontera, existe un portfolio más eficiente con el mismo riesgo pero mayor retorno.
                """)



# ══════════════════════════════════════════
#  TAB 8 — MONTE CARLO
# ══════════════════════════════════════════
with tabs[14]:
    st.subheader("🎲 Monte Carlo — escenarios a futuro")
    st.caption("Simula miles de caminos posibles para tu renta variable con los pesos actuales. "
               "No predice: muestra el rango de resultados razonables según el retorno y la volatilidad que supongas.")
    if st.session_state.portfolio.empty:
        st.info("Importá tu cartera primero (solapa 📥 Importar)")
    elif 'port_stats' not in st.session_state:
        st.warning("⬅️ Cargá datos de mercado primero (el cálculo usa la volatilidad real de tu cartera)")
    else:
        df_port = calc_portfolio_weights(st.session_state.portfolio.copy())
        total = df_port['Total'].iloc[0]
        ps = st.session_state['port_stats']

        c1, c2, c3 = st.columns(3)
        mu = c1.number_input("Retorno anual supuesto (%)", value=float(round(min(ps['ret'], 0.15) * 100, 1)),
            min_value=-30.0, max_value=60.0, step=0.5,
            help=f"El retorno histórico de tu cartera fue {ps['ret']*100:.1f}% anual, pero proyectarlo tal cual "
                 f"suele ser optimista. Por defecto se usa el menor entre el histórico y 15%.") / 100
        vol = c2.number_input("Volatilidad anual (%)", value=float(round(ps['vol'] * 100, 1)),
            min_value=1.0, max_value=150.0, step=0.5,
            help="Por defecto, la volatilidad real de tu cartera en la ventana de análisis.") / 100
        anios_mc = c3.selectbox("Horizonte", [1, 2, 3, 5], index=0, format_func=lambda x: f"{x} año{'s' if x > 1 else ''}")

        n_days = int(252 * anios_mc)
        np.random.seed(42)
        paths = monte_carlo(mu, vol, total, 2000, n_days)
        finales = paths[:, -1]
        p5, p50, p95 = np.percentile(finales, [5, 50, 95])
        prob_perdida = (finales < total).mean() * 100

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Hoy", fmt_usd(total))
        m2.metric("Escenario malo (P5)", fmt_usd(p5), delta=f"{(p5/total-1)*100:.1f}%")
        m3.metric("Escenario medio (P50)", fmt_usd(p50), delta=f"{(p50/total-1)*100:.1f}%")
        m4.metric("Escenario bueno (P95)", fmt_usd(p95), delta=f"{(p95/total-1)*100:.1f}%")
        st.caption(f"Probabilidad de terminar el período con menos de lo que tenés hoy: **{prob_perdida:.0f}%**")

        x = np.arange(n_days + 1)
        bandas = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=bandas[4], line=dict(width=0), showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=x, y=bandas[0], fill='tonexty', fillcolor='rgba(0,212,255,0.12)',
                                 line=dict(width=0), name='Rango P5–P95'))
        fig.add_trace(go.Scatter(x=x, y=bandas[3], line=dict(width=0), showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=x, y=bandas[1], fill='tonexty', fillcolor='rgba(0,212,255,0.25)',
                                 line=dict(width=0), name='Rango P25–P75'))
        fig.add_trace(go.Scatter(x=x, y=bandas[2], line=dict(color='#00d4ff', width=2.5), name='Mediana'))
        fig.add_hline(y=total, line_dash='dash', line_color='#94a3b8', annotation_text='Capital actual')
        fig.update_layout(height=420, xaxis_title='Días hábiles', yaxis_title='u$s',
            paper_bgcolor='#111827', plot_bgcolor='#111827', font_color='#e2e8f0',
            legend=dict(font=dict(size=11, color='#e2e8f0'), bgcolor='rgba(0,0,0,0)'))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("⚠️ Supone retornos con distribución normal. En crisis reales las caídas suelen ser más fuertes "
                   "que lo que muestra el escenario malo.")


# ══════════════════════════════════════════
#  OPTIMIZACIÓN — SIMULADOR DE BETA POR SECTOR
# ══════════════════════════════════════════
CAMPOS_AGRUP = {"Sector macro": "Sector_Macro", "Subsector": "Sector_Detalle", "País": "Pais"}

def mapa_grupos(campo):
    """Ticker (Yahoo) → grupo, a partir de la clasificación del Excel importado."""
    m = {}
    bz = st.session_state.get('balanz_data')
    if bz is not None and len(bz) and campo in bz.columns:
        rv = bz[bz['Renta'].astype(str).str.strip() == 'Variable']
        for _, r in rv.iterrows():
            v = str(r.get(campo, '')).strip()
            if v and v not in ('0', 'None', 'nan', '#N/A'):
                m.setdefault(ticker_yf_balanz(r['Ticker'], r.get('Instrumento', '')), v)
    return m

with tabs[15]:
    st.subheader("🎚️ Simulador de beta por sector")
    st.caption("Mirá tu renta variable por sector, probá subir unos y bajar otros, y fijate qué pasa con la beta "
               "y la volatilidad. Al subir un sector se supone que comprás más de lo que ya tenés en él, en la misma proporción.")
    if st.session_state.portfolio.empty:
        st.info("Importá tu cartera primero (solapa 📥 Importar)")
    elif st.session_state.hist_data is None or st.session_state.get('bench_data') is None:
        st.warning("⬅️ Cargá datos de mercado primero")
    else:
        agrup = st.radio("Agrupar por", list(CAMPOS_AGRUP.keys()), horizontal=True, key="sim_agrup")
        campo = CAMPOS_AGRUP[agrup]
        hist = st.session_state.hist_data
        bench = st.session_state.bench_data
        dfp = calc_portfolio_weights(st.session_state.portfolio.copy())
        total_rv = float(dfp['Monto_USD'].sum())
        mapa = mapa_grupos(campo)
        dfp['Grupo'] = [mapa.get(t) or (s if campo == 'Sector_Macro' and s else 'Sin clasificar')
                        for t, s in zip(dfp['Ticker'], dfp['Sector'].fillna(''))]
        if not mapa and campo != 'Sector_Macro':
            st.warning("No encuentro esa clasificación: volvé a importar el Excel para traer subsector y país.")

        # ── Serie de retornos de cada grupo (ventana común, pesos actuales dentro del grupo) ──
        pc, _info = preparar_precios(hist, dfp['Ticker'].tolist())
        rets = pc.pct_change().dropna()
        R, sin_datos = {}, []
        for g, sub in dfp.groupby('Grupo'):
            tk = [t for t in sub['Ticker'] if t in rets.columns]
            if not tk:
                sin_datos.append(g)
                continue
            ww = sub.groupby('Ticker')['Monto_USD'].sum().reindex(tk).values
            R[g] = pd.Series(rets[tk].values @ (ww / ww.sum()), index=rets.index)

        # ── Sectores nuevos (teóricos) ──
        with st.expander("➕ Agregar un sector nuevo (teórico)", expanded=False):
            st.caption("Poné un nombre, el % que le querés asignar y uno o más tickers representativos separados por coma "
                       "(acciones o un ETF del sector). La app calcula su beta con datos reales.")
            nuevos_df = st.data_editor(
                st.session_state.get('sim_nuevos', pd.DataFrame({'Sector': [''], 'Tickers': [''], '% a asignar': [0.0]})),
                num_rows="dynamic", use_container_width=True, key="sim_nuevos_editor",
                column_config={'% a asignar': st.column_config.NumberColumn(format="%.2f", step=0.5),
                               'Tickers': st.column_config.TextColumn(help="Ej: COHR, AAOI")})
        nuevos = []
        for _, nr in nuevos_df.iterrows():
            nombre = str(nr.get('Sector') or '').strip()
            tks = [yf_ticker(x) for x in str(nr.get('Tickers') or '').replace(';', ',').split(',') if x.strip()]
            pct = float(nr.get('% a asignar') or 0)
            if not nombre or not tks:
                continue
            if nombre in R:
                nombre = f"{nombre} (nuevo)"
            with st.spinner(f"Bajando datos de {', '.join(tks)}..."):
                p_new, _b, _n = fetch_data(tks, f"{years}y")
            if p_new is None or p_new.empty:
                st.warning(f"No hay datos para {', '.join(tks)}.")
                continue
            r_new = p_new.reindex(pc.index).ffill(limit=5).pct_change()
            r_new = r_new.loc[rets.index].mean(axis=1, skipna=True)   # igual peso entre los tickers
            if r_new.notna().sum() < 60:
                st.warning(f"{nombre}: historia insuficiente en la ventana de análisis.")
                continue
            R[nombre] = r_new
            nuevos.append({'Grupo': nombre, 'pct': pct, 'tickers': ', '.join(tks)})

        R = pd.DataFrame(R)
        precios_g = (1 + R.fillna(0)).cumprod()
        beta_g = {g: calc_beta(precios_g[g].where(R[g].notna()), bench, beta_freq) for g in R.columns}
        vol_g = (R.std() * np.sqrt(252)).to_dict()
        cov = R.cov() * 252

        # ── Tabla editable ──
        act = dfp.groupby('Grupo')['Monto_USD'].sum().sort_values(ascending=False)
        base = pd.DataFrame({'Grupo': act.index, '% actual': (act.values / total_rv * 100).round(2)})
        base['Beta'] = base['Grupo'].map(beta_g).round(2)
        base['Volatilidad %'] = (base['Grupo'].map(vol_g) * 100).round(1)
        base['Cambio (pp)'] = 0.0
        key_tab = f"sim_tabla_{campo}"
        if st.button("↺ Volver a cero los cambios"):
            st.session_state.pop(key_tab, None)
            st.rerun()
        st.markdown("##### Escribí en **Cambio** cuántos puntos subís (+) o bajás (−) cada grupo")
        ed = st.data_editor(base, key=key_tab, hide_index=True, use_container_width=True,
            disabled=['Grupo', '% actual', 'Beta', 'Volatilidad %'],
            column_config={
                '% actual': st.column_config.NumberColumn(format="%.2f%%"),
                'Beta': st.column_config.NumberColumn(format="%.2f", help="Beta del grupo vs SPY con los pesos actuales de sus activos"),
                'Volatilidad %': st.column_config.NumberColumn(format="%.1f%%"),
                'Cambio (pp)': st.column_config.NumberColumn(format="%+.2f", step=0.5),
            })

        # ── Resultado ──
        res = ed[['Grupo', '% actual', 'Beta']].copy()
        res['Cambio (pp)'] = ed['Cambio (pp)'].fillna(0).astype(float)
        for nv in nuevos:
            res = pd.concat([res, pd.DataFrame([{'Grupo': nv['Grupo'], '% actual': 0.0,
                             'Beta': round(beta_g.get(nv['Grupo'], np.nan), 2), 'Cambio (pp)': nv['pct']}])],
                            ignore_index=True)
        res['% nuevo'] = res['% actual'] + res['Cambio (pp)']
        suma_cambios = res['Cambio (pp)'].sum()
        negativos = res[res['% nuevo'] < -0.005]

        if abs(suma_cambios) > 0.05:
            st.error(f"🚨 Los cambios no suman cero: suman **{suma_cambios:+.2f} pp** "
                     f"(la cartera nueva da {res['% nuevo'].sum():.2f}%). "
                     f"{'Te falta bajar' if suma_cambios > 0 else 'Te falta subir'} {abs(suma_cambios):.2f} pp en algún grupo.")
        if not negativos.empty:
            st.error("🚨 Quedan con % negativo: " + ", ".join(negativos['Grupo']) + ". No podés bajar más de lo que tenés.")

        def _beta_vol(pcts):
            g = [x for x in pcts.index if x in cov.columns and pd.notna(beta_g.get(x))]
            x = pcts.reindex(g).fillna(0).clip(lower=0).values
            if x.sum() <= 0:
                return np.nan, np.nan, pd.Series(dtype=float)
            x = x / x.sum()
            b = float(np.dot(x, [beta_g[k] for k in g]))
            C = cov.loc[g, g].fillna(0).values
            var = float(x @ C @ x)
            aporte = pd.Series(x * (C @ x) / var * 100 if var > 0 else np.nan, index=g)
            return b, np.sqrt(var), aporte

        b0, v0, _ = _beta_vol(res.set_index('Grupo')['% actual'])
        b1, v1, aporte1 = _beta_vol(res.set_index('Grupo')['% nuevo'])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Beta actual", f"{b0:.2f}")
        c2.metric("Beta simulada", f"{b1:.2f}", delta=f"{b1-b0:+.2f}", delta_color="off")
        c3.metric("Volatilidad actual", f"{v0*100:.1f}%")
        c4.metric("Volatilidad simulada", f"{v1*100:.1f}%", delta=f"{(v1-v0)*100:+.1f} pp", delta_color="inverse")

        res['u$s a mover'] = (res['Cambio (pp)'] / 100 * total_rv).round(0)
        res['u$s nuevo'] = (res['% nuevo'] / 100 * total_rv).round(0)
        res['Aporte a la beta'] = (res['% nuevo'] / 100 * res['Beta']).round(3)
        res['Aporte al riesgo %'] = res['Grupo'].map(aporte1).round(1)
        cambios = res[res['Cambio (pp)'].abs() > 0.001]
        st.markdown("##### Resultado")
        st.dataframe(res[['Grupo', '% actual', 'Cambio (pp)', '% nuevo', 'u$s a mover', 'u$s nuevo',
                          'Beta', 'Aporte a la beta', 'Aporte al riesgo %']]
                     .sort_values('% nuevo', ascending=False),
            hide_index=True, use_container_width=True,
            column_config={
                '% actual': st.column_config.NumberColumn(format="%.2f%%"),
                'Cambio (pp)': st.column_config.NumberColumn(format="%+.2f"),
                '% nuevo': st.column_config.NumberColumn(format="%.2f%%"),
                'u$s a mover': st.column_config.NumberColumn(format="%+,.0f"),
                'u$s nuevo': st.column_config.NumberColumn(format="%,.0f"),
                'Aporte al riesgo %': st.column_config.NumberColumn(format="%.1f%%",
                    help="Qué parte de la volatilidad de la cartera simulada aporta cada grupo"),
            })
        if not cambios.empty:
            st.caption("Movimientos: " + " · ".join(
                f"{r['Grupo']} {'+' if r['u$s a mover'] > 0 else ''}u$s {r['u$s a mover']:,.0f}" for _, r in cambios.iterrows()))
        notas = []
        if sin_datos:
            notas.append(f"Sin datos de precios (no entran en beta ni volatilidad): {', '.join(sin_datos)}.")
        if _info.get('cortos'):
            notas.append(f"Activos con historia corta excluidos del cálculo: {', '.join(_info['cortos'])}.")
        notas.append(f"Beta {beta_freq.lower()} vs SPY, ventana desde {pc.index[0]:%m/%Y}. "
                     "Puede diferir unas centésimas de la solapa Métricas porque acá se calcula por grupo.")
        st.caption(" ".join(notas))


# ══════════════════════════════════════════
#  GUÍA DE USO
# ══════════════════════════════════════════
with T[6]:
    st.subheader("📖 Guía de uso")
    g1, g2, g3 = st.tabs(["🚶 Paso a paso", "📊 Qué significa cada resultado", "❓ Preguntas frecuentes"])

    with g1:
        st.markdown("""
### Antes de entrar: preparar el Excel
1. En tu Excel, solapa **"Actualizar de Balanz"**, pegá la tenencia de Balanz **en la vista en dólares** (los valores vienen con "u$s").
2. Lo que no viene de Balanz (BTC, Wallbit) cargalo **en dólares, el número solo**.
3. Lo de bancos (FCI, plazos fijos, cajas de ahorro) cargalo **en pesos**, con un tipo de instrumento que diga **"Banco"**. La app los pasa a dólares sola con el MEP del día.
4. Si compraste algo nuevo, clasificalo en **"Detalle instrumentos"** (renta, sector, país). La hoja "Revisar!" te avisa si falta algo.
5. **Guardá el Excel.**

### En la app
**Paso 1 — Importar** (solapa 📥 Importar)
- Subí el Excel. El tipo de cambio viene solo (MEP automático).
- Revisá la vista previa: la columna **"Leído en"** dice USD o ARS para cada instrumento. Lo de bancos tiene que decir ARS y todo lo demás USD.
- Compará el **total en u$s** con el que te muestra Balanz más tus bancos. Si hay un cartel rojo, leelo antes de seguir.
- Tocá **"Confirmar e importar cartera"**.

**Paso 2 — Cargar datos de mercado** (panel izquierdo)
- Tocá **🚀 Cargar datos de mercado**. Baja la historia de precios de Yahoo Finance (tarda uno o dos minutos).
- Sin este paso, las solapas de Riesgo y Optimización quedan vacías.

**Paso 3 — Mirar la foto general** (🏠 Cartera total)
- Cuánto tenés en renta variable, renta fija y liquidez; en pesos y dólares; por sector.

**Paso 4 — Analizar el riesgo** (📐 Riesgo → Métricas)
- Arriba, el resumen con semáforos. Después la comparación contra el SPY, la beta móvil y qué activos te aportan más riesgo.

**Paso 5 — Simular un rebalanceo** (🎯 Optimización → 🎚️ Simulador de beta)
- Elegí agrupar por sector, subsector o país.
- En **Cambio** poné cuántos puntos subís (+) o bajás (−). Tienen que sumar cero.
- Mirá cómo cambian la beta y la volatilidad, y cuántos u$s mover en cada grupo.
- Para evaluar un sector que no tenés, usá **"Agregar un sector nuevo"** con tickers de ejemplo.

**Paso 6 — Afinar por activo** (🧩 Renta variable)
- Una vez decidido el movimiento por sector, elegí qué activos comprar o vender dentro de cada uno.

**Paso 7 — Exportar** (panel izquierdo → 📤 Exportar resumen)
- Genera un Excel con todo lo calculado. Sirve para guardar un registro o para mandármelo si algo no te cierra.

### Configuración del panel izquierdo
- **Años de historia:** cuánto pasado se descarga. 5 es un buen equilibrio.
- **Historia mínima para el análisis de cartera:** los activos más nuevos que esto quedan fuera de los cálculos de cartera, para no achicar la ventana de todos. Se avisa qué % de la cartera queda afuera.
- **Frecuencia para calcular Beta:** semanal (recomendada) o diaria.
- **Tasa libre de riesgo:** el rendimiento "sin riesgo" contra el que se comparan Sharpe, Sortino y Alfa.
""")

    with g2:
        st.markdown("""
### Retorno
- **Retorno anual:** promedio de los retornos diarios, anualizado. Tiende a ser algo mayor que el CAGR.
- **CAGR:** cuánto creció la cartera por año, en forma compuesta. Es el número "real" de crecimiento.
- **Alfa de Jensen:** cuánto ganaste **por encima** de lo que te correspondía por tu beta. Positivo = tus elecciones agregaron valor más allá de seguir al mercado. Negativo = te hubiera ido mejor con un índice de igual riesgo de mercado.

### Riesgo
- **Volatilidad:** cuánto oscila la cartera en un año. 15% = en un año normal, la cartera se mueve en un rango de ±15% respecto de su promedio.
- **Beta vs SPY:** cuánto se mueve tu cartera **junto con** el SPY. Beta 1 = si el SPY cae 10%, tu cartera cae ~10% por ese motivo. Beta 0,5 = cae ~5%. **Ojo:** la beta no mide todo el riesgo, solo la parte ligada a Wall Street. Bitcoin o las acciones argentinas pueden tener beta baja y riesgo muy alto.
- **Beta móvil:** la beta calculada mes a mes sobre los 12 meses anteriores. Muestra si es estable o si cambia según la época.
- **Máx Drawdown:** la peor caída desde un máximo hasta un mínimo. Es lo que se siente en el bolsillo.
- **VaR 1d 95%:** la pérdida diaria que solo se supera 1 de cada 20 días. VaR 2% = en 19 de cada 20 días perdés menos de 2%.
- **CVaR 1d 95%:** cuánto perdés **en promedio** en ese peor 5% de los días. Es más honesto que el VaR, porque mira qué tan malos son los días malos.
- **Aporte al riesgo:** qué parte de la volatilidad total aporta cada activo o sector. Si algo pesa 10% y aporta 30% del riesgo, es el que manda.

### Retorno ajustado por riesgo
- **Sharpe:** retorno por encima de la tasa sin riesgo, por cada unidad de volatilidad. Más de 1 es muy bueno, entre 0,5 y 1 aceptable.
- **Sortino:** igual que Sharpe pero solo castiga la volatilidad a la baja. Útil con activos que tienen subas fuertes.
- **Calmar:** CAGR dividido la peor caída. Responde: ¿cuánto crecimiento obtuve por cada punto de caída que tuve que soportar?

### Otras herramientas
- **Correlación:** de -1 a 1, cuánto se mueven juntos dos activos. Cerca de 1 = no diversifican entre sí.
- **Frontera eficiente:** las mejores combinaciones teóricas de riesgo y retorno con tus activos, y dónde está tu cartera hoy.
- **Carteras óptimas (Máx Sharpe / Mín Varianza):** referencias teóricas armadas desde cero mirando el pasado. No son recomendaciones.
- **Stress test:** cuánto perdería tu cartera en escenarios de crisis.
- **Monte Carlo:** miles de caminos posibles a futuro. El escenario malo (P5) es lo que solo un 5% de los caminos hace peor.
- **Simulador de beta:** qué pasa con la beta y la volatilidad si movés porcentajes entre sectores.

### Semáforos
Son umbrales orientativos de uso habitual en gestión de carteras (por ejemplo, Sharpe mayor a 1 en verde). No son reglas: una cartera agresiva a propósito va a tener volatilidad en rojo, y está bien si es lo que buscás.
""")

    with g3:
        st.markdown("""
**¿Por qué mi beta es baja si tengo mucho Bitcoin?**
Porque Bitcoin es muy volátil pero se mueve poco junto con el SPY. Su riesgo aparece en la volatilidad y en el aporte al riesgo, no en la beta.

**¿Por qué la beta de una acción argentina es tan baja?**
Porque se mueve sobre todo por factores locales (política, riesgo país), no por Wall Street. No significa que sea defensiva.

**¿Por qué la beta cambia unas centésimas entre solapas?**
Porque algunas se calculan activo por activo y otras por grupo, y la ventana de datos puede variar levemente.

**¿Con qué dólar se valúa?**
Con el MEP de venta del día (dolarapi.com). Solo afecta a lo que cargás en pesos (bancos); lo de Balanz ya viene en dólares.

**Un activo no aparece en los análisis.**
Puede ser que Yahoo no tenga datos de ese ticker, o que tenga menos historia que el mínimo configurado. La app avisa en cada caso.

**¿Hay que volver a importar cada vez?**
La app guarda la última cartera importada, pero se puede perder cuando se actualiza. Ante la duda, importá de nuevo el Excel: es un minuto.
""")


# ══════════════════════════════════════════
#  DATOS — SECTORES
# ══════════════════════════════════════════
with _dt[2]:
    st.subheader("🏷️ Sectores")
    st.caption("Normalmente los sectores vienen de tu Excel. Esto es solo para cargas manuales.")
    st.write(", ".join(st.session_state.sectors) if st.session_state.sectors else "Sin sectores")
    new_sector = st.text_input("Nuevo sector", placeholder="Ej: REITs")
    if st.button("➕ Agregar sector") and new_sector.strip():
        if new_sector.strip() not in st.session_state.sectors:
            st.session_state.sectors.append(new_sector.strip())
            save_persistent(st.session_state.portfolio, st.session_state.instruments, st.session_state.sectors, st.session_state.get("sector_targets",{}))
            st.success(f"Sector '{new_sector}' agregado")
        else:
            st.warning("Ya existe ese sector")

    sector_to_del = st.selectbox("Eliminar sector", ["—"] + st.session_state.sectors)
    if st.button("🗑️ Eliminar sector") and sector_to_del != "—":
        st.session_state.sectors.remove(sector_to_del)
        save_persistent(st.session_state.portfolio, st.session_state.instruments, st.session_state.sectors, st.session_state.get("sector_targets",{}))
        st.success(f"Sector '{sector_to_del}' eliminado")
