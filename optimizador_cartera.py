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

# ══════════════════════════════════════════════════════════════════════
#  v5 — UNA SOLA PÁGINA: importar arriba, análisis en desplegables abajo
# ══════════════════════════════════════════════════════════════════════
VERSION = "v5.0"
FONDO = '#111827'

def estilo(fig, alto=380, titulo=None):
    fig.update_layout(paper_bgcolor=FONDO, plot_bgcolor=FONDO, font_color='#e2e8f0',
                      title_font_color='#00d4ff', title_font_size=14, height=alto,
                      margin=dict(t=45 if titulo else 20, b=30, l=10, r=10),
                      legend=dict(font=dict(size=11, color='#e2e8f0'), bgcolor='rgba(0,0,0,0)'))
    if titulo:
        fig.update_layout(title=titulo)
    return fig

def es_ticker_yf_valido(t):
    """Tickers que Yahoo no puede resolver (bonos argentinos, FCI, efectivo) quedan fuera del análisis histórico."""
    import re
    t = str(t).strip()
    if t in ('BCMMUSDA', 'Dolares Cash', 'Pesos Cash', 'Plazo fijo'):
        return False
    if re.match(r'^(GD|AL|AE|BP|TX|TV|DICP|CUAP)[0-9]', t):
        return False
    if t == 'BTC-USD':
        return True
    if len(t) > 6 and '.' not in t and not t.endswith('.BA') and '-' not in t:
        return False
    return True

# ─────────────────────────────────────────
#  PANEL LATERAL: solo ajustes de cálculo
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Ajustes del análisis")
    st.caption("Podés dejarlos como están. Si cambiás algo, el análisis se recalcula solo.")
    years = st.slider("Años de historia", min_value=1, max_value=20, value=5,
        help="Cuántos años de precios se usan para medir rendimiento, riesgo, beta y correlaciones.")
    PERIOD = f"{years}y"
    rf_rate = st.number_input("Tasa libre de riesgo (%)", value=2.0, step=0.25,
        help="Lo que rendiría una inversión sin riesgo en dólares (ej: un bono del Tesoro de EEUU). "
             "Se usa en Sharpe, Sortino y alfa.") / 100
    st.session_state['min_hist_anios'] = st.slider(
        "Historia mínima de cada activo (años)", min_value=0.0, max_value=10.0, value=2.0, step=0.5,
        help="Los activos más nuevos que esto quedan fuera de los cálculos de cartera para no acortar "
             "la ventana de todos. 0 = incluir todos.")
    beta_freq = st.selectbox("Frecuencia para beta y correlación", ["Semanal", "Diaria"], index=0,
        help="Semanal (viernes a viernes) es lo recomendado: las acciones argentinas y Bitcoin operan en "
             "otros horarios y días, y con datos diarios la beta y la correlación salen distorsionadas.")
    st.markdown("---")
    if st.session_state.get('hist_data') is not None:
        if st.button("📤 Preparar resumen en Excel", use_container_width=True,
                     help="Un Excel con la configuración, métricas, cada activo, la cartera total y los precios usados."):
            with st.spinner("Armando el resumen..."):
                try:
                    st.session_state.resumen_xlsx = generar_resumen_excel(years, rf_rate, beta_freq)
                except Exception as e:
                    st.session_state.resumen_xlsx = None
                    st.error(f"No se pudo armar el resumen: {e}")
        if st.session_state.get('resumen_xlsx'):
            st.download_button("⬇️ Descargar resumen", data=st.session_state.resumen_xlsx,
                file_name=f"resumen_cartera_{pd.Timestamp.today():%Y-%m-%d}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True)
    st.caption(f"📊 Optimizador de Cartera {VERSION} · Datos: Yahoo Finance")


# ─────────────────────────────────────────
#  DATOS DE MERCADO (se bajan solos al importar)
# ─────────────────────────────────────────
def asegurar_mercado(forzar=False):
    """Baja los precios de la renta variable si faltan o cambió la cartera o los años.
    No hace falta apretar ningún botón: se llama al importar y al abrir la app."""
    port = st.session_state.portfolio
    if port is None or port.empty:
        return
    port = port.copy()
    port['Ticker'] = port['Ticker'].map(yf_ticker)
    st.session_state.portfolio = port
    todos = port.loc[port['Monto_USD'] > 0, 'Ticker'].tolist()
    tks = [t for t in todos if es_ticker_yf_valido(t)]
    clave = (tuple(sorted(tks)), PERIOD)
    if not forzar and st.session_state.get('hist_data') is not None and st.session_state.get('mercado_clave') == clave:
        return
    if not tks:
        return
    if forzar:
        fetch_data.clear()
    with st.spinner(f"Bajando precios de {len(tks)} activos de Yahoo Finance ({years} años)…"):
        precios, bench, notas = fetch_data(tks, PERIOD)
    st.session_state.data_notas = notas + ([f"Fuera del análisis histórico (Yahoo no los tiene): "
                                            f"{', '.join(t for t in todos if t not in tks)}."]
                                           if len(tks) < len(todos) else [])
    if precios is None:
        st.session_state.hist_data = None
        st.session_state.bench_data = None
        st.error(" ".join(notas))
        return
    st.session_state.hist_data = precios
    st.session_state.bench_data = bench
    st.session_state.tickers_loaded = list(precios.columns)
    st.session_state.mercado_clave = clave
    st.session_state.mercado_hora = pd.Timestamp.now(tz='America/Argentina/Buenos_Aires')


# ─────────────────────────────────────────
#  1 · IMPORTAR
# ─────────────────────────────────────────
def seccion_importar():
    st.markdown("Subí tu Excel (solapa **'Actualizar de Balanz'**). Al confirmar, la app baja sola los precios de mercado y arma el análisis.")

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
                        elif 'actual' in hl and 'numero' in hl.replace('ú', 'u'):
                            col_map['V_Num'] = i               # "V. Actual ( Numero )": todo en dólares, ya convertido en el Excel
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
                    usa_col_num = 'V_Num' in col_map
                    if usa_col_num:
                        tipos_en_pesos = []
                        st.success("✅ Uso la columna **'V. Actual ( Numero )'** de tu Excel: todos los valores se toman "
                                   "en **dólares** tal cual (los de bancos ya vienen convertidos desde el Excel).")
                    else:
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
                        v_actual_raw = row[col_map['V_Num']] if usa_col_num and len(row) > col_map['V_Num'] \
                                       else row[col_map.get('V_Actual', 8)]
                        v_actual_str = str(v_actual_raw).strip()
                        v_actual_num, moneda_texto = leer_valor(v_actual_raw)
                        if v_actual_num is None:
                            if v_actual_str not in ('', 'None', '-', 'nan', '#N/A'):
                                filas_ilegibles.append(f"{ticker} ('{v_actual_str}')")
                            continue
                        tipo_inst = str(row[_idx_inst] or '').strip() if len(row) > _idx_inst else ''
                        # Regla: por defecto DÓLARES. Pesos si el tipo está marcado como "en pesos".
                        # Si el valor trae explícito u$s o $, manda eso.
                        if usa_col_num:
                            es_usd_real = True
                        elif moneda_texto:
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
                        if st.button("✅ Confirmar, importar y analizar", type="primary"):
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
                            st.session_state.mercado_clave = None      # fuerza bajar precios de la cartera nueva
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


# ─────────────────────────────────────────
#  BASE COMÚN DEL ANÁLISIS (se calcula una vez)
# ─────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def _optimas(rets, rf):
    return max_sharpe(rets, rf=rf), min_variance(rets)

def base_analisis():
    hist = st.session_state.get('hist_data')
    bench = st.session_state.get('bench_data')
    if hist is None or st.session_state.portfolio.empty:
        return None
    dfp = calc_portfolio_weights(st.session_state.portfolio.copy())
    disp = [t for t in (st.session_state.tickers_loaded or dfp['Ticker'].tolist()) if t in hist.columns]
    pc, info = preparar_precios(hist, disp)
    if pc.shape[1] < 1:
        return None
    tks = list(pc.columns)
    w = pesos_para(dfp, tks)
    serie = serie_cartera(pc, w)
    rets = pc.pct_change().dropna()
    b = bench.reindex(pc.index).ffill() if bench is not None else None
    return dict(dfp=dfp, pc=pc, info=info, tks=tks, w=w, serie=serie, rets=rets, bench=bench, b=b,
                total_rv=float(dfp['Monto_USD'].sum()))

def ret_frec(df):
    """Retornos a la frecuencia elegida (semanal por defecto) para beta y correlación."""
    if str(beta_freq).startswith('Semanal'):
        return df.resample('W-FRI').last().pct_change().dropna(how='all')
    return df.pct_change().dropna(how='all')


# ─────────────────────────────────────────
#  2 · COMPOSICIÓN
# ─────────────────────────────────────────
def norm_renta(r):
    r = str(r or '').strip()
    if 'ariable' in r: return 'Renta variable'
    if 'ija' in r: return 'Renta fija'
    if 'iquid' in r or 'Plazo Fijo' in r: return 'Liquidez'
    return 'Otros'

def seccion_composicion():
    bz = st.session_state.balanz_data.copy()
    tc = float(st.session_state.get('balanz_tipo_cambio') or 1)
    bz['u$s'] = bz['V_Actual_Pesos'] / tc
    bz['Renta_N'] = bz['Renta'].apply(norm_renta)
    bz['Moneda_N'] = bz['Moneda'].apply(lambda m: 'Dólares' if 'olar' in str(m) else 'Pesos')
    T = bz['u$s'].sum()

    st.markdown("#### Cartera total")
    c1, c2 = st.columns(2)
    for col, campo, titulo, colores in [(c1, 'Renta_N', 'Por tipo de renta', ['#00d4ff', '#4ade80', '#fbbf24', '#94a3b8']),
                                        (c2, 'Moneda_N', 'Por moneda', ['#00d4ff', '#f87171'])]:
        g = bz.groupby(campo)['u$s'].sum().sort_values(ascending=False)
        with col:
            fig = go.Figure(go.Pie(labels=g.index, values=g.values, hole=0.5, sort=False,
                                   marker_colors=colores[:len(g)], textinfo='label+percent'))
            st.plotly_chart(estilo(fig, 260, titulo).update_layout(showlegend=False), use_container_width=True)
            st.dataframe(pd.DataFrame({'': g.index, 'u$s': g.values.round(0), '%': (g.values / T * 100).round(1)}),
                         hide_index=True, use_container_width=True,
                         column_config={'u$s': st.column_config.NumberColumn(format="%,.0f"),
                                        '%': st.column_config.NumberColumn(format="%.1f%%")})
    st.caption(f"Total: **u$s {T:,.0f}** · tipo de cambio usado para lo que está en pesos: ${tc:,.2f} "
               f"({st.session_state.get('balanz_tc_fuente', '')})")

    st.markdown("---")
    st.markdown("#### Renta variable por sector")
    dfp = calc_portfolio_weights(st.session_state.portfolio.copy())
    total_rv = dfp['Monto_USD'].sum()
    tg = dict(st.session_state.get('sector_targets') or {})
    if tg and 0 < sum(tg.values()) <= 1.5:              # en el Excel pueden venir como fracción (0,15)
        tg = {k: v * 100 for k, v in tg.items()}
    sec = (dfp.groupby('Sector').agg(u=('Monto_USD', 'sum'), Activos=('Ticker', lambda x: ', '.join(x)))
              .sort_values('u', ascending=False).reset_index())
    sec['% RV'] = sec['u'] / total_rv * 100
    sec['Target Excel %'] = sec['Sector'].map(tg)
    sec['Desvío (pp)'] = sec['% RV'] - sec['Target Excel %']
    fig = go.Figure()
    fig.add_trace(go.Bar(y=sec['Sector'], x=sec['% RV'], orientation='h', name='Hoy',
                         marker_color='rgba(0,212,255,0.75)', text=sec['% RV'].round(1).astype(str) + '%',
                         textposition='outside'))
    if tg:
        fig.add_trace(go.Bar(y=sec['Sector'], x=sec['Target Excel %'], orientation='h', name='Target del Excel',
                             marker_color='rgba(251,191,36,0.6)'))
    fig.update_layout(barmode='group', yaxis=dict(autorange='reversed'), xaxis_title='% de la renta variable')
    st.plotly_chart(estilo(fig, max(320, 34 * len(sec))), use_container_width=True)
    st.dataframe(sec[['Sector', 'u', '% RV', 'Target Excel %', 'Desvío (pp)', 'Activos']].rename(columns={'u': 'u$s'}),
                 hide_index=True, use_container_width=True,
                 column_config={'u$s': st.column_config.NumberColumn(format="%,.0f"),
                                '% RV': st.column_config.NumberColumn(format="%.1f%%"),
                                'Target Excel %': st.column_config.NumberColumn(format="%.1f%%"),
                                'Desvío (pp)': st.column_config.NumberColumn(format="%+.1f",
                                    help="Hoy menos target. Positivo: tenés más de lo que dice tu Excel.")})
    st.caption(f"Renta variable: **u$s {total_rv:,.0f}**. Los targets se leen de la hoja *Sector Macro - Target* "
               "de tu Excel y no se editan acá: si los querés cambiar, cambialos en el Excel y volvé a importar.")


# ─────────────────────────────────────────
#  3 · SEMÁFORO
# ─────────────────────────────────────────
# (nombre, clave, formato, regla, verde, amarillo, mayor_es_mejor, qué es, criterio)
INDICADORES = [
    ("Crecimiento anual (CAGR)", 'CAGR', 'pct', 'mm', 0.08, 0.04, True,
     "Cuánto creció tu renta variable por año, en promedio compuesto, en el período analizado.",
     "🟢 8% o más · 🟡 entre 4% y 8% · 🔴 menos de 4%"),
    ("Volatilidad anual", 'Volatilidad', 'pct', 'mm', 0.15, 0.25, False,
     "Cuánto se mueve el valor de la cartera hacia arriba y abajo en un año. Más volatilidad = más sobresaltos.",
     "🟢 15% o menos · 🟡 entre 15% y 25% · 🔴 más de 25% (el SPY suele andar en 15-20%)"),
    ("Sharpe", 'Sharpe', 'num', 'mm', 1.0, 0.5, True,
     "Cuánto rendimiento extra (sobre la tasa libre de riesgo) obtenés por cada unidad de volatilidad. Mide si el riesgo 'paga'.",
     "🟢 1 o más · 🟡 entre 0,5 y 1 · 🔴 menos de 0,5"),
    ("Sortino", 'Sortino', 'num', 'mm', 1.5, 0.8, True,
     "Como el Sharpe, pero solo cuenta las bajas como riesgo (las subas fuertes no te penalizan).",
     "🟢 1,5 o más · 🟡 entre 0,8 y 1,5 · 🔴 menos de 0,8"),
    ("Caída máxima", 'Máx Drawdown', 'pct', 'dd', 0.20, 0.35, False,
     "La peor caída desde un máximo hasta el mínimo siguiente en el período. Es lo que más se siente en el bolsillo.",
     "🟢 hasta −20% · 🟡 entre −20% y −35% · 🔴 peor que −35%"),
    ("Calmar", 'Calmar', 'num', 'mm', 0.5, 0.25, True,
     "Crecimiento anual dividido la caída máxima. Cuánto ganás por cada punto de caída que tuviste que aguantar.",
     "🟢 0,5 o más · 🟡 entre 0,25 y 0,5 · 🔴 menos de 0,25"),
    ("Beta vs SPY", 'Beta', 'num', 'beta', 1.1, 1.3, False,
     "Cuánto se mueve tu cartera cuando se mueve el mercado (SPY). Beta 1,3 = si el SPY cae 10%, tu cartera tiende a caer 13%.",
     "🟢 hasta 1,1 (parecido al mercado o menos) · 🟡 entre 1,1 y 1,3 · 🔴 más de 1,3"),
    ("Alfa (anual)", 'Alfa', 'pct', 'mm', 0.0, -0.03, True,
     "Lo que rendiste de más o de menos respecto de lo que 'correspondía' por tu beta. Positivo = le ganaste al mercado ajustando por riesgo.",
     "🟢 0% o más · 🟡 entre −3% y 0% · 🔴 peor que −3%"),
    ("Pérdida de un mal día (VaR 95%)", 'VaR 1d 95%', 'pct', 'mm', 0.02, 0.03, False,
     "En 19 de cada 20 días no perdiste más que esto. Solo 1 día de cada 20 fue peor.",
     "🟢 hasta 2% · 🟡 entre 2% y 3% · 🔴 más de 3% (el SPY ronda 1,5-2%)"),
    ("Pérdida promedio de los peores días (CVaR)", 'CVaR 1d 95%', 'pct', 'mm', 0.03, 0.045, False,
     "Cuánto perdiste en promedio en ese 5% de días malos. Mide qué tan feos son los días feos.",
     "🟢 hasta 3% · 🟡 entre 3% y 4,5% · 🔴 más de 4,5%"),
]

def _color(v, regla, verde, amarillo, mayor):
    if v is None or pd.isna(v):
        return 'blanco'
    if regla == 'dd':
        v = -v
    if mayor:
        return 'verde' if v >= verde else ('amarillo' if v >= amarillo else 'rojo')
    return 'verde' if v <= verde else ('amarillo' if v <= amarillo else 'rojo')

def _fmt(v, f):
    if v is None or pd.isna(v):
        return "—"
    return f"{v*100:.1f}%" if f == 'pct' else f"{v:.2f}"

def seccion_semaforo(B):
    mc = metricas_serie(B['serie'], B['bench'], rf_rate, beta_freq)
    spy = metricas_serie(B['b'], B['bench'], rf_rate, beta_freq) if B['b'] is not None else {}
    grupos = {'verde': [], 'amarillo': [], 'rojo': [], 'blanco': []}
    for nom, k, f, regla, ve, am, mayor, que, crit in INDICADORES:
        grupos[_color(mc.get(k), regla, ve, am, mayor)].append((nom, k, f, que, crit))
    st.caption("Pasá el cursor por el **?** de cada indicador para ver qué es y con qué criterio se pinta. "
               "Abajo de cada valor tenés el del SPY en el mismo período, para comparar.")
    titulos = {'verde': '🟢 Bien', 'amarillo': '🟡 Atento', 'rojo': '🔴 A revisar', 'blanco': '⚪ Sin dato'}
    cols = st.columns(3)
    for col, g in zip(cols, ['verde', 'amarillo', 'rojo']):
        with col:
            st.markdown(f"##### {titulos[g]} ({len(grupos[g])})")
            if not grupos[g]:
                st.caption("Ninguno")
            for nom, k, f, que, crit in grupos[g]:
                st.metric(nom, _fmt(mc.get(k), f), help=f"{que}\n\nCriterio: {crit}")
                if spy:
                    st.caption(f"SPY: {_fmt(spy.get(k), f)}")
    if grupos['blanco']:
        st.caption("⚪ Sin dato: " + ", ".join(n for n, *_ in grupos['blanco']))
    st.caption(f"Renta variable con tus pesos actuales, desde {B['pc'].index[0]:%m/%Y}. "
               "Umbrales orientativos de uso habitual, no reglas.")


# ─────────────────────────────────────────
#  4 · RENDIMIENTO vs MERCADO
# ─────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def _qqq(period):
    try:
        raw = yf.download(['QQQ'], period=period, auto_adjust=True, progress=False)
        s = raw['Close'] if isinstance(raw.columns, pd.MultiIndex) else raw['Close']
        return s['QQQ'] if isinstance(s, pd.DataFrame) else s
    except Exception:
        return None

def seccion_rendimiento(B):
    serie = B['serie']
    lineas = {'Tu renta variable': serie}
    if B['b'] is not None:
        lineas['SPY'] = B['b'] / B['b'].iloc[0]
    q = _qqq(PERIOD)
    if q is not None:
        q = q.reindex(serie.index).ffill().dropna()
        if len(q) > 60:
            lineas['QQQ (Nasdaq)'] = q / q.iloc[0]
    fig = go.Figure()
    for (n, s), c, d in zip(lineas.items(), ['#00d4ff', '#f87171', '#fbbf24'], [None, 'dash', 'dot']):
        fig.add_trace(go.Scatter(x=s.index, y=(s / s.iloc[0] - 1) * 100, name=n,
                                 line=dict(color=c, width=2.5 if d is None else 1.6, dash=d)))
    fig.add_hline(y=0, line_dash='dot', line_color='rgba(255,255,255,0.2)')
    fig.update_layout(yaxis_title='Rendimiento acumulado %')
    st.plotly_chart(estilo(fig, 400), use_container_width=True)

    filas = {}
    for n, s in lineas.items():
        m = metricas_serie(s, B['bench'], rf_rate, beta_freq)
        filas[n] = {'Rendimiento total': (s.iloc[-1] / s.iloc[0] - 1) * 100, 'Por año (CAGR)': m['CAGR'] * 100,
                    'Volatilidad': m['Volatilidad'] * 100, 'Beta': m['Beta'], 'Caída máxima': m['Máx Drawdown'] * 100,
                    'Sharpe': m['Sharpe'], 'Alfa por año': (m['Alfa'] * 100 if pd.notna(m['Alfa']) else np.nan)}
    tabla = pd.DataFrame(filas).T
    st.dataframe(tabla, use_container_width=True,
                 column_config={c: st.column_config.NumberColumn(format="%.1f%%") for c in
                                ['Rendimiento total', 'Por año (CAGR)', 'Volatilidad', 'Caída máxima', 'Alfa por año']}
                 | {'Beta': st.column_config.NumberColumn(format="%.2f",
                        help="1 = se mueve igual que el SPY. 1,3 = 30% más sensible al mercado."),
                    'Sharpe': st.column_config.NumberColumn(format="%.2f")})

    # Lectura en palabras
    if 'SPY' in tabla.index:
        yo, spy = tabla.loc['Tu renta variable'], tabla.loc['SPY']
        dif = yo['Por año (CAGR)'] - spy['Por año (CAGR)']
        txt = (f"**Tu renta variable rindió {yo['Por año (CAGR)']:.1f}% por año contra {spy['Por año (CAGR)']:.1f}% del SPY** "
               f"({'+' if dif >= 0 else ''}{dif:.1f} puntos por año). ")
        if pd.notna(yo['Beta']):
            b = yo['Beta']
            txt += (f"Pero lo hizo con una **beta de {b:.2f}**: es {abs(b-1)*100:.0f}% "
                    f"{'más' if b >= 1 else 'menos'} sensible al mercado que el SPY. ")
        if pd.notna(yo['Alfa por año']):
            a = yo['Alfa por año']
            txt += ("Ajustando por ese riesgo de mercado, " +
                    (f"**le ganaste al mercado por {a:.1f} puntos por año (alfa positiva)** 👍." if a > 0.5 else
                     f"**rendiste {abs(a):.1f} puntos por año menos de lo que correspondía por el riesgo que tomaste (alfa negativa)** — "
                     "o sea, tomaste más riesgo sin que te lo paguen." if a < -0.5 else
                     "**quedaste prácticamente en lo que correspondía por el riesgo que tomaste** (alfa cerca de cero)."))
        (st.success if pd.notna(yo['Alfa por año']) and yo['Alfa por año'] > 0.5 else
         st.warning if pd.notna(yo['Alfa por año']) and yo['Alfa por año'] < -0.5 else st.info)(txt)
    st.caption("Alfa = rendimiento real menos el rendimiento 'esperado' por tu beta (tasa libre de riesgo + beta × "
               "lo que rindió el SPY por encima de esa tasa). Es la forma de comparar dos carteras con distinto riesgo.")


# ─────────────────────────────────────────
#  5 · CORRELACIONES ALTAS
# ─────────────────────────────────────────
def seccion_correlacion(B):
    r = ret_frec(B['pc'])
    if r.shape[1] < 2:
        st.info("Hacen falta al menos 2 activos con datos.")
        return
    c = r.corr()
    sec = B['dfp'].set_index('Ticker')['Sector'].to_dict()
    peso = dict(zip(B['tks'], B['w'] * 100))
    pares = []
    cols = list(c.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = c.iloc[i, j]
            if pd.notna(v) and v >= 0.65:
                a, b = cols[i], cols[j]
                pares.append({'Activo 1': a, 'Activo 2': b, 'Correlación': round(float(v), 2),
                              'Nivel': '🔴 Muy alta' if v >= 0.80 else '🟠 Alta',
                              'Sector 1': sec.get(a, ''), 'Sector 2': sec.get(b, ''),
                              'Peso conjunto %': round(peso.get(a, 0) + peso.get(b, 0), 2)})
    c1, c2 = st.columns(2)
    n80 = sum(p['Correlación'] >= 0.80 for p in pares)
    c1.metric("🔴 Pares con correlación ≥ 0,80", n80,
              help="Se mueven casi igual: en la práctica es como tener más de lo mismo.")
    c2.metric("🟠 Pares entre 0,65 y 0,80", len(pares) - n80,
              help="Se mueven bastante juntos: diversifican poco entre sí.")
    if not pares:
        st.success("✅ Ningún par de activos supera 0,65 de correlación.")
    else:
        df = pd.DataFrame(pares).sort_values('Correlación', ascending=False)

        def pintar(fila):
            col = 'rgba(248,113,113,0.25)' if fila['Correlación'] >= 0.80 else 'rgba(251,146,60,0.22)'
            return [f'background-color: {col}'] * len(fila)
        st.dataframe(df.style.apply(pintar, axis=1).format({'Correlación': '{:.2f}', 'Peso conjunto %': '{:.2f}%'}),
                     hide_index=True, use_container_width=True, height=min(600, 38 + 35 * len(df)))
        st.caption("Solo aparecen los pares con correlación de 0,65 o más. *Peso conjunto* = cuánto suman los dos "
                   "dentro de tu renta variable: un par muy correlacionado que pesa mucho importa más que uno chico.")
    st.caption(f"Correlación con retornos {beta_freq.lower()}es en dólares, desde {B['pc'].index[0]:%m/%Y}. "
               "Va de −1 a 1: 1 = se mueven igual; 0 = no tienen relación.")


# ─────────────────────────────────────────
#  6 · CAÍDAS: REALES Y ESCENARIOS
# ─────────────────────────────────────────
def episodios_caida(s, n=3):
    dd = s / s.cummax() - 1
    eps, i, idx = [], 0, s.index
    vals = dd.values
    while i < len(vals):
        if vals[i] < 0:
            j = i
            while j < len(vals) and vals[j] < 0:
                j += 1
            seg = dd.iloc[i:j]
            fondo = seg.idxmin()
            pico = idx[i - 1] if i > 0 else idx[i]
            eps.append({'pico': pico, 'fondo': fondo, 'caida': float(seg.min()),
                        'recupero': idx[j] if j < len(vals) else None})
            i = j
        else:
            i += 1
    return sorted(eps, key=lambda e: e['caida'])[:n]

def seccion_caidas(B):
    s = B['serie']
    dd = (s / s.cummax() - 1) * 100
    fig = go.Figure(go.Scatter(x=dd.index, y=dd.values, fill='tozeroy', name='Tu renta variable',
                               line=dict(color='#f87171', width=1.5), fillcolor='rgba(248,113,113,0.15)'))
    if B['b'] is not None:
        dsp = (B['b'] / B['b'].cummax() - 1) * 100
        fig.add_trace(go.Scatter(x=dsp.index, y=dsp.values, name='SPY', line=dict(color='#94a3b8', width=1.2, dash='dot')))
    fig.update_layout(yaxis_title='% por debajo del máximo anterior')
    st.plotly_chart(estilo(fig, 320, 'Caídas reales desde cada máximo'), use_container_width=True)

    st.markdown("##### Tus 3 peores caídas reales y qué hizo el SPY en esas mismas fechas")
    filas = []
    for e in episodios_caida(s):
        spy = (B['b'].loc[e['fondo']] / B['b'].loc[e['pico']] - 1) * 100 if B['b'] is not None else np.nan
        filas.append({'Desde': e['pico'].strftime('%d/%m/%Y'), 'Hasta': e['fondo'].strftime('%d/%m/%Y'),
                      'Tu renta variable': round(e['caida'] * 100, 1), 'SPY mismo período': round(spy, 1),
                      'u$s de hoy': round(e['caida'] * B['total_rv'], 0),
                      'Recuperó': e['recupero'].strftime('%m/%Y') if e['recupero'] is not None else 'todavía no'})
    if filas:
        st.dataframe(pd.DataFrame(filas), hide_index=True, use_container_width=True,
                     column_config={'Tu renta variable': st.column_config.NumberColumn(format="%.1f%%"),
                                    'SPY mismo período': st.column_config.NumberColumn(format="%.1f%%"),
                                    'u$s de hoy': st.column_config.NumberColumn(format="%,.0f",
                                        help="Lo que significaría esa caída sobre lo que tenés hoy en renta variable.")})

    st.markdown("##### Escenarios hipotéticos")
    beta = calc_beta(s, B['bench'], beta_freq) if B['bench'] is not None else np.nan
    if pd.isna(beta):
        st.caption("Sin beta disponible.")
        return
    esc = [("Corrección leve", -0.05), ("Corrección moderada", -0.10), ("Mercado bajista", -0.20),
           ("Como COVID (feb-mar 2020)", -0.34), ("Como 2008 (Lehman)", -0.38)]
    df = pd.DataFrame([{'Si el SPY cae': f"{d*100:.0f}%", 'Escenario': n, 'Tu renta variable caería ≈': f"{d*beta*100:.1f}%",
                        'En u$s': f"u$s {abs(d*beta*B['total_rv']):,.0f}"} for n, d in esc])
    st.dataframe(df, hide_index=True, use_container_width=True)
    st.caption(f"Es una estimación simple: **caída del SPY × tu beta ({beta:.2f})**. No es historia real (eso es la "
               "tabla de arriba): sirve para dimensionar. En crisis las correlaciones suben y la caída real puede ser mayor.")


# ─────────────────────────────────────────
#  8 · MONTE CARLO
# ─────────────────────────────────────────
def seccion_montecarlo(B):
    st.caption("Simula 2.000 futuros posibles para tu renta variable, con la volatilidad real de tu cartera y el "
               "rendimiento anual que vos supongas. **No predice**: te da el rango razonable de resultados y la "
               "probabilidad de terminar abajo.")
    r_d = B['serie'].pct_change().dropna()
    ret_h, vol_h = float(r_d.mean() * 252), float(r_d.std() * np.sqrt(252))
    total = B['total_rv']
    c1, c2, c3 = st.columns(3)
    mu = c1.number_input("Rendimiento anual supuesto (%)", value=float(round(min(ret_h, 0.10) * 100, 1)),
        min_value=-30.0, max_value=60.0, step=0.5, key="mc_mu",
        help=f"Tu histórico fue {ret_h*100:.1f}% anual, pero proyectarlo tal cual suele ser optimista. "
             "Por defecto: el menor entre tu histórico y 10%.") / 100
    vol = c2.number_input("Volatilidad anual (%)", value=float(round(vol_h * 100, 1)), min_value=1.0,
        max_value=150.0, step=0.5, key="mc_vol", help="Por defecto, la volatilidad real de tu cartera.") / 100
    anios = c3.selectbox("Horizonte", [1, 2, 3, 5], index=0, key="mc_h", format_func=lambda x: f"{x} año{'s' if x > 1 else ''}")
    n_days = int(252 * anios)
    np.random.seed(42)
    paths = monte_carlo(mu, vol, total, 2000, n_days)
    fin = paths[:, -1]
    p5, p50, p95 = np.percentile(fin, [5, 50, 95])
    st.info(f"Con estos supuestos, dentro de **{anios} año{'s' if anios > 1 else ''}** tus u$s {total:,.0f} de renta "
            f"variable terminarían **entre u$s {p5:,.0f} y u$s {p95:,.0f}** en 9 de cada 10 casos "
            f"(lo más probable, alrededor de u$s {p50:,.0f}). Probabilidad de terminar con menos que hoy: "
            f"**{(fin < total).mean()*100:.0f}%**.")
    x = np.arange(n_days + 1)
    bandas = np.percentile(paths, [5, 50, 95], axis=0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=bandas[2], line=dict(width=0), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=x, y=bandas[0], fill='tonexty', fillcolor='rgba(0,212,255,0.15)',
                             line=dict(width=0), name='9 de cada 10 casos'))
    fig.add_trace(go.Scatter(x=x, y=bandas[1], line=dict(color='#00d4ff', width=2.5), name='Caso del medio'))
    fig.add_hline(y=total, line_dash='dash', line_color='#94a3b8', annotation_text='Hoy')
    fig.update_layout(xaxis_title='Días hábiles desde hoy', yaxis_title='u$s')
    st.plotly_chart(estilo(fig, 380), use_container_width=True)
    st.caption("⚠️ Supone que los rendimientos se reparten 'normalmente'. En las crisis reales las caídas suelen ser "
               "peores que el borde de abajo.")


# ─────────────────────────────────────────
#  9 · FRONTERA EFICIENTE (avanzado)
# ─────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def _nube(rets, rf, n_sim=2500):
    np.random.seed(42)
    n = rets.shape[1]
    out = []
    for _ in range(n_sim):
        w = np.random.dirichlet(np.ones(n))
        r, v, s = portfolio_metrics(w, rets, rf)
        out.append((v * 100, r * 100, s))
    return np.array(out)

def seccion_frontera(B):
    st.caption("Compara tu cartera con miles de combinaciones posibles de **los mismos activos**. Las carteras "
               "'óptimas' miran solo el pasado y suelen concentrarse en pocos activos que anduvieron bien: "
               "son una referencia, no una receta.")
    if not st.toggle("Calcular (tarda unos segundos)", key="fe_on"):
        return
    rets = B['rets']
    with st.spinner("Calculando…"):
        nube = _nube(rets, rf_rate)
        w_ms, w_mv = _optimas(rets, rf_rate)
    fig = go.Figure(go.Scatter(x=nube[:, 0], y=nube[:, 1], mode='markers', name='Combinaciones posibles',
                               marker=dict(size=4, color=nube[:, 2], colorscale='Viridis', opacity=0.5,
                                           colorbar=dict(title='Sharpe'))))
    res = []
    for n, w, col, sym in [('Tu cartera', B['w'], '#00d4ff', 'x'), ('Máx Sharpe', w_ms, '#f87171', 'star'),
                           ('Mín volatilidad', w_mv, '#4ade80', 'circle')]:
        r, v, s = portfolio_metrics(w, rets, rf_rate)
        fig.add_trace(go.Scatter(x=[v * 100], y=[r * 100], mode='markers+text', text=[n], textposition='top center',
                                 name=n, marker=dict(size=16, color=col, symbol=sym)))
        top = sorted(zip(B['tks'], w), key=lambda z: -z[1])[:6]
        res.append({'Cartera': n, 'Rendimiento anual %': round(r * 100, 1), 'Volatilidad %': round(v * 100, 1),
                    'Sharpe': round(s, 2), 'Principales pesos': ', '.join(f"{t} {p*100:.0f}%" for t, p in top if p > 0.005)})
    fig.update_layout(xaxis_title='Volatilidad anual %', yaxis_title='Rendimiento anual %')
    st.plotly_chart(estilo(fig, 480), use_container_width=True)
    st.dataframe(pd.DataFrame(res), hide_index=True, use_container_width=True)


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

def seccion_beta():
    st.caption("Mirá tu renta variable por sector, probá subir unos y bajar otros, y fijate qué pasa con la beta "
               "y la volatilidad. Al subir un sector se supone que comprás más de lo que ya tenés en él, en la misma proporción.")
    if st.session_state.portfolio.empty:
        st.info("Importá tu cartera primero (solapa 📥 Importar)")
    elif st.session_state.hist_data is None or st.session_state.get('bench_data') is None:
        st.warning("Faltan los precios de mercado.")
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
        if st.checkbox("➕ Agregar un sector nuevo (teórico)", key="sim_nuevo_on"):
            st.caption("Poné un nombre, el % que le querés asignar y uno o más tickers representativos separados por coma "
                       "(acciones o un ETF del sector). La app calcula su beta con datos reales.")
            nuevos_df = st.data_editor(
                st.session_state.get('sim_nuevos', pd.DataFrame({'Sector': [''], 'Tickers': [''], '% a asignar': [0.0]})),
                num_rows="dynamic", use_container_width=True, key="sim_nuevos_editor",
                column_config={'% a asignar': st.column_config.NumberColumn(format="%.2f", step=0.5),
                               'Tickers': st.column_config.TextColumn(help="Ej: COHR, AAOI")})
        else:
            nuevos_df = pd.DataFrame({'Sector': [], 'Tickers': [], '% a asignar': []})
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
#  PÁGINA
# ══════════════════════════════════════════
st.markdown("# 📊 Optimizador de Cartera")
hay_cartera = st.session_state.get('balanz_data') is not None and not st.session_state.portfolio.empty

with st.expander("📥 1 · Importar cartera" + (" — ✅ cargada" if hay_cartera else ""), expanded=not hay_cartera):
    seccion_importar()

if not hay_cartera:
    st.info("👆 Subí tu Excel para empezar. Al confirmar, la app baja sola los precios de mercado y arma todo el análisis.")
    st.stop()

asegurar_mercado()
bz0 = st.session_state.balanz_data
_tc0 = float(st.session_state.get('balanz_tipo_cambio') or 1)
c_a, c_b = st.columns([5, 1])
_hora = st.session_state.get('mercado_hora')
c_a.caption(f"Cartera: **{len(bz0)} instrumentos · u$s {bz0['V_Actual_Pesos'].sum()/_tc0:,.0f}** · renta variable "
            f"u$s {st.session_state.portfolio['Monto_USD'].sum():,.0f} · precios de mercado "
            f"{'al ' + _hora.strftime('%d/%m %H:%M') if _hora is not None else 'no disponibles'} ({years} años)")
if c_b.button("🔄 Actualizar precios", use_container_width=True):
    asegurar_mercado(forzar=True)
    st.rerun()
for _n in st.session_state.get('data_notas', []) or []:
    st.caption(_n)

with st.expander("🧭 2 · Composición de la cartera", expanded=True):
    seccion_composicion()

B = base_analisis()
if B is None:
    st.warning("No hay precios de mercado para analizar la renta variable. Probá con 🔄 Actualizar precios.")
    st.stop()
if B['info'].get('cortos') or B['info'].get('excluidos'):
    with st.expander("ℹ️ Qué activos quedan fuera del análisis histórico y por qué"):
        mostrar_aviso_ventana(B['info'])

with st.expander("🚦 3 · Semáforo de tu renta variable"):
    seccion_semaforo(B)
with st.expander("📈 4 · Rendimiento vs el mercado (con beta y riesgo)"):
    seccion_rendimiento(B)
with st.expander("🔗 5 · Correlaciones altas (≥ 0,65)"):
    seccion_correlacion(B)
with st.expander("📉 6 · Caídas: las reales y escenarios"):
    seccion_caidas(B)
with st.expander("🎚️ 7 · Simulador de beta por sector, subsector o país"):
    st.caption("La beta de cada grupo sale de **tus propios instrumentos** dentro de ese grupo, ponderados por lo que "
               "tenés en cada uno (no es la beta del sector o del país en general). Ej.: la beta de 'Argentina' es la "
               "de tu YPF, Pampa, Vista, etc., con tus montos.")
    seccion_beta()
with st.expander("🎲 8 · Monte Carlo: rango de resultados a futuro"):
    seccion_montecarlo(B)
with st.expander("🌐 9 · Frontera eficiente (avanzado)"):
    seccion_frontera(B)
