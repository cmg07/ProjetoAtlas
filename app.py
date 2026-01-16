# ATLAS | Porto Seguro B (Evolução Premium)
# 1 ativo por vez | Excel-driven | sem corretora (execução simulada)
# ---------------------------------------------------------------

from __future__ import annotations

import os
import re
import math
import time
import json
import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import streamlit as st
import plotly.graph_objects as go

# yfinance é a fonte principal (quando disponível)
import yfinance as yf


# =========================
# CONFIG / ESTILO
# =========================

APP_TITLE = "ATLAS"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
XLSX_PATH = os.path.join(DATA_DIR, "Ativos.xlsx")
LEDGER_PATH = os.path.join(os.path.dirname(__file__), ".atlas_ledger.csv")

# Para reduzir travas: timeouts curtos e menos tentativas no fetch
YF_TIMEOUT = 12  # segundos (aproximado por etapa)
MAX_ROWS_TABLE = 2000

# Paleta (dark premium, com destaque vermelho discreto)
CSS = r"""
<style>
/* Base */
html, body, [class*="css"]  {
  background: #07090d !important;
  color: #e8edf2 !important;
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji","Segoe UI Emoji" !important;
}

/* Remover “Made with Streamlit” / menu */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Sidebar */
section[data-testid="stSidebar"]{
  background: linear-gradient(180deg, #0b0f18, #07090d) !important;
  border-right: 1px solid rgba(255,255,255,0.06);
}
section[data-testid="stSidebar"] * { color: #e8edf2 !important; }

/* Títulos */
.atlas-title {
  font-size: 44px;
  font-weight: 900;
  letter-spacing: 2px;
  margin: 0 0 8px 0;
}
.atlas-sub {
  color: rgba(232,237,242,0.65);
  font-size: 12px;
  letter-spacing: 1px;
  margin-top: -6px;
}

/* Cards */
.atlas-card {
  background: radial-gradient(1200px 400px at 10% 10%, rgba(255,0,77,0.10), rgba(255,0,77,0.00)),
              linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 16px;
  padding: 14px 14px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.30);
}

/* KPI */
.kpi-label {
  color: rgba(232,237,242,0.70);
  font-size: 12px;
  letter-spacing: 1px;
  text-transform: uppercase;
  margin-bottom: 6px;
}
.kpi-value {
  font-size: 28px;
  font-weight: 900;
  letter-spacing: 0.5px;
  margin-bottom: 6px;
}
.kpi-hint {
  color: rgba(232,237,242,0.55);
  font-size: 12px;
  line-height: 1.3;
}

/* Badges */
.badge {
  display:inline-block;
  padding: 6px 10px;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.04);
  font-size: 12px;
  letter-spacing: 0.5px;
  margin-right: 8px;
}

/* Alerts */
.atlas-warn {
  border-left: 4px solid #ff004d;
  padding: 12px 12px;
  border-radius: 12px;
  background: rgba(255,0,77,0.10);
  border: 1px solid rgba(255,0,77,0.20);
}
.atlas-ok {
  border-left: 4px solid #00d084;
  padding: 12px 12px;
  border-radius: 12px;
  background: rgba(0,208,132,0.10);
  border: 1px solid rgba(0,208,132,0.18);
}

/* Tabs spacing */
.stTabs [data-baseweb="tab"] {
  font-weight: 700 !important;
}
</style>
"""


# =========================
# MODELOS
# =========================

@dataclass
class AssetRow:
    ticker: str
    name: str
    category: str
    source: str = "Excel"


def _norm_str(x: str) -> str:
    return re.sub(r"\s+", " ", str(x).strip())


def detect_asset_type(ticker: str, category: str) -> str:
    t = ticker.upper().strip()

    # Heurística por ticker
    if t.startswith("^"):
        return "ÍNDICE"
    if t.endswith("=X") or "/" in t:
        return "CÂMBIO"
    if t.endswith("-USD") or t.endswith("-BRL") or t.endswith("-USDT"):
        return "CRIPTO"
    if t.endswith(".SA"):
        return "AÇÃO BR"
    if t.endswith(".TO") or t.endswith(".L") or t.endswith(".F") or t.endswith(".DE"):
        return "AÇÃO GLOBAL"
    # Heurística por categoria
    c = category.upper()
    if "COMMOD" in c:
        return "COMMODITY"
    if "ETF" in c:
        return "ETF"
    if "AÇÃO" in c:
        return "AÇÃO"
    return "OUTROS"


def sanitize_ticker(raw: str, category: str) -> str:
    """
    Corrige o problema “PetrobrasPN” etc:
    - Sempre respeita o ticker já existente no Excel.
    - Só faz pequenos ajustes para formatos comuns.
    """
    if raw is None:
        return ""
    t = _norm_str(raw)

    # Se veio como "PETR4" em Ações Brasil, completa .SA
    if re.fullmatch(r"[A-Z]{4}\d{1,2}", t.upper()) and "BRASIL" in category.upper():
        return t.upper() + ".SA"

    # Câmbio do Excel pode estar como "BRL/COP" -> Yahoo costuma ser "COPBRL=X" OU "BRLxxx=X" (varia).
    # Aqui NÃO inventamos mapeamento: mantemos o que está no Excel.
    # O usuário já vai alimentar Ativos.xlsx com tickers válidos.
    return t


# =========================
# UNIVERSE (EXCEL)
# =========================

@st.cache_data(show_spinner=False)
def load_universe_from_excel(xlsx_path: str) -> pd.DataFrame:
    """
    Lê o Excel e cria um universo padronizado.
    Espera abas como: Índices, Ações Mundiais, Mercado de Câmbio, Commodities, Ações Brasil, Criptomoedas (etc).
    Em cada aba:
      Coluna A = ticker (ou par)
      Coluna B = nome/descrição (opcional)
    """
    if not os.path.exists(xlsx_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {xlsx_path}")

    # Pandas precisa de openpyxl
    try:
        xl = pd.ExcelFile(xlsx_path)
    except ImportError as e:
        raise ImportError("Falta 'openpyxl'. Instale com: pip install openpyxl") from e

    rows: List[AssetRow] = []
    for sheet in xl.sheet_names:
        df = xl.parse(sheet_name=sheet, header=None)
        if df.empty:
            continue

        # pega apenas 2 primeiras colunas
        a = df.iloc[:, 0].dropna().astype(str).map(_norm_str)
        b = pd.Series([""] * len(a))
        if df.shape[1] >= 2:
            b = df.iloc[: len(a), 1].fillna("").astype(str).map(_norm_str)

        # categoria = nome da aba, com normalização para UX
        category = normalize_category(sheet)

        for tick, name in zip(a.tolist(), b.tolist()):
            tick2 = sanitize_ticker(tick, category)
            if not tick2:
                continue
            nm = name if name else tick2
            rows.append(AssetRow(ticker=tick2, name=nm, category=category))

    uni = pd.DataFrame([r.__dict__ for r in rows])
    uni = uni.drop_duplicates(subset=["ticker", "category"]).reset_index(drop=True)

    # coluna "display" para seleção clara
    uni["display"] = uni["name"] + "  —  " + uni["ticker"]
    uni["type"] = [detect_asset_type(t, c) for t, c in zip(uni["ticker"], uni["category"])]

    # ordena por categoria e nome
    uni = uni.sort_values(["category", "name"]).reset_index(drop=True)
    return uni


def normalize_category(sheet_name: str) -> str:
    s = sheet_name.strip().lower()

    # padronização solicitada
    if "ações brasil" in s or "acoes brasil" in s:
        return "Ações Brasil"
    if "ações mundiais" in s or "acoes mundiais" in s or "ações américas" in s or "acoes americas" in s:
        return "Ações Globais"
    if "mercado de câmbio" in s or "mercado de cambio" in s or "câmbio" in s or "cambio" in s:
        return "Mercado de Câmbio"
    if "cripto" in s:
        return "Cripto"
    if "commod" in s:
        return "Commodities"
    if "índice" in s or "indice" in s:
        return "Índices"
    if "etf" in s:
        return "ETFs"

    # fallback: title-case
    return sheet_name.strip().title()


# =========================
# DADOS DE MERCADO
# =========================

def _hash_key(*parts: str) -> str:
    h = hashlib.md5()
    h.update("|".join(parts).encode("utf-8", errors="ignore"))
    return h.hexdigest()


@st.cache_data(show_spinner=False, ttl=60 * 30)
def fetch_ohlc_yf(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """
    Busca OHLC do Yahoo via yfinance.
    NÃO é chamado no boot: apenas quando usuário clicar "Analisar".
    Possui fallback leve.
    """
    # yfinance às vezes precisa "auto_adjust=False" para consistência
    df = yf.download(
        tickers=ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        threads=False,
        progress=False,
        group_by="column",
    )

    if df is None or len(df) == 0:
        return pd.DataFrame()

    # Se vier MultiIndex (quando ticker múltiplo), reduz
    if isinstance(df.columns, pd.MultiIndex):
        # tenta pegar a "primeira" coluna do primeiro nível
        # estrutura típica: ('Open','PETR4.SA')
        # seleciona pelo ticker se existir
        if ticker in df.columns.get_level_values(-1):
            df = df.xs(ticker, axis=1, level=-1)
        else:
            df = df.droplevel(-1, axis=1)

    # padroniza colunas
    df = df.rename(columns={c: c.title() for c in df.columns})
    keep = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    df = df[keep].copy()

    df.index = pd.to_datetime(df.index)
    df = df.dropna(subset=["Close"], how="any")

    return df


def fetch_ohlc_with_fallback(ticker: str, period: str, interval: str) -> Tuple[pd.DataFrame, Dict]:
    """
    Tenta (period, interval). Se falhar, tenta combinações "mais estáveis".
    Retorna df + debug.
    """
    attempts = []
    combos = [(period, interval)]

    # fallback mínimo, baseado no que você já viu dar certo
    if (period, interval) != ("2y", "1d"):
        combos.append(("2y", "1d"))
    if (period, interval) != ("1y", "1d"):
        combos.append(("1y", "1d"))
    if (period, interval) != ("6mo", "1d"):
        combos.append(("6mo", "1d"))

    used = None
    for p, i in combos:
        t0 = time.time()
        try:
            df = fetch_ohlc_yf(ticker, p, i)
        except Exception as e:
            attempts.append({"period": p, "interval": i, "ok": False, "error": str(e), "sec": round(time.time() - t0, 3)})
            continue

        ok = len(df) > 5 and "Close" in df.columns
        attempts.append({"period": p, "interval": i, "ok": ok, "rows": int(len(df)), "sec": round(time.time() - t0, 3)})
        if ok:
            used = {"period": p, "interval": i}
            return df, {"attempts": attempts, "used": used}

    return pd.DataFrame(), {"attempts": attempts, "used": None}


# =========================
# INDICADORES / TÉCNICO
# =========================

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["Return"] = out["Close"].pct_change()
    out["SMA20"] = out["Close"].rolling(20).mean()
    out["SMA50"] = out["Close"].rolling(50).mean()
    out["EMA20"] = out["Close"].ewm(span=20, adjust=False).mean()

    # STD / ZScore (garantir série 1D)
    out["STD20"] = out["Close"].rolling(20).std(ddof=0)
    z = (out["Close"] - out["SMA20"]) / out["STD20"]
    out["ZScore"] = pd.to_numeric(z, errors="coerce")

    # RSI 14
    delta = out["Close"].diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(gain, index=out.index).rolling(14).mean()
    roll_down = pd.Series(loss, index=out.index).rolling(14).mean()
    rs = roll_up / (roll_down + 1e-12)
    out["RSI14"] = 100 - (100 / (1 + rs))

    # ATR 14
    high = out["High"] if "High" in out.columns else out["Close"]
    low = out["Low"] if "Low" in out.columns else out["Close"]
    prev_close = out["Close"].shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["ATR14"] = tr.rolling(14).mean()

    # Bollinger 20/2
    out["BB_MID"] = out["SMA20"]
    out["BB_UP"] = out["SMA20"] + 2 * out["STD20"]
    out["BB_DN"] = out["SMA20"] - 2 * out["STD20"]

    # Vol anualizada (a partir de retornos)
    out["VolAnn"] = out["Return"].rolling(20).std(ddof=0) * np.sqrt(252)

    # Equity / Drawdown
    out["Equity"] = (1 + out["Return"].fillna(0)).cumprod()
    out["Peak"] = out["Equity"].cummax()
    out["Drawdown"] = out["Equity"] / out["Peak"] - 1

    return out


def regime_label(z: float, rsi: float, vol: float) -> Tuple[str, str]:
    """
    Retorna (regime, cor_sugerida) para UI.
    """
    if np.isnan(z) or np.isnan(rsi):
        return "Sem regime", "neutral"

    # extremos de zscore
    if z >= 2.0 and rsi >= 70:
        return "Sobrecomprado (extremo)", "warn"
    if z <= -2.0 and rsi <= 30:
        return "Sobrevendido (extremo)", "ok"

    if rsi >= 70:
        return "Sobrecomprado", "warn"
    if rsi <= 30:
        return "Sobrevendido", "ok"

    # vol alta
    if not np.isnan(vol) and vol >= 0.45:
        return "Volatilidade hostil", "warn"

    return "Neutro/Transição", "neutral"


def decision_translation(z: float, rsi: float, dd: float, cost_drag_bps: float) -> Dict[str, str]:
    """
    “Tradução” do que o sistema está dizendo (não compre pq / compre pq).
    """
    bullets_no = []
    bullets_yes = []

    # Motivos de defesa
    if np.isnan(z) or np.isnan(rsi):
        bullets_no.append("Dados insuficientes para sinal técnico confiável.")
    else:
        if z >= 2.0:
            bullets_no.append("Preço distante da média (Z-score alto): risco de mean-reversion contra a entrada.")
        if rsi >= 70:
            bullets_no.append("RSI alto: pressão compradora pode estar saturada.")
        if dd <= -0.12:
            bullets_no.append("Drawdown relevante: mercado pode estar em fase de stress/retomada instável.")

    if cost_drag_bps >= 15:
        bullets_no.append("Custos (spread/slippage) altos: você paga para comprar incerteza.")

    # Motivos de ataque
    if not np.isnan(z) and z <= -1.5:
        bullets_yes.append("Preço abaixo do 'fair zone' estatístico: potencial de retorno por reversão.")
    if not np.isnan(rsi) and rsi <= 35:
        bullets_yes.append("RSI baixo: probabilidade de alívio técnico aumenta.")
    if dd > -0.08:
        bullets_yes.append("Drawdown controlado: regime mais estável para operar.")

    if not bullets_yes:
        bullets_yes.append("Sem edge claro agora: aguarde melhor assimetria (trigger incompleto).")

    if not bullets_no:
        bullets_no.append("Nada crítico contra — mas exija gatilho (setup) e controle de risco.")

    return {
        "nao_compre": " • " + "\n • ".join(bullets_no),
        "compre": " • " + "\n • ".join(bullets_yes),
    }


# =========================
# RISCO / RUÍNA / CUSTOS
# =========================

def risk_of_ruin_mc(win_rate: float, payoff: float, risk_per_trade: float, ruin_limit: float, horizon: int, n_paths: int = 2000) -> float:
    """
    Simulação simples: capital inicial=1.0, arrisca RPT por trade.
    - win => +payoff*RPT
    - loss => -RPT
    Ruína ocorre ao cair abaixo de (1 - ruin_limit).
    """
    if horizon <= 0 or n_paths <= 0:
        return float("nan")

    rng = np.random.default_rng(42)
    cap0 = 1.0
    ruin_threshold = cap0 * (1.0 - ruin_limit)

    ruined = 0
    for _ in range(n_paths):
        cap = cap0
        for _t in range(horizon):
            if rng.random() < win_rate:
                cap *= (1.0 + payoff * risk_per_trade)
            else:
                cap *= (1.0 - risk_per_trade)
            if cap <= ruin_threshold:
                ruined += 1
                break

    return ruined / n_paths


def cost_drag_bps(spread_bps: float, slippage_bps: float, commission_bps: float) -> float:
    return float(spread_bps + slippage_bps + commission_bps)


# =========================
# FUNDAMENTALISTA (YAHOO)
# =========================

@st.cache_data(show_spinner=False, ttl=60 * 60)
def fetch_fundamentals_yf(ticker: str) -> Dict:
    """
    Busca informações fundamentalistas via yfinance.Ticker.
    Pode falhar por cobertura/região — tratamos isso no caller.
    """
    t = yf.Ticker(ticker)
    info = {}
    try:
        info = t.get_info()
    except Exception:
        # mantém vazio
        info = {}

    # tabelas (podem vir vazias)
    def _safe_df(getter):
        try:
            df = getter()
            if df is None:
                return pd.DataFrame()
            if isinstance(df, pd.DataFrame):
                return df
            return pd.DataFrame(df)
        except Exception:
            return pd.DataFrame()

    financials = _safe_df(lambda: t.financials)
    balance = _safe_df(lambda: t.balance_sheet)
    cashflow = _safe_df(lambda: t.cashflow)

    # tentativa de news (instável)
    news = []
    try:
        news = t.get_news()
    except Exception:
        news = []

    return {
        "info": info or {},
        "financials": financials,
        "balance": balance,
        "cashflow": cashflow,
        "news": news or [],
    }


# =========================
# CONTABILIDADE (UPLOAD / PREPARO)
# =========================

def parse_uploaded_statement(file) -> pd.DataFrame:
    if file is None:
        return pd.DataFrame()
    name = getattr(file, "name", "").lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(file)
    return pd.DataFrame()


# =========================
# PLOT
# =========================

def plot_price(df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()

    # Candles se tiver OHLC
    if {"Open", "High", "Low", "Close"}.issubset(df.columns):
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="Preço",
            )
        )
    else:
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Preço", mode="lines"))

    # overlays
    for col, nm in [("SMA20", "SMA20"), ("SMA50", "SMA50"), ("BB_UP", "BB+2σ"), ("BB_DN", "BB-2σ")]:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], name=nm, mode="lines"))

    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=520,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(rangeslider_visible=False)
    return fig


def plot_rsi(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI14"], name="RSI14", mode="lines"))
    fig.add_hline(y=70, line_width=1)
    fig.add_hline(y=30, line_width=1)
    fig.update_layout(template="plotly_dark", height=240, margin=dict(l=10, r=10, t=40, b=10), title="RSI (14)")
    return fig


def plot_drawdown(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Drawdown"], name="Drawdown", mode="lines"))
    fig.update_layout(template="plotly_dark", height=240, margin=dict(l=10, r=10, t=40, b=10), title="Drawdown")
    return fig


def plot_equity(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Equity"], name="Equity", mode="lines"))
    fig.update_layout(template="plotly_dark", height=240, margin=dict(l=10, r=10, t=40, b=10), title="Retorno cumulativo")
    return fig


def plot_vol(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["VolAnn"], name="Vol anual", mode="lines"))
    fig.update_layout(template="plotly_dark", height=240, margin=dict(l=10, r=10, t=40, b=10), title="Volatilidade anualizada")
    return fig


# =========================
# LEDGER (Simulado)
# =========================

def ledger_init():
    if not os.path.exists(LEDGER_PATH):
        pd.DataFrame(columns=["ts", "ticker", "side", "qty", "price", "note"]).to_csv(LEDGER_PATH, index=False)


def ledger_append(ticker: str, side: str, qty: float, price: float, note: str = ""):
    ledger_init()
    row = {
        "ts": datetime.utcnow().isoformat(),
        "ticker": ticker,
        "side": side,
        "qty": float(qty),
        "price": float(price),
        "note": note,
    }
    df = pd.read_csv(LEDGER_PATH)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(LEDGER_PATH, index=False)


def ledger_read() -> pd.DataFrame:
    ledger_init()
    try:
        return pd.read_csv(LEDGER_PATH)
    except Exception:
        return pd.DataFrame(columns=["ts", "ticker", "side", "qty", "price", "note"])


# =========================
# UI HELPERS
# =========================

def kpi_card(label: str, value: str, hint: str) -> str:
    return f"""
    <div class="atlas-card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-hint">{hint}</div>
    </div>
    """


def top_header(asset_name: str, ticker: str, category: str, last_price: Optional[float], z: Optional[float], rsi: Optional[float], vol: Optional[float], regime: str):
    st.markdown(CSS, unsafe_allow_html=True)

    st.markdown(f'<div class="atlas-title">{APP_TITLE}</div>', unsafe_allow_html=True)

    # badges (sem o "boot: ...")
    b1 = f'<span class="badge">ATIVO</span> {asset_name}'
    b2 = f'<span class="badge">TICKER</span> {ticker}'
    b3 = f'<span class="badge">CATEGORIA</span> {category}'
    st.markdown(f"{b1} &nbsp;&nbsp; {b2} &nbsp;&nbsp; {b3}", unsafe_allow_html=True)

    cols = st.columns(5)
    cols[0].markdown(kpi_card("Preço Atual", "-" if last_price is None else f"{last_price:,.4f}", "Último fechamento (ou último candle)."), unsafe_allow_html=True)
    cols[1].markdown(kpi_card("Z-Score", "-" if z is None or np.isnan(z) else f"{z:,.2f}", "Distância estatística da SMA20 em desvios-padrão."), unsafe_allow_html=True)
    cols[2].markdown(kpi_card("RSI14", "-" if rsi is None or np.isnan(rsi) else f"{rsi:,.1f}", "Momento/pressão: >70 sobrecomprado, <30 sobrevendido."), unsafe_allow_html=True)
    cols[3].markdown(kpi_card("Vol (Anual)", "-" if vol is None or np.isnan(vol) else f"{100*vol:,.1f}%", "Ruído do ativo. Vol alta exige stops mais largos."), unsafe_allow_html=True)
    cols[4].markdown(kpi_card("Regime", regime, "Leitura rápida do estado do mercado."), unsafe_allow_html=True)


def sidebar_controls(universe: pd.DataFrame) -> Tuple[str, str, str, str]:
    st.sidebar.markdown("## Seleção de Ativo (1 por vez)")

    cats = universe["category"].unique().tolist() if not universe.empty else []
    if not cats:
        st.sidebar.error("Nenhuma categoria encontrada no Ativos.xlsx.")
        return "", "", "2y", "1d"

    category = st.sidebar.selectbox("Categoria", cats, index=0)

    sub = universe[universe["category"] == category].copy()
    opts = sub["display"].tolist()
    if not opts:
        st.sidebar.error("Categoria vazia no Excel.")
        return category, "", "2y", "1d"

    display = st.sidebar.selectbox("Ativo", opts, index=0)
    row = sub[sub["display"] == display].iloc[0]
    ticker = row["ticker"]

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Parâmetros de Dados")
    period = st.sidebar.selectbox("Período", ["7d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"], index=5, help="Janela histórica para o cálculo de indicadores.")
    interval = st.sidebar.selectbox("Intervalo", ["1m","2m","5m","15m","30m","60m","90m","1h","1d","5d","1wk","1mo"], index=8, help="Granularidade do candle. Recomendo 1d para estabilidade.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Risco & Custos")
    with st.sidebar.expander("Modelagem (didática)", expanded=False):
        st.markdown(
            "- **Win Rate**: % de trades vencedores.\n"
            "- **Payoff (R)**: ganho médio em múltiplos do risco.\n"
            "- **Risk-per-trade**: fração do capital arriscada por trade.\n"
            "- **Ruína**: perda máxima tolerada (ex.: 30%).\n"
            "- **Custos**: spread+slippage+comissão em bps (impactam seu edge)."
        )

    return category, ticker, period, interval


# =========================
# SUB-ESFERAS (por tipo)
# =========================

def math_subsphere(asset_type: str, df: pd.DataFrame) -> None:
    st.markdown("### Sub-esfera Matemática (por tipo de ativo)")
    if df.empty:
        st.info("Carregue dados para ativar esta camada.")
        return

    last = df.iloc[-1]
    z = float(last.get("ZScore", np.nan))
    rsi = float(last.get("RSI14", np.nan))
    vol = float(last.get("VolAnn", np.nan))
    atr = float(last.get("ATR14", np.nan))
    dd = float(last.get("Drawdown", np.nan))

    if asset_type in ["AÇÃO BR", "AÇÃO GLOBAL", "AÇÃO"]:
        st.markdown(
            "- **Microestrutura**: tendência vs. reversão (Z-score) + pressão (RSI).\n"
            "- **Risco**: ATR como proxy de stop mínimo; vol anual para sizing.\n"
            "- **Qualidade**: drawdown te diz o regime de stress."
        )
        st.code(
            "Stop técnico (exemplo):\n"
            "  stop = close - k * ATR14  (k ~ 1.5 a 3.0 dependendo da vol)\n"
            "Sizing (exemplo):\n"
            "  position = (capital * risk_per_trade) / (stop_distance)",
            language="text",
        )
    elif asset_type in ["CÂMBIO"]:
        st.markdown(
            "- **FX** tende a regimes longos + choques macro.\n"
            "- Use **vol/ATR** para proteger contra gaps.\n"
            "- Em FX, **custos (spread)** pesam mais: trate custo como 'imposto de entrada'."
        )
    elif asset_type in ["CRIPTO"]:
        st.markdown(
            "- Cripto tem **caudas** (risco de gap) e **vol estrutural alta**.\n"
            "- Z-score e RSI funcionam bem para extremos, mas exija filtro de tendência.\n"
            "- Custos + slippage podem explodir em candle curto."
        )
    elif asset_type in ["COMMODITY"]:
        st.markdown(
            "- Commodities são muito sensíveis a oferta/demanda e eventos.\n"
            "- ATR e drawdown guiam agressividade.\n"
            "- Z-score alto pode persistir em ciclos."
        )
    elif asset_type in ["ÍNDICE", "ETF"]:
        st.markdown(
            "- Índices/ETFs: tendência costuma ser mais limpa que ações individuais.\n"
            "- Z-score e SMA20/50 ajudam a filtrar regime.\n"
            "- Drawdown é excelente para mapear stress."
        )
    else:
        st.markdown(
            "- Ativo fora do cluster padrão.\n"
            "- Use: Z-score (reversão), RSI (pressão), vol/ATR (risco), drawdown (regime)."
        )

    st.markdown("---")
    st.markdown("**Snapshot matemático**")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Z", f"{z:.2f}" if not np.isnan(z) else "-")
    c2.metric("RSI", f"{rsi:.1f}" if not np.isnan(rsi) else "-")
    c3.metric("Vol(ann)", f"{100*vol:.1f}%" if not np.isnan(vol) else "-")
    c4.metric("ATR14", f"{atr:.4f}" if not np.isnan(atr) else "-")
    c5.metric("DD", f"{100*dd:.1f}%" if not np.isnan(dd) else "-")


# =========================
# MAIN
# =========================

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")

    # load universe (boot rápido)
    try:
        uni = load_universe_from_excel(XLSX_PATH)
    except Exception as e:
        st.markdown(CSS, unsafe_allow_html=True)
        st.error(f"Falha ao carregar Ativos.xlsx: {e}")
        st.info("Confirme o caminho: data/Ativos.xlsx e instale openpyxl: pip install openpyxl")
        return

    category, ticker, period, interval = sidebar_controls(uni)
    if not ticker:
        st.markdown(CSS, unsafe_allow_html=True)
        st.warning("Selecione um ativo.")
        return

    row = uni[(uni["ticker"] == ticker) & (uni["category"] == category)].iloc[0]
    asset_name = row["name"]
    asset_type = row["type"]

    # Botão de análise (não fetch no boot)
    st.sidebar.markdown("---")
    analyze = st.sidebar.button("Analisar ativo", type="primary", use_container_width=True)

    # estado
    if "analysis" not in st.session_state:
        st.session_state["analysis"] = {}

    if analyze or st.session_state["analysis"].get("ticker") != ticker or st.session_state["analysis"].get("period") != period or st.session_state["analysis"].get("interval") != interval:
        with st.spinner("Carregando dados e calculando indicadores..."):
            df_raw, debug = fetch_ohlc_with_fallback(ticker, period, interval)
            if df_raw.empty:
                st.session_state["analysis"] = {
                    "ticker": ticker,
                    "period": period,
                    "interval": interval,
                    "df": pd.DataFrame(),
                    "debug": debug,
                    "computed": False,
                }
            else:
                df = compute_indicators(df_raw)
                st.session_state["analysis"] = {
                    "ticker": ticker,
                    "period": period,
                    "interval": interval,
                    "df": df,
                    "debug": debug,
                    "computed": True,
                }

    data = st.session_state["analysis"]
    df = data.get("df", pd.DataFrame())
    debug = data.get("debug", {})
    computed = bool(data.get("computed", False))

    # KPIs / Header
    if computed and not df.empty:
        last = df.iloc[-1]
        last_price = float(last.get("Close", np.nan))
        z = float(last.get("ZScore", np.nan))
        rsi = float(last.get("RSI14", np.nan))
        vol = float(last.get("VolAnn", np.nan))
        regime, _ = regime_label(z, rsi, vol)
    else:
        last_price = None
        z = None
        rsi = None
        vol = None
        regime = "Sem dados"

    top_header(asset_name, ticker, category, last_price, z, rsi, vol, regime)

    # Tabs
    tabs = st.tabs(["Painel", "Técnico", "Risco & Custos", "Fundamentalista", "Contabilidade", "Macro & News", "Execução (Sim)", "Ledger", "Admin"])

    # =========================
    # PAINEL
    # =========================
    with tabs[0]:
        if not computed or df.empty:
            st.markdown('<div class="atlas-warn"><b>Sem dados retornados</b> para esse ativo/período/intervalo (mesmo com fallback).<br>'
                        'Sugestão: tente <b>2y + 1d</b>.<br>'
                        'Se o ticker for BR e não existir no Yahoo, ajuste o ticker no <b>Ativos.xlsx</b>.</div>', unsafe_allow_html=True)
            if debug:
                with st.expander("Debug (tentativas de download)"):
                    st.json(debug)
            return

        dd = float(df["Drawdown"].iloc[-1])
        # custos
        st.markdown("### Tradução da decisão (operacional)")
        c1, c2, c3 = st.columns([1.2, 1.2, 1.0])

        with c3:
            st.markdown("**Custos** (impactam seu edge)")
            spread = st.number_input("Spread (bps)", min_value=0.0, max_value=200.0, value=8.0, step=1.0, help="Custo médio de entrada/saída. Em FX/cripto pode ser maior em stress.")
            slippage = st.number_input("Slippage (bps)", min_value=0.0, max_value=200.0, value=5.0, step=1.0, help="Pior execução do que o preço teórico.")
            comm = st.number_input("Comissão (bps)", min_value=0.0, max_value=200.0, value=0.0, step=1.0, help="Se sua corretora cobra por ordem.")
            drag = cost_drag_bps(spread, slippage, comm)
            st.metric("Custo total (bps)", f"{drag:.1f}")

        trans = decision_translation(float(df["ZScore"].iloc[-1]), float(df["RSI14"].iloc[-1]), dd, drag)

        with c1:
            st.markdown('<div class="atlas-warn"><b>NÃO COMPRE AGORA (se)</b><br><br>' + trans["nao_compre"].replace("\n", "<br>") + "</div>", unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="atlas-ok"><b>COMPRE/OPERE (se)</b><br><br>' + trans["compre"].replace("\n", "<br>") + "</div>", unsafe_allow_html=True)

        st.markdown("---")
        st.plotly_chart(plot_price(df, f"{ticker} | Preço + Médias + Bollinger"), use_container_width=True)

        cA, cB = st.columns(2)
        with cA:
            st.plotly_chart(plot_rsi(df), use_container_width=True)
            st.plotly_chart(plot_vol(df), use_container_width=True)
        with cB:
            st.plotly_chart(plot_equity(df), use_container_width=True)
            st.plotly_chart(plot_drawdown(df), use_container_width=True)

        st.markdown("---")
        math_subsphere(asset_type, df)

    # =========================
    # TÉCNICO
    # =========================
    with tabs[1]:
        if df.empty:
            st.info("Sem dados. Clique em **Analisar ativo**.")
        else:
            st.subheader("Camada Técnica (abissal, mas prática)")
            st.markdown(
                "- **Z-score** te diz quão esticado o preço está vs. média.\n"
                "- **RSI** mede pressão (não é gatilho sozinho).\n"
                "- **ATR/Vol** ditam stop e tamanho.\n"
                "- **Drawdown** é o termômetro do regime."
            )
            st.plotly_chart(plot_price(df, f"{ticker} | Técnico (candles + overlays)"), use_container_width=True)
            st.dataframe(df.tail(200), use_container_width=True, height=420)

    # =========================
    # RISCO & CUSTOS
    # =========================
    with tabs[2]:
        st.subheader("Risco, custos e robustez do setup")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            win = st.slider("Win Rate estimado", 0.10, 0.90, 0.50, 0.01)
        with c2:
            payoff = st.slider("Payoff médio (R)", 0.50, 5.00, 1.80, 0.05)
        with c3:
            rpt = st.slider("Risk-per-Trade", 0.001, 0.05, 0.01, 0.001)
        with c4:
            ruin = st.slider("Limite de ruína (perda do capital)", 0.10, 0.90, 0.30, 0.01)

        horizon = st.slider("Horizonte (trades)", 50, 2000, 400, 10)

        spread = st.number_input("Spread (bps)", min_value=0.0, max_value=200.0, value=8.0, step=1.0)
        slippage = st.number_input("Slippage (bps)", min_value=0.0, max_value=200.0, value=5.0, step=1.0)
        comm = st.number_input("Comissão (bps)", min_value=0.0, max_value=200.0, value=0.0, step=1.0)
        drag = cost_drag_bps(spread, slippage, comm)

        st.markdown("---")
        rr = risk_of_ruin_mc(win, payoff, rpt, ruin, horizon, n_paths=2500)

        cA, cB, cC = st.columns(3)
        cA.metric("Risco de Ruína (MC)", f"{100*rr:.1f}%")
        cB.metric("Custo total (bps)", f"{drag:.1f}")
        cC.metric("Edge mínimo p/ empatar", f"{drag:.1f} bps")

        st.info(
            "Leitura: se **ruína** estiver alta, seu sistema (win/payoff/rpt) é frágil. "
            "Se **custos** estiverem altos, você precisa de edge maior para não operar no zero."
        )

    # =========================
    # FUNDAMENTALISTA
    # =========================
    with tabs[3]:
        st.subheader("Fundamentalista (Yahoo)")
        st.caption("Se o ativo não tiver cobertura (comum em BR/alguns tickers), esta camada pode retornar vazio.")

        if st.button("Carregar Fundamentalista (Yahoo)", type="primary"):
            with st.spinner("Buscando fundamentalista..."):
                try:
                    fund = fetch_fundamentals_yf(ticker)
                    st.session_state["fund"] = fund
                except Exception as e:
                    st.session_state["fund"] = {"error": str(e)}

        fund = st.session_state.get("fund", None)
        if not fund:
            st.info("Clique no botão para carregar.")
        else:
            if "error" in fund:
                st.error(fund["error"])
            info = fund.get("info", {}) or {}
            c1, c2, c3 = st.columns(3)
            c1.metric("Market Cap", f"{info.get('marketCap','-')}")
            c2.metric("P/E", f"{info.get('trailingPE','-')}")
            c3.metric("Dividend Yield", f"{info.get('dividendYield','-')}")

            with st.expander("Info (raw)"):
                st.json({k: info.get(k) for k in ["shortName","longName","sector","industry","country","currency","exchange","quoteType","website"]})

            st.markdown("### Demonstrações (se disponíveis)")
            fin = fund.get("financials", pd.DataFrame())
            bal = fund.get("balance", pd.DataFrame())
            cf = fund.get("cashflow", pd.DataFrame())

            cols = st.columns(3)
            with cols[0]:
                st.markdown("**Income Statement**")
                st.dataframe(fin if not fin.empty else pd.DataFrame({"status":["Sem dados"]}), use_container_width=True, height=360)
            with cols[1]:
                st.markdown("**Balance Sheet**")
                st.dataframe(bal if not bal.empty else pd.DataFrame({"status":["Sem dados"]}), use_container_width=True, height=360)
            with cols[2]:
                st.markdown("**Cash Flow**")
                st.dataframe(cf if not cf.empty else pd.DataFrame({"status":["Sem dados"]}), use_container_width=True, height=360)

    # =========================
    # CONTABILIDADE
    # =========================
    with tabs[4]:
        st.subheader("Contabilidade (módulo pronto para demonstrativos públicos)")
        st.caption("Como o Brasil tem lacunas no Yahoo, esta camada fica preparada para fontes CVM/B3/Investing. Hoje: upload de demonstrativos.")

        f = st.file_uploader("Upload de demonstrativo (CSV ou Excel)", type=["csv", "xlsx", "xls"])
        df_stmt = parse_uploaded_statement(f)
        if df_stmt.empty:
            st.info("Envie um arquivo para visualizar e habilitar métricas contábeis.")
            st.markdown(
                "**Sugestão de layout mínimo**:\n"
                "- Data/Período\n"
                "- Receita, Lucro Líquido\n"
                "- Ativo Total, Passivo Total, Patrimônio\n"
                "- Caixa Operacional, Caixa Livre\n"
            )
        else:
            st.success("Demonstrativo carregado.")
            st.dataframe(df_stmt.head(200), use_container_width=True, height=420)

            st.markdown("### Métricas contábeis (auto-detect, best effort)")
            cols = [c.lower() for c in df_stmt.columns]
            st.caption("Se as colunas não baterem, renomeie no Excel/CSV e reenviar.")

    # =========================
    # MACRO & NEWS
    # =========================
    with tabs[5]:
        st.subheader("Macro & News (links confiáveis)")
        st.caption("Como a API de notícias do Yahoo é instável, usamos hubs globais. Depois a gente pluga um agregador robusto (RSS/NewsAPI).")

        links = [
            ("Reuters", "https://www.reuters.com/"),
            ("Bloomberg", "https://www.bloomberg.com/"),
            ("Financial Times", "https://www.ft.com/"),
            ("WSJ", "https://www.wsj.com/"),
            ("The Economist", "https://www.economist.com/"),
            ("CNBC", "https://www.cnbc.com/"),
            ("Investing.com", "https://www.investing.com/"),
            ("Valor Econômico (BR)", "https://valor.globo.com/"),
            ("InfoMoney (BR)", "https://www.infomoney.com.br/"),
            ("Brazil Journal (BR)", "https://braziljournal.com/"),
        ]
        st.markdown("### Portais")
        for name, url in links:
            st.markdown(f"- [{name}]({url})")

        # tentativa opcional de Yahoo news
        if st.toggle("Tentar notícias do Yahoo (opcional)", value=False):
            with st.spinner("Tentando Yahoo news..."):
                try:
                    t = yf.Ticker(ticker)
                    news = t.get_news()
                except Exception:
                    news = []
            if not news:
                st.warning("Sem notícias retornadas pelo Yahoo para este ticker.")
            else:
                st.markdown("### Yahoo (se disponível)")
                for n in news[:12]:
                    title = n.get("title", "Sem título")
                    link = n.get("link", "")
                    pub = n.get("publisher", "")
                    st.markdown(f"- [{title}]({link})  \n  _{pub}_")

    # =========================
    # EXECUÇÃO (SIM)
    # =========================
    with tabs[6]:
        st.subheader("Execução (Simulada) — pronto para corretora (futuro)")
        st.caption("Aqui ficam os blocos de BUY/SELL/STOP/TAKE e roteamento. Por enquanto, logamos no Ledger.")

        if df.empty:
            st.info("Carregue dados para habilitar execução simulada.")
        else:
            last_px = float(df["Close"].iloc[-1])
            c1, c2, c3 = st.columns([1, 1, 2])
            with c1:
                qty = st.number_input("Quantidade (sim)", min_value=0.0, value=1.0, step=1.0)
            with c2:
                side = st.selectbox("Lado", ["BUY", "SELL"])
            with c3:
                note = st.text_input("Nota", value="Simulação ATLAS")

            if st.button("Registrar ordem no Ledger (sim)", type="primary"):
                ledger_append(ticker, side, qty, last_px, note)
                st.success("Ordem registrada.")

            st.markdown("### Blocos prontos (futuro corretora)")
            st.code(
                "BUY / SELL\n"
                "STOP (fixo / ATR / estrutura)\n"
                "TAKE PROFIT (RR alvo)\n"
                "TRAILING STOP\n"
                "Limites de custo (spread/slippage) antes de executar\n"
                "Kill-switch por ruína/regime",
                language="text",
            )

    # =========================
    # LEDGER
    # =========================
    with tabs[7]:
        st.subheader("Ledger (sim)")
        df_led = ledger_read()
        st.dataframe(df_led.tail(300), use_container_width=True, height=420)

    # =========================
    # ADMIN
    # =========================
    with tabs[8]:
        st.subheader("Admin / Diagnóstico")
        st.markdown("### Universo")
        st.write(f"Total de ativos: **{len(uni)}**")
        st.dataframe(uni.head(200), use_container_width=True, height=420)

        st.markdown("### Debug de dados")
        if debug:
            st.json(debug)
        else:
            st.write("-")

        st.markdown("### Checagens rápidas")
        st.code(
            f"Ativos.xlsx: {XLSX_PATH}\n"
            f"Existe? {os.path.exists(XLSX_PATH)}\n"
            f"Data dir: {DATA_DIR}\n"
            f"Ledger: {LEDGER_PATH}",
            language="text",
        )


if __name__ == "__main__":
    main()
