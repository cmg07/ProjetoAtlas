# ============================================================
# ATLAS | SUPREMO (1-Ativo-Por-Vez) | BOOT instantâneo
# - Sem corretora (última barreira)
# - Técnica + Microestrutura (spread/slippage) + Risco + Fundamental
# - Macro Pulse (proxies Yahoo) + Notícias (Yahoo + Google News RSS)
# - Backtest com custos (spread + slippage + comissão)
#
# Requisitos (no seu requirements já tem quase tudo):
# streamlit, yfinance, pandas, numpy, plotly, requests, beautifulsoup4
# ============================================================

from __future__ import annotations

import os
import re
import time
import math
import json
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import requests
from bs4 import BeautifulSoup

warnings.filterwarnings("ignore")
st.set_page_config(layout="wide", page_title="ATLAS", initial_sidebar_state="expanded")

# ---------------------------
# VISUAL (CSS)
# ---------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@300;400;600;800&display=swap');

:root{
  --bg:#020202;
  --panel:#070707;
  --panel2:#0b0b0b;
  --text:#e5e7eb;
  --muted:#9ca3af;
  --red:#ff0000;
  --green:#00ff41;
  --amber:#fbbf24;
  --line:#111827;
}

html, body, [class*="css"]  {
  font-family: 'Inter', sans-serif;
  background: var(--bg) !important;
  color: var(--text) !important;
}
.stApp { background: var(--bg); }
section[data-testid="stSidebar"]{
  background: linear-gradient(180deg, #050505 0%, #0a0a0a 100%) !important;
  border-right: 1px solid #0f0f0f !important;
}
.atlas-header{
  background: radial-gradient(1200px 240px at 18% 30%, rgba(255,0,0,0.18), transparent 55%),
              linear-gradient(135deg, #030303 0%, #0b0b0b 100%);
  border-left: 10px solid var(--red);
  border-radius: 14px;
  padding: 22px 26px;
  margin: 10px 0 14px 0;
  box-shadow: 0 0 0 1px rgba(255,255,255,0.04), 0 8px 30px rgba(0,0,0,0.6);
}
.atlas-title{
  font-size: 42px; font-weight: 800; margin: 0; color: #ffffff;
}
.atlas-sub{
  margin-top: 10px;
  font-family: 'JetBrains Mono', monospace;
  color: var(--muted);
  font-size: 12px;
}
.badge{
  display:inline-block;
  font-family:'JetBrains Mono', monospace;
  font-size: 11px;
  padding: 3px 8px;
  border-radius: 999px;
  margin-right: 8px;
  border:1px solid rgba(255,255,255,0.08);
  background: rgba(255,255,255,0.03);
}
.badge-ok{ color: var(--green); border-color: rgba(0,255,65,0.25); }
.badge-warn{ color: var(--red); border-color: rgba(255,0,0,0.25); }
.badge-amb{ color: var(--amber); border-color: rgba(251,191,36,0.25); }

.section-title{
  font-family:'JetBrains Mono', monospace;
  text-transform: uppercase;
  letter-spacing: 7px;
  font-weight: 800;
  color: var(--red);
  border-bottom: 2px solid var(--red);
  padding-bottom: 10px;
  margin: 22px 0 14px 0;
}
.kpi-row{ display:flex; gap:12px; flex-wrap:wrap; }
.kpi{
  flex: 1 1 220px;
  background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
  border:1px solid rgba(255,255,255,0.06);
  border-radius: 14px;
  padding: 14px 14px;
  min-height: 96px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.35);
}
.kpi .label{
  font-family:'JetBrains Mono', monospace;
  color: var(--muted);
  font-size: 11px;
  letter-spacing: 2px;
  text-transform: uppercase;
}
.kpi .value{
  font-size: 30px; font-weight: 800; margin-top: 6px; color: #fff;
}
.kpi .explain{
  margin-top: 6px; color: #cbd5e1; font-size: 12px; line-height: 1.25rem;
}
.hl-green{ color: var(--green); font-weight: 800; }
.hl-red{ color: var(--red); font-weight: 800; }
.hl-amb{ color: var(--amber); font-weight: 800; }
.hl-mono{ font-family:'JetBrains Mono', monospace; }
.note{
  background: #05070a;
  border-left: 10px solid #3b82f6;
  border-right: 1px solid rgba(255,255,255,0.08);
  border-top: 1px solid rgba(255,255,255,0.08);
  border-bottom: 1px solid rgba(255,255,255,0.08);
  padding: 14px 16px;
  border-radius: 0 12px 12px 0;
  margin: 8px 0 14px 0;
  color: #e5e7eb;
  line-height: 1.65;
}
.hr{ height:1px; background: rgba(255,255,255,0.07); margin: 14px 0; }
.small-muted{ color: var(--muted); font-size: 12px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ---------------------------
# UTIL (texto/format)
# ---------------------------
def didactic(prefix: str, body: str) -> str:
    prefix = prefix.strip()
    if not (prefix.startswith("Isso significa que") or prefix.startswith("Na prática")):
        prefix = "Na prática..."
    return f"{prefix} {body}".strip()

def fmt_num(x: float, nd: int = 2) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "N/A"
    return f"{x:.{nd}f}"

def fmt_pct(x: float, nd: int = 1) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "N/A"
    return f"{x*100:.{nd}f}%"

def clamp01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))

def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def safe_float(x, default=np.nan) -> float:
    try:
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return default
        return v
    except Exception:
        return default

# ---------------------------
# UNIVERSE (BOOT local only)
# ---------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_FILES = [
    "Mercados de Ações Américas.csv",
    "Tendências da Bolsa de Valores e Ações em Destaque.csv",
    "Preços de Commodities em Tempo Real_.csv",
    "Pares por Moeda.csv",
    "Todas criptos.txt",
    "listaativos.txt",
    "listaativos_500_extremo.txt",
    "listaativos_1000.txt",
]

TICKER_RE = re.compile(r"^[A-Z0-9\^\=\-\.]{1,25}$")

@dataclass
class AssetRow:
    ticker: str
    name: str
    category: str
    source: str

def _safe_read_text(path: str) -> str:
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except Exception:
            pass
    return ""

def _try_read_csv(path: str) -> Optional[pd.DataFrame]:
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return None

def _normalize_ticker(raw: str) -> str:
    raw = str(raw).strip()
    raw = raw.replace("—", "-").replace("–", "-")
    raw = raw.replace(" ", "")
    raw = raw.replace("$", "")
    return raw

def parse_txt_tickers(path: str, category: str, source: str) -> List[AssetRow]:
    txt = _safe_read_text(path)
    out: List[AssetRow] = []
    for line in txt.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = re.split(r"[;\,\|\t]{1}", line, maxsplit=1)
        ticker = _normalize_ticker(parts[0])
        name = parts[1].strip() if len(parts) > 1 else ""
        if ticker:
            out.append(AssetRow(ticker=ticker, name=name or ticker, category=category, source=source))
    return out

def parse_generic_csv(path: str, fallback_category: str, source: str) -> List[AssetRow]:
    df = _try_read_csv(path)
    if df is None or df.empty:
        return []
    df.columns = [str(c).strip().lower() for c in df.columns]

    ticker_cols = [c for c in df.columns if c in ("ticker", "symbol", "ativo", "asset", "code")]
    name_cols = [c for c in df.columns if c in ("name", "nome", "descricao", "description", "empresa")]
    cat_cols = [c for c in df.columns if c in ("categoria", "category", "classe", "class")]

    ticker_col = ticker_cols[0] if ticker_cols else df.columns[0]
    name_col = name_cols[0] if name_cols else None
    cat_col = cat_cols[0] if cat_cols else None

    out: List[AssetRow] = []
    for _, r in df.iterrows():
        t = _normalize_ticker(r.get(ticker_col, ""))
        if not t:
            continue
        nm = str(r.get(name_col, "")).strip() if name_col else ""
        ct = str(r.get(cat_col, "")).strip() if cat_col else ""
        out.append(AssetRow(ticker=t, name=nm or t, category=ct or fallback_category, source=source))
    return out

def _looks_like_bad_row(t: str) -> bool:
    # elimina coisas tipo "NOME", "PREÇO", "VAR", "VOL."
    bad = {"NOME", "PREÇO", "PRECO", "VAR", "VALOR", "VOL", "VOL.", "VOLUME", "1"}
    if t.upper() in bad:
        return True
    # elimina token muito curto sem sentido
    if len(t) <= 1:
        return True
    return False

@st.cache_data(show_spinner=False)
def build_universe_local(app_dir: str) -> Tuple[pd.DataFrame, List[str]]:
    rows: List[AssetRow] = []
    missing: List[str] = []

    for fn in DEFAULT_FILES:
        p = os.path.join(app_dir, fn)
        if not os.path.exists(p):
            missing.append(fn)
            continue

        lower = fn.lower()
        if lower.endswith(".txt"):
            if "cripto" in lower:
                cat = "CRIPTOS"
            elif "1000" in lower:
                cat = "UNIVERSO EXTENSO (1000)"
            elif "500" in lower:
                cat = "UNIVERSO EXTENSO (500)"
            else:
                cat = "UNIVERSO (TXT)"
            rows += parse_txt_tickers(p, category=cat, source=fn)

        elif lower.endswith(".csv"):
            if "commodit" in lower:
                cat = "COMMODITIES"
            elif "pares" in lower or "moeda" in lower or "cambio" in lower or "câmbio" in lower:
                cat = "CÂMBIO"
            elif "acoes" in lower or "ações" in lower or "américas" in lower:
                cat = "AÇÕES"
            else:
                cat = "UNIVERSO (CSV)"
            rows += parse_generic_csv(p, fallback_category=cat, source=fn)

    if not rows:
        rows = [
            AssetRow("PETR4.SA", "Petrobras PN", "AÇÕES BR", "seed"),
            AssetRow("VALE3.SA", "Vale ON", "AÇÕES BR", "seed"),
            AssetRow("^GSPC", "S&P 500", "ÍNDICES", "seed"),
            AssetRow("^VIX", "VIX", "MACRO", "seed"),
            AssetRow("USDBRL=X", "USD/BRL", "CÂMBIO", "seed"),
            AssetRow("BTC-USD", "Bitcoin", "CRIPTOS", "seed"),
            AssetRow("GC=F", "Ouro", "COMMODITIES", "seed"),
            AssetRow("CL=F", "Petróleo WTI", "COMMODITIES", "seed"),
        ]

    dfu = pd.DataFrame([r.__dict__ for r in rows])
    dfu["ticker"] = dfu["ticker"].astype(str).str.strip()
    dfu["name"] = dfu["name"].astype(str).str.strip()
    dfu["category"] = dfu["category"].astype(str).str.strip()
    dfu["source"] = dfu["source"].astype(str).str.strip()

    dfu = dfu[dfu["ticker"] != ""].copy()
    dfu = dfu[~dfu["ticker"].apply(_looks_like_bad_row)].copy()

    # filtro de plausibilidade (não bloqueia tickers esquisitos, mas reduz lixo)
    dfu = dfu[dfu["ticker"].apply(lambda x: bool(TICKER_RE.match(x)) or len(x) >= 3)].copy()

    dfu = dfu.drop_duplicates(subset=["ticker"], keep="first")
    dfu = dfu.sort_values(["category", "ticker"]).reset_index(drop=True)
    return dfu, missing

# ---------------------------
# DATA FETCH (only on analyze)
# ---------------------------
def _download_once(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(
        tickers=ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        group_by="column",
        progress=False,
        threads=False,
    )
    if df is None:
        return pd.DataFrame()
    return df

@st.cache_data(show_spinner=False, ttl=900)
def fetch_ohlc_robust(ticker: str, period: str, interval: str, retries: int = 2) -> pd.DataFrame:
    """
    Robustez:
    - retries
    - fallback periods quando vier vazio
    - normaliza MultiIndex/Close DataFrame
    """
    ticker = ticker.strip()
    fallback_periods = [period, "1y", "6mo", "3mo", "1mo", "7d"]
    tried = 0

    for per in fallback_periods:
        for _ in range(max(1, retries)):
            tried += 1
            try:
                df = _download_once(ticker, per, interval)
            except Exception:
                df = pd.DataFrame()

            if df is None or df.empty:
                continue

            # MultiIndex? reduzir
            if isinstance(df.columns, pd.MultiIndex):
                try:
                    t2 = df.columns.get_level_values(1).unique().tolist()
                    use = t2[0] if t2 else ticker
                    df = df.xs(use, axis=1, level=1)
                except Exception:
                    df.columns = ["_".join(map(str, c)).strip() for c in df.columns]

            # padroniza colunas
            ren = {c: str(c).title() for c in df.columns}
            df = df.rename(columns=ren)

            # garante essenciais
            for col in ["Open", "High", "Low", "Close"]:
                if col not in df.columns:
                    return pd.DataFrame()

            # Close pode vir DF
            if "Close" in df.columns and isinstance(df["Close"], pd.DataFrame):
                df["Close"] = df["Close"].iloc[:, 0]

            # Volume ausente
            if "Volume" not in df.columns:
                df["Volume"] = np.nan

            df = df.dropna(subset=["Close"]).copy()
            if not df.empty:
                return df

    return pd.DataFrame()

@st.cache_data(show_spinner=False, ttl=1800)
def fetch_info(ticker: str) -> Dict:
    try:
        return yf.Ticker(ticker).info or {}
    except Exception:
        return {}

@st.cache_data(show_spinner=False, ttl=600)
def fetch_bid_ask(ticker: str) -> Tuple[float, float]:
    """
    Tenta pegar bid/ask; se não existir, retorna NaN.
    """
    try:
        tk = yf.Ticker(ticker)
        fi = getattr(tk, "fast_info", None)
        bid = getattr(fi, "bid", np.nan) if fi is not None else np.nan
        ask = getattr(fi, "ask", np.nan) if fi is not None else np.nan
        bid = safe_float(bid, np.nan)
        ask = safe_float(ask, np.nan)
        return bid, ask
    except Exception:
        return (np.nan, np.nan)

@st.cache_data(show_spinner=False, ttl=600)
def fetch_yahoo_news(ticker: str) -> List[Dict]:
    try:
        news = yf.Ticker(ticker).news or []
        out = []
        for n in news:
            title = (n.get("title") or "").strip() or "Sem título"
            link = (n.get("link") or "").strip()
            pub = (n.get("publisher") or "").strip() or "N/A"
            ts = n.get("providerPublishTime")
            dt = ""
            try:
                if ts:
                    dt = datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%d %H:%M")
            except Exception:
                dt = ""
            if not title and not link:
                continue
            out.append({"title": title, "link": link, "source": f"Yahoo | {pub}", "dt": dt})
        return out[:10]
    except Exception:
        return []

def fetch_google_news_rss(query: str, max_items: int = 10) -> List[Dict]:
    """
    RSS do Google News (sem chave).
    Ex: https://news.google.com/rss/search?q=PETR4.SA
    """
    try:
        url = "https://news.google.com/rss/search"
        params = {"q": query, "hl": "pt-BR", "gl": "BR", "ceid": "BR:pt-419"}
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return []
        soup = BeautifulSoup(r.text, "xml")
        items = soup.find_all("item")
        out = []
        for it in items[:max_items]:
            title = (it.title.text or "").strip() if it.title else "Sem título"
            link = (it.link.text or "").strip() if it.link else ""
            pub = (it.source.text or "Google News").strip() if it.source else "Google News"
            dt = (it.pubDate.text or "").strip() if it.pubDate else ""
            out.append({"title": title, "link": link, "source": pub, "dt": dt})
        return out
    except Exception:
        return []

# ---------------------------
# ENGINES (técnico / risco / custos / backtest)
# ---------------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["Close"] = pd.to_numeric(d["Close"], errors="coerce")

    # returns
    d["Return"] = d["Close"].pct_change()

    # SMA/EMA
    d["SMA20"] = d["Close"].rolling(20, min_periods=20).mean()
    d["SMA50"] = d["Close"].rolling(50, min_periods=50).mean()
    d["SMA200"] = d["Close"].rolling(200, min_periods=200).mean()
    d["EMA20"] = d["Close"].ewm(span=20, adjust=False).mean()
    d["EMA50"] = d["Close"].ewm(span=50, adjust=False).mean()

    # STD + ZScore (robusto)
    d["STD20"] = d["Close"].rolling(20, min_periods=20).std(ddof=0)
    d["ZScore"] = pd.to_numeric((d["Close"] - d["SMA20"]) / d["STD20"], errors="coerce")

    # RSI14 + RSI2
    def rsi(series: pd.Series, n: int) -> pd.Series:
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0).rolling(n, min_periods=n).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(n, min_periods=n).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    d["RSI14"] = rsi(d["Close"], 14)
    d["RSI2"] = rsi(d["Close"], 2)

    # Bollinger (20,2)
    d["BB_MID"] = d["SMA20"]
    d["BB_UP"] = d["SMA20"] + 2 * d["STD20"]
    d["BB_DN"] = d["SMA20"] - 2 * d["STD20"]

    # ATR14
    tr1 = (d["High"] - d["Low"]).abs()
    tr2 = (d["High"] - d["Close"].shift(1)).abs()
    tr3 = (d["Low"] - d["Close"].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    d["ATR14"] = tr.rolling(14, min_periods=14).mean()

    # Vol anual (proxy)
    d["VolAnn"] = d["Return"].rolling(20, min_periods=20).std(ddof=0) * np.sqrt(252)

    # Trigger “reversão extrema” (ajustável)
    d["Trigger_Revert"] = (d["ZScore"] <= -2.0) & (d["RSI2"] < 15)

    # Trend filter
    d["TrendUp"] = d["EMA20"] > d["EMA50"]
    d["TrendDn"] = d["EMA20"] < d["EMA50"]

    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=["Close"])
    return d

def risk_of_ruin(win_rate: float, payoff_r: float, risk_per_trade: float, ruin_level: float, horizon: int) -> float:
    # modelo conservador e estável
    p = float(win_rate)
    b = float(payoff_r)
    f = float(risk_per_trade)
    rl = float(ruin_level)
    T = int(horizon)

    if f <= 0:
        return 0.0
    if rl <= 0 or rl >= 1:
        return 1.0

    mu = p * b - (1 - p) * 1.0
    var = p * (b - mu) ** 2 + (1 - p) * (-1 - mu) ** 2
    sigma = math.sqrt(max(var, 1e-12))

    drift = mu * f
    vol = sigma * f
    barrier = math.log(max(1e-9, 1 - rl))
    if vol <= 0:
        return 1.0 if drift <= 0 else 0.0

    z = (barrier - drift * T) / (vol * math.sqrt(max(1, T)))
    p_touch = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    return clamp01(p_touch)

def estimate_spread_bps(last_price: float, bid: float, ask: float, fallback_bps: float) -> float:
    """
    Se bid/ask existirem, spread = (ask-bid)/mid em bps.
    Senão, usa fallback manual.
    """
    if np.isfinite(bid) and np.isfinite(ask) and ask > bid and last_price > 0:
        mid = (ask + bid) / 2.0
        if mid > 0:
            return float(((ask - bid) / mid) * 10000.0)
    return float(fallback_bps)

def costs_per_roundtrip_bps(spread_bps: float, slippage_bps: float, commission_bps: float) -> float:
    """
    Custo de ida+volta em bps.
    - Spread: normalmente você paga ~1 spread por roundtrip (entra no ask e sai no bid)
    - Slippage: modela execução “pior” em bps (entrada+saída)
    - Comissão: bps totais por roundtrip
    """
    return float(spread_bps + 2.0 * slippage_bps + commission_bps)

def backtest_reversion(df: pd.DataFrame, cost_bps_roundtrip: float) -> pd.DataFrame:
    """
    Estratégia mínima e estável:
    - Entrada: Trigger_Revert True
    - Saída: ZScore volta para >= -0.5 OU RSI14 > 55 OU stop por ATR (simples)
    - Custos: desconta bps no trade (roundtrip)
    """
    d = df.copy()
    d["pos"] = 0
    d["trade_ret"] = 0.0

    in_pos = False
    entry_price = np.nan

    # stop por ATR: 2*ATR abaixo do entry
    stop_mult = 2.0

    for i in range(1, len(d) - 1):
        if not in_pos:
            if bool(d["Trigger_Revert"].iloc[i]) and np.isfinite(d["ATR14"].iloc[i]):
                in_pos = True
                d.at[d.index[i], "pos"] = 1
                entry_price = float(d["Close"].iloc[i])
        else:
            d.at[d.index[i], "pos"] = 1
            price = float(d["Close"].iloc[i])
            atr = float(d["ATR14"].iloc[i]) if np.isfinite(d["ATR14"].iloc[i]) else np.nan
            stop_level = entry_price - stop_mult * atr if np.isfinite(atr) else -np.inf

            exit_cond = (float(d["ZScore"].iloc[i]) >= -0.5) or (float(d["RSI14"].iloc[i]) > 55) or (price <= stop_level)
            if exit_cond:
                # sai no próximo close (i+1) para evitar lookahead
                exit_price = float(d["Close"].iloc[i + 1])
                gross = (exit_price / entry_price) - 1.0

                # desconta custo em bps (roundtrip)
                cost = cost_bps_roundtrip / 10000.0
                net = gross - cost

                d.at[d.index[i + 1], "trade_ret"] = net
                in_pos = False
                entry_price = np.nan

    d["equity"] = (1 + d["trade_ret"].fillna(0)).cumprod()
    d["peak"] = d["equity"].cummax()
    d["drawdown"] = d["equity"] / d["peak"] - 1.0
    return d

def monte_carlo(last_price: float, mu_daily: float, sigma_daily: float, days: int = 30, n_paths: int = 500) -> np.ndarray:
    if last_price <= 0 or not np.isfinite(last_price):
        return np.zeros((n_paths, days + 1))
    rng = np.random.default_rng(42)
    shocks = rng.normal(loc=(mu_daily - 0.5 * sigma_daily ** 2), scale=sigma_daily, size=(n_paths, days))
    log_paths = np.cumsum(shocks, axis=1)
    log_paths = np.hstack([np.zeros((n_paths, 1)), log_paths])
    return last_price * np.exp(log_paths)

def macro_watchlist_for(asset_ticker: str) -> List[Tuple[str, str]]:
    """
    Proxies macro via Yahoo (sem chave):
    - funciona para quase tudo e não trava.
    """
    return [
        ("^GSPC", "S&P 500"),
        ("^VIX", "VIX"),
        ("^TNX", "US 10Y Yield"),
        ("DX-Y.NYB", "DXY (US Dollar Index)"),
        ("CL=F", "WTI (Oil)"),
        ("GC=F", "Gold"),
        ("USDBRL=X", "USD/BRL"),
        ("BTC-USD", "BTC"),
    ]

@st.cache_data(show_spinner=False, ttl=600)
def fetch_macro_panel() -> pd.DataFrame:
    rows = []
    for t, name in macro_watchlist_for(""):
        df = fetch_ohlc_robust(t, period="6mo", interval="1d", retries=1)
        if df.empty:
            continue
        c = df["Close"].dropna()
        if len(c) < 10:
            continue
        ret = c.pct_change().dropna()
        vol = float(ret.rolling(20, min_periods=10).std(ddof=0).iloc[-1] * np.sqrt(252))
        mom = float((c.iloc[-1] / c.iloc[-21] - 1.0)) if len(c) > 21 else float(c.iloc[-1] / c.iloc[0] - 1.0)
        rows.append({"ticker": t, "name": name, "last": float(c.iloc[-1]), "mom_1m": mom, "vol_ann": vol})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("ticker").reset_index(drop=True)

def macro_regime(macro: pd.DataFrame) -> Tuple[str, str]:
    if macro is None or macro.empty:
        return ("Macro indisponível", didactic("Na prática...", "os proxies macro não carregaram agora (rate-limit/instabilidade). Tente de novo."))
    # heurísticas simples e úteis
    vix = macro[macro["ticker"] == "^VIX"]["last"]
    dxy = macro[macro["ticker"] == "DX-Y.NYB"]["mom_1m"]
    tnx = macro[macro["ticker"] == "^TNX"]["mom_1m"]

    vix_val = float(vix.iloc[0]) if len(vix) else np.nan
    dxy_m = float(dxy.iloc[0]) if len(dxy) else np.nan
    tnx_m = float(tnx.iloc[0]) if len(tnx) else np.nan

    risk_off = False
    reasons = []
    if np.isfinite(vix_val) and vix_val >= 20:
        risk_off = True
        reasons.append("VIX acima de 20 (stress)")
    if np.isfinite(dxy_m) and dxy_m > 0.02:
        risk_off = True
        reasons.append("DXY forte 1M (aperto global)")
    if np.isfinite(tnx_m) and tnx_m > 0.02:
        risk_off = True
        reasons.append("Yields subindo (desconto maior)")

    if risk_off:
        return ("RISK-OFF / DEFENSIVO", didactic("Na prática...", "o macro está hostil: " + "; ".join(reasons) + ". Reduza risco e exija setups mais extremos."))
    return ("RISK-ON / NEUTRO", didactic("Na prática...", "macro sem sinal de stress dominante. Ainda assim, sua execução deve respeitar custos e volatilidade."))

# ---------------------------
# PLOTS
# ---------------------------
def plot_price(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Preço"
    ))
    for col, nm in [("SMA20", "SMA20"), ("EMA20", "EMA20"), ("EMA50", "EMA50"), ("SMA200", "SMA200")]:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], name=nm))
    if "BB_UP" in df.columns and "BB_DN" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_UP"], name="BB_UP", opacity=0.35))
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_DN"], name="BB_DN", opacity=0.35))

    fig.update_layout(
        template="plotly_dark", height=560,
        title=f"{ticker} | Preço + Tendência + Bollinger",
        margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig

def plot_indicators(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.44, 0.28, 0.28])
    fig.add_trace(go.Scatter(x=df.index, y=df["ZScore"], name="ZScore"), row=1, col=1)
    fig.add_hline(y=2.0, line_dash="dot", row=1, col=1)
    fig.add_hline(y=-2.0, line_dash="dot", row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["RSI14"], name="RSI14"), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", row=2, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["VolAnn"], name="VolAnn (20d)"), row=3, col=1)
    fig.update_layout(template="plotly_dark", height=560, margin=dict(l=10, r=10, t=40, b=10))
    return fig

def plot_mc(paths: np.ndarray) -> go.Figure:
    fig = go.Figure()
    if paths.size == 0:
        fig.update_layout(template="plotly_dark", height=520)
        return fig
    n_paths, _ = paths.shape
    sample = min(120, n_paths)
    for i in range(sample):
        fig.add_trace(go.Scatter(y=paths[i], mode="lines", opacity=0.12, showlegend=False))
    p05 = np.percentile(paths, 5, axis=0)
    p50 = np.percentile(paths, 50, axis=0)
    p95 = np.percentile(paths, 95, axis=0)
    fig.add_trace(go.Scatter(y=p05, mode="lines", name="P05"))
    fig.add_trace(go.Scatter(y=p50, mode="lines", name="P50"))
    fig.add_trace(go.Scatter(y=p95, mode="lines", name="P95"))
    fig.update_layout(template="plotly_dark", height=520, title="Monte Carlo | Cone P05/P50/P95 (30d)")
    return fig

def plot_backtest(bt: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10, row_heights=[0.62, 0.38])
    fig.add_trace(go.Scatter(x=bt.index, y=bt["equity"], name="Equity"), row=1, col=1)
    fig.add_trace(go.Scatter(x=bt.index, y=bt["drawdown"], name="Drawdown"), row=2, col=1)
    fig.update_layout(template="plotly_dark", height=520, title="Backtest | Equity & Drawdown (com custos)")
    return fig

# ---------------------------
# FUNDAMENTAL SCORE (simples, objetivo, útil)
# ---------------------------
def fundamental_snapshot(info: Dict) -> Dict[str, str]:
    def g(k, default="N/A"):
        v = info.get(k, None)
        if v is None:
            return default
        if isinstance(v, (int, float)):
            # converte yields
            if "yield" in k.lower() or k.lower().endswith("yield"):
                return fmt_pct(float(v), 2)
            return fmt_num(float(v), 2)
        return str(v)

    return {
        "Nome": (info.get("longName") or info.get("shortName") or "N/A"),
        "Setor": (info.get("sector") or "N/A"),
        "Indústria": (info.get("industry") or "N/A"),
        "Moeda": (info.get("currency") or "N/A"),
        "Bolsa": (info.get("exchange") or info.get("fullExchangeName") or "N/A"),
        "MarketCap": f"{int(info.get('marketCap')):,}".replace(",", ".") if isinstance(info.get("marketCap"), (int, float)) else "N/A",
        "P/L (trailingPE)": g("trailingPE"),
        "P/VP (priceToBook)": g("priceToBook"),
        "Dividend Yield": g("dividendYield"),
        "Earnings Growth": g("earningsGrowth"),
        "Revenue Growth": g("revenueGrowth"),
        "ROE": g("returnOnEquity"),
        "Margem Oper.": g("operatingMargins"),
    }

def fundamental_score(info: Dict) -> Tuple[float, List[str]]:
    """
    Score heurístico (0..100) — não “prever preço”, mas dar leitura objetiva.
    """
    score = 50.0
    notes = []

    pe = safe_float(info.get("trailingPE"), np.nan)
    pb = safe_float(info.get("priceToBook"), np.nan)
    dy = safe_float(info.get("dividendYield"), np.nan)
    eg = safe_float(info.get("earningsGrowth"), np.nan)
    rg = safe_float(info.get("revenueGrowth"), np.nan)
    roe = safe_float(info.get("returnOnEquity"), np.nan)
    om = safe_float(info.get("operatingMargins"), np.nan)

    if np.isfinite(roe):
        if roe > 0.15:
            score += 10; notes.append("ROE forte")
        elif roe < 0.05:
            score -= 8; notes.append("ROE fraco")
    if np.isfinite(om):
        if om > 0.15:
            score += 8; notes.append("Margem forte")
        elif om < 0.05:
            score -= 6; notes.append("Margem fraca")
    if np.isfinite(eg):
        if eg > 0.10:
            score += 8; notes.append("Earnings growth bom")
        elif eg < 0:
            score -= 8; notes.append("Earnings em queda")
    if np.isfinite(rg):
        if rg > 0.08:
            score += 6; notes.append("Receita crescendo")
        elif rg < 0:
            score -= 6; notes.append("Receita caindo")
    if np.isfinite(dy):
        if dy >= 0.04:
            score += 6; notes.append("Dividendos relevantes")
    if np.isfinite(pe):
        if pe < 10:
            score += 5; notes.append("P/L baixo (pode ser valor ou risco)")
        elif pe > 30:
            score -= 6; notes.append("P/L alto (precificação exigente)")
    if np.isfinite(pb):
        if pb < 1.5:
            score += 3; notes.append("P/VP moderado")
        elif pb > 5:
            score -= 4; notes.append("P/VP caro")

    score = float(np.clip(score, 0, 100))
    if not notes:
        notes = ["Dados fundamentais limitados no Yahoo para este ativo."]
    return score, notes

# ---------------------------
# VEREDITO (decisão “operacional”)
# ---------------------------
def verdict(trigger: bool, risk_ruin: float, vol_ann: float, macro_reg: str, cost_bps_rt: float) -> Tuple[str, str, List[str], List[str]]:
    buy = []
    avoid = []

    if trigger:
        buy.append("Trigger de reversão fechou (ZScore ≤ -2 e RSI2 < 15).")
    else:
        avoid.append("Trigger não fechou; sem assimetria clara de reversão.")

    if np.isfinite(vol_ann) and vol_ann <= 0.60:
        buy.append("Volatilidade dentro de faixa operável.")
    elif np.isfinite(vol_ann):
        avoid.append("Volatilidade hostil; stop caro e ruído alto.")

    if risk_ruin <= 0.25:
        buy.append("Risco de ruína aceitável para seus inputs.")
    else:
        avoid.append("Risco de ruína alto; sizing/edge insuficientes.")

    if "RISK-OFF" in macro_reg:
        avoid.append("Macro em modo defensivo (RISK-OFF); exija setups extremos e reduza risco.")
    else:
        buy.append("Macro sem stress dominante (RISK-ON/Neutro).")

    # custos (spread/slippage/comissão)
    if cost_bps_rt >= 25:
        avoid.append(f"Custo roundtrip alto (~{cost_bps_rt:.1f} bps); você precisa de movimento maior para compensar.")
    else:
        buy.append(f"Custo roundtrip controlado (~{cost_bps_rt:.1f} bps).")

    if trigger and risk_ruin <= 0.25 and (not np.isfinite(vol_ann) or vol_ann <= 0.60) and "RISK-OFF" not in macro_reg:
        head = "EXECUÇÃO AUTORIZADA"
        expl = didactic("Na prática...", "o setup tem assimetria e o risco/custos não sabotam a execução. Operar ainda exige plano (stop, tamanho e invalidação).")
    else:
        head = "OBSERVAÇÃO DEFENSIVA"
        expl = didactic(
            "Na prática...",
            "a estrutura atual recomenda defesa: ou o trigger não fechou, ou o risco estatístico está alto, ou a volatilidade/macro está hostil. Operar aqui é pagar spread para comprar incerteza."
        )

    return head, expl, buy, avoid

# ---------------------------
# UI HELPERS
# ---------------------------
def header(boot_ms: float, universe_size: int, diag: bool):
    st.markdown(
        f"""
        <div class="atlas-header">
          <div class="atlas-title">ATLAS</div>
          <div class="atlas-sub">
            <span class="badge badge-ok">boot ok</span>
            <span class="badge">universe:{universe_size}</span>
            <span class="badge">boot:{boot_ms:.0f}ms</span>
            <span class="badge">{'diagnóstico:on' if diag else 'diagnóstico:off'}</span>
            <span class="badge badge-warn">sem corretora</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def kpi(label: str, value: str, explain: str):
    st.markdown(
        f"""
        <div class="kpi">
          <div class="label">{label}</div>
          <div class="value">{value}</div>
          <div class="explain">{explain}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# MAIN
# ============================================================
def main():
    t0 = time.time()
    diag = st.sidebar.toggle("Modo diagnóstico", value=False, help="Exibe estado interno e logs sem quebrar o fluxo.")

    # BOOT instantâneo (local only)
    universe, missing = build_universe_local(APP_DIR)
    boot_ms = (time.time() - t0) * 1000.0

    header(boot_ms, int(len(universe)), diag)

    # -----------------------
    # SIDEBAR: dados
    # -----------------------
    st.sidebar.markdown("### Parâmetros de Dados")
    period = st.sidebar.selectbox(
        "Período",
        ["7d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"],
        index=5,
        help="Isso significa que: define quanto histórico entra nas métricas. Na prática: períodos longos estabilizam médias; curtos tornam sinais mais reativos.",
    )
    interval = st.sidebar.selectbox(
        "Intervalo",
        ["1d", "1h", "30m", "15m", "5m"],
        index=0,
        help="Isso significa que: granularidade do candle. Na prática: intraday pode ter buracos/limites do Yahoo; 1d é mais confiável.",
    )

    st.sidebar.markdown("### Microestrutura / Custos (impacto real)")
    fallback_spread_bps = st.sidebar.slider(
        "Spread (fallback) em bps",
        0.0, 200.0, 20.0, 1.0,
        help="Isso significa que: custo por roundtrip pelo spread quando bid/ask não existe. Na prática: em ações BR/cripto isso muda tudo no resultado.",
    )
    slippage_bps = st.sidebar.slider(
        "Slippage em bps (por lado)",
        0.0, 100.0, 5.0, 1.0,
        help="Isso significa que: execução pior que o preço teórico (entrada e saída). Na prática: aumenta em volatilidade e baixa liquidez.",
    )
    commission_bps = st.sidebar.slider(
        "Comissão roundtrip em bps",
        0.0, 100.0, 5.0, 1.0,
        help="Isso significa que: custo total de corretagem/taxas por ida+volta (em bps). Na prática: coloca isso agora para não se enganar no backtest.",
    )

    st.sidebar.markdown("### Risco de Ruína (inputs)")
    win_rate = st.sidebar.slider("Win Rate estimado", 0.10, 0.90, 0.45, 0.01,
                                help="Na prática: se você superestima, a ruína ‘some’ no papel e aparece no real.")
    payoff_r = st.sidebar.slider("Payoff médio (R-múltiplo)", 0.50, 5.00, 1.50, 0.05,
                                help="Na prática: payoff baixo exige win rate alto para sobreviver.")
    risk_per_trade = st.sidebar.slider("Risk-per-Trade", 0.00, 0.05, 0.01, 0.001,
                                help="Na prática: sizing é o volante do risco. Subir isso ‘quebra’ sistemas medianos.")
    ruin_level = st.sidebar.slider("Limite de ruína (perda do capital)", 0.10, 0.90, 0.30, 0.01,
                                help="Na prática: sua barreira de dor. Quanto menor, mais fácil ‘tocar’ a ruína.")
    horizon = st.sidebar.slider("Horizonte (trades)", 50, 800, 200, 10,
                                help="Na prática: horizonte maior expõe a fragilidade do sizing em sequências ruins.")

    # -----------------------
    # SIDEBAR: seleção (1 ativo)
    # -----------------------
    st.sidebar.markdown("### Seleção de Ativos (1 por vez)")
    if universe.empty:
        st.sidebar.error("Universo vazio. Coloque seus arquivos .csv/.txt na pasta do app.py.")
        st.stop()

    categories = sorted(universe["category"].unique().tolist())
    cat = st.sidebar.selectbox("Categoria", categories, index=0,
                              help="Na prática: isso evita caos. Você analisa um ativo por vez, profundo.")

    sub = universe[universe["category"] == cat].copy()
    sub["label"] = sub["ticker"].astype(str) + " — " + sub["name"].astype(str)
    # busca
    q = st.sidebar.text_input("Busca rápida", value="", help="Digite parte do ticker ou nome.")
    if q.strip():
        qq = q.strip().lower()
        sub = sub[sub["label"].str.lower().str.contains(qq, na=False)].copy()

    # limita apenas para render (sem travar)
    max_show = 1500
    sub = sub.head(max_show)

    choice = st.sidebar.selectbox("Ativo (ticker — descrição)", sub["label"].tolist(), index=0)
    ticker = choice.split("—", 1)[0].strip()

    analyze = st.sidebar.button("ANALISAR ATIVO", use_container_width=True,
                                help="Dispara coleta + todos os motores. Nada roda no boot.")

    if diag and missing:
        st.sidebar.warning("Arquivos faltantes (não bloqueia):\n- " + "\n- ".join(missing))

    # -----------------------
    # Pipeline sempre visível
    # -----------------------
    st.markdown('<div class="section-title">Pipeline</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="note">
          <span class="hl-mono hl-green">UI ONLINE</span> — {didactic("Na prática...", "o BOOT é instantâneo. Se algo demorar, é Yahoo/Internet no momento da análise.")}
          <div class="hr"></div>
          <span class="hl-mono">Ativo selecionado:</span> <span class="hl-green">{ticker}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not analyze:
        st.info(didactic("Isso significa que...", "o terminal está pronto. Clique em **ANALISAR ATIVO** para executar o bloco completo."))
        st.stop()

    # =========================================================
    # EXECUÇÃO (somente agora chamamos rede)
    # =========================================================
    with st.spinner("Coletando OHLC (Yahoo) e calculando motores..."):
        df_raw = fetch_ohlc_robust(ticker, period=period, interval=interval, retries=2)

    if df_raw.empty:
        st.error("Sem dados retornados pelo Yahoo para esse ticker/período/intervalo (mesmo com fallback). Troque ticker/período/intervalo.")
        st.caption(didactic("Na prática...", "isso ocorre em tickers delistados, símbolos incorretos ou intervalos intraday com restrição."))
        st.stop()

    df = add_indicators(df_raw)
    if df.empty or len(df) < 30:
        st.error("Dados insuficientes para indicadores (aumente o período).")
        st.stop()

    last = df.iloc[-1]
    last_price = float(last["Close"])

    # bid/ask e spread
    bid, ask = fetch_bid_ask(ticker)
    spread_bps = estimate_spread_bps(last_price, bid, ask, fallback_spread_bps)
    cost_rt_bps = costs_per_roundtrip_bps(spread_bps, slippage_bps, commission_bps)

    # risco de ruína
    ruin_p = risk_of_ruin(win_rate, payoff_r, risk_per_trade, ruin_level, horizon)

    # monte carlo
    ret = df["Return"].dropna()
    mu = float(ret.mean()) if len(ret) else 0.0
    sig = float(ret.std(ddof=0)) if len(ret) else 0.0
    paths = monte_carlo(last_price, mu, sig, days=30, n_paths=500)

    # macro + regime
    macro = fetch_macro_panel()
    macro_reg, macro_expl = macro_regime(macro)

    # news (camadas)
    info = fetch_info(ticker)
    yahoo_news = fetch_yahoo_news(ticker)
    gnews = fetch_google_news_rss(query=ticker, max_items=10)
    # dedup simples
    seen = set()
    news = []
    for n in (yahoo_news + gnews):
        key = (n.get("title",""), n.get("link",""))
        if key in seen:
            continue
        seen.add(key)
        news.append(n)
    news = news[:15]

    # fundamental
    f_snap = fundamental_snapshot(info)
    f_score, f_notes = fundamental_score(info)

    # backtest com custos
    bt = backtest_reversion(df, cost_bps_roundtrip=cost_rt_bps)

    # veredito
    head, expl, buy, avoid = verdict(
        trigger=bool(last["Trigger_Revert"]),
        risk_ruin=float(ruin_p),
        vol_ann=float(last["VolAnn"]) if np.isfinite(float(last["VolAnn"])) else np.nan,
        macro_reg=macro_reg,
        cost_bps_rt=cost_rt_bps,
    )

    # =========================================================
    # KPIs
    # =========================================================
    st.markdown('<div class="section-title">Diagnóstico Atual</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    with c1:
        kpi("Preço", fmt_num(last_price, 2), didactic("Na prática...", "é o último close disponível; ele ancora stops, sizing e custos em bps."))
    with c2:
        kpi("Z-Score", fmt_num(float(last["ZScore"]), 2), didactic("Na prática...", "extremos (±2) são zonas de decisão — não de conforto."))
    with c3:
        kpi("RSI2", fmt_num(float(last["RSI2"]), 1), didactic("Na prática...", "mede exaustão curtíssima. É o ‘termômetro de pânico’ do candle."))
    with c4:
        kpi("Vol (anual)", fmt_pct(float(last["VolAnn"])) if np.isfinite(float(last["VolAnn"])) else "N/A",
            didactic("Na prática...", "vol alta exige stop maior e risco por trade menor."))
    with c5:
        kpi("Custo RT", f"{cost_rt_bps:.1f} bps",
            didactic("Na prática...", "isso é o pedágio (spread+slippage+comissão). Sem superar isso, o trade é negativo por construção."))
    with c6:
        kpi("Risco de Ruína", fmt_pct(float(ruin_p)), didactic("Na prática...", "se estiver alto, você está ‘apostando’ e não operando um sistema."))

    # =========================================================
    # TABS
    # =========================================================
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
        ["📈 Mercado", "🧠 Indicadores", "🏛️ Fundamental", "🌍 Macro", "📰 News", "🧪 Backtest", "⚖️ Veredito"]
    )

    with tab1:
        st.caption(didactic("Na prática...", "estrutura de preço + tendência + volatilidade implícita (Bollinger)."))
        st.plotly_chart(plot_price(df, ticker), use_container_width=True)
        st.dataframe(df.tail(25), use_container_width=True)

    with tab2:
        st.caption(didactic("Na prática...", "ZScore/RSI/Vol mostram distância, fadiga e risco do terreno."))
        st.plotly_chart(plot_indicators(df), use_container_width=True)

        colA, colB = st.columns(2)
        with colA:
            st.markdown("### ✅ **Compre / Observe compra**")
            st.markdown(didactic("Na prática...", "você compra quando assimetria > custos e risco é aceitável."))
            if buy:
                for r in buy:
                    st.markdown(f"- {r}")
            else:
                st.markdown("- Nenhum argumento forte de compra detectado.")
        with colB:
            st.markdown("### ❌ **Não compre / Defesa**")
            st.markdown(didactic("Na prática...", "você não compra quando o mercado te força a pagar caro por incerteza."))
            if avoid:
                for r in avoid:
                    st.markdown(f"- {r}")
            else:
                st.markdown("- Nenhum bloqueio crítico detectado.")

    with tab3:
        st.markdown("### Sumário / Glossário do Ativo")
        st.caption(didactic("Na prática...", "isso melhora identificação e dá contexto de ‘qual negócio’ você está operando."))
        st.json(f_snap)

        st.markdown("### Score Fundamental (0-100)")
        st.markdown(f"**Score:** <span class='hl-amb'>{f_score:.0f}</span>", unsafe_allow_html=True)
        for n in f_notes:
            st.markdown(f"- {n}")

        st.markdown("### Nota operacional")
        st.info(didactic("Na prática...", "fundamental não é gatilho de entrada; ele define ‘qual ativo merece risco’ e como você trata notícias/eventos."))

    with tab4:
        st.markdown(f"### Macro Regime: **{macro_reg}**")
        st.write(macro_expl)
        st.markdown("### Painel Macro (proxies via Yahoo)")
        if macro is None or macro.empty:
            st.warning("Macro indisponível agora.")
        else:
            st.dataframe(macro, use_container_width=True)

    with tab5:
        st.markdown("### Notícias (camadas: Yahoo + Google News RSS)")
        st.caption(didactic("Na prática...", "notícia muda regime e invalida setup. Use para não ser surpreendido."))
        if not news:
            st.warning("Sem notícias retornadas agora (limite/instabilidade).")
        else:
            for n in news:
                title = n.get("title", "Sem título")
                link = n.get("link", "")
                src = n.get("source", "N/A")
                dt = n.get("dt", "")
                if link:
                    st.markdown(f"- **{title}**  \n  *{src} | {dt}*  \n  {link}")
                else:
                    st.markdown(f"- **{title}**  \n  *{src} | {dt}*")

    with tab6:
        st.caption(didactic("Na prática...", "esse backtest já cobra custos. Se ele mal fica acima de 1.0 com custos, sua estratégia não paga o pedágio do mercado."))
        if bt is None or bt.empty:
            st.warning("Backtest indisponível.")
        else:
            st.plotly_chart(plot_backtest(bt), use_container_width=True)

            trades = int((bt["trade_ret"] != 0).sum()) if "trade_ret" in bt.columns else 0
            final_eq = float(bt["equity"].iloc[-1]) if "equity" in bt.columns else 1.0
            max_dd = float(bt["drawdown"].min()) if "drawdown" in bt.columns else 0.0
            st.write({"trades": trades, "equity_final": final_eq, "max_drawdown": max_dd})

            st.markdown("### Monte Carlo (dispersão de curto prazo)")
            st.plotly_chart(plot_mc(paths), use_container_width=True)

    with tab7:
        icon = "🟢" if head == "EXECUÇÃO AUTORIZADA" else "🔴"
        st.markdown(f"## {icon} **{head}**")
        st.write(expl)

        st.markdown("### Comandos operacionais (base futura — sem corretora)")
        st.caption(didactic("Na prática...", "isto é o ‘contrato’ para integração: buy/stop/tp virão depois. Agora é plano, não ordem."))

        # Sugestão de stop/tp (heurística) usando ATR
        atr = float(last["ATR14"]) if np.isfinite(float(last["ATR14"])) else np.nan
        stop = last_price - 2.0 * atr if np.isfinite(atr) else np.nan
        tp = last_price + 2.5 * atr if np.isfinite(atr) else np.nan

        plan = {
            "timestamp": now_str(),
            "ticker": ticker,
            "market": {
                "price": last_price,
                "bid": None if not np.isfinite(bid) else float(bid),
                "ask": None if not np.isfinite(ask) else float(ask),
                "spread_bps": float(spread_bps),
                "slippage_bps_per_side": float(slippage_bps),
                "commission_bps_roundtrip": float(commission_bps),
                "cost_roundtrip_bps": float(cost_rt_bps),
            },
            "signals": {
                "trigger_revert": bool(last["Trigger_Revert"]),
                "zscore": float(last["ZScore"]) if np.isfinite(float(last["ZScore"])) else None,
                "rsi2": float(last["RSI2"]) if np.isfinite(float(last["RSI2"])) else None,
                "rsi14": float(last["RSI14"]) if np.isfinite(float(last["RSI14"])) else None,
                "trend_up": bool(last["TrendUp"]) if "TrendUp" in df.columns else None,
                "vol_ann": float(last["VolAnn"]) if np.isfinite(float(last["VolAnn"])) else None,
            },
            "risk": {
                "risk_per_trade": float(risk_per_trade),
                "ruin_prob": float(ruin_p),
                "ruin_level": float(ruin_level),
                "horizon_trades": int(horizon),
            },
            "macro": {
                "regime": macro_reg,
            },
            "orders_stub": {
                "buy": {"enabled": False, "type": "market/limit", "qty": None},
                "stop": {"enabled": False, "type": "stop", "level": None if not np.isfinite(stop) else float(stop)},
                "take_profit": {"enabled": False, "type": "limit", "level": None if not np.isfinite(tp) else float(tp)},
            },
        }

        st.code(json.dumps(plan, ensure_ascii=False, indent=2), language="json")

        st.markdown("### Tradução objetiva (o que fazer agora)")
        if head == "EXECUÇÃO AUTORIZADA":
            st.success(didactic("Na prática...", "você pode estruturar uma entrada COM stop e sizing. O setup existe, mas custos ainda mandam: não force execução em baixa liquidez."))
        else:
            st.warning(didactic("Na prática...", "não opere. Espere o trigger fechar e/ou o macro aliviar e/ou reduza custos (melhor ativo/horário) e risco por trade."))

    if diag:
        st.markdown('<div class="section-title">Diagnóstico Interno</div>', unsafe_allow_html=True)
        st.write({
            "app_dir": APP_DIR,
            "ticker": ticker,
            "period": period,
            "interval": interval,
            "rows_raw": int(len(df_raw)),
            "rows_final": int(len(df)),
            "spread_bps": float(spread_bps),
            "cost_rt_bps": float(cost_rt_bps),
            "news_count": len(news),
            "macro_rows": 0 if macro is None else int(len(macro)),
        })


if __name__ == "__main__":
    main()
