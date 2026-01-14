import sys
import os
import re
import math
import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import streamlit as st
import yfinance as yf

import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ======================================================================================
# ATLAS | Sovereign Financial Intelligence (Integrated + Scanner + Circuit Breaker)
# ======================================================================================
APP_DIR = os.path.dirname(os.path.abspath(__file__))

EXPECTED_FILES = [
    "Mercados de Ações Américas.csv",
    "Tendências da Bolsa de Valores e Ações em Destaque.csv",
    "Preços de Commodities em Tempo Real_.csv",
    "Pares por Moeda.csv",
    "Todas criptos.txt",
    "listaativos.txt",
    "listaativos_500_extremo.txt",
]

# ======================================================================================
# STREAMLIT CONFIG + CSS (estética soberana + ênfase)
# ======================================================================================
st.set_page_config(layout="wide", page_title="ATLAS | Sovereign Financial Intelligence", page_icon="🧠")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@300;400;600;800&display=swap');

.stApp { background-color: #020202; color: #e2e8f0; font-family: 'Inter', sans-serif; }

.overlord-header {
  background: linear-gradient(135deg, #000000 0%, #0a0a0a 100%);
  padding: 34px;
  border-left: 12px solid #ff0000;
  border-radius: 10px;
  margin-bottom: 18px;
}

.section-title {
  font-family: 'JetBrains Mono';
  color: #ff0000;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: 6px;
  border-bottom: 2px solid #ff0000;
  margin-top: 18px;
  margin-bottom: 16px;
  padding-bottom: 10px;
}

.card {
  background: #070707;
  border: 1px solid #111;
  padding: 14px 16px;
  border-radius: 10px;
}

.small-mono {
  font-family: 'JetBrains Mono';
  color: #94a3b8;
  font-size: 12px;
}

.badge-green{
  display:inline-block; padding:3px 10px; border-radius:999px;
  border:1px solid #00ff41; color:#00ff41;
  font-family:'JetBrains Mono'; font-size:12px;
}
.badge-red{
  display:inline-block; padding:3px 10px; border-radius:999px;
  border:1px solid #ff0000; color:#ff0000;
  font-family:'JetBrains Mono'; font-size:12px;
}
.badge-gray{
  display:inline-block; padding:3px 10px; border-radius:999px;
  border:1px solid #334155; color:#cbd5e1;
  font-family:'JetBrains Mono'; font-size:12px;
}
.badge-blue{
  display:inline-block; padding:3px 10px; border-radius:999px;
  border:1px solid #3b82f6; color:#93c5fd;
  font-family:'JetBrains Mono'; font-size:12px;
}

.cmd-box {
  background: linear-gradient(135deg, #050505 0%, #0a0a0a 100%);
  border: 1px solid #111;
  border-left: 10px solid #00ff41;
  padding: 18px 18px;
  border-radius: 12px;
}
.cmd-box-def {
  border-left: 10px solid #ff0000;
}
.cmd-title{
  font-family:'JetBrains Mono';
  letter-spacing: 3px;
  font-weight: 800;
  font-size: 18px;
}
.cmd-main{
  font-size: 42px;
  font-weight: 900;
  line-height: 1.0;
}
.cmd-sub{
  margin-top: 10px;
  font-family:'JetBrains Mono';
  color: #cbd5e1;
  font-size: 12px;
  line-height: 1.6;
}

.kpi {
  font-size: 34px;
  font-weight: 900;
}
.kpi-label{
  font-family:'JetBrains Mono';
  color:#94a3b8;
  font-size: 12px;
}

.emph-red{ color:#ff0000; font-weight: 900; }
.emph-green{ color:#00ff41; font-weight: 900; }
.emph-blue{ color:#93c5fd; font-weight: 900; }
.emph-gray{ color:#cbd5e1; font-weight: 800; }

hr { border:none; border-top:1px solid #111; margin:16px 0; }
a { color:#00ff41 !important; }

</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ======================================================================================
# HARDENED UTILS
# ======================================================================================
def _norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _file_exists(path: str) -> bool:
    return bool(path) and os.path.exists(path) and os.path.isfile(path)

def _safe_read_text(path: str) -> str:
    if not _file_exists(path):
        return ""
    for enc in ("utf-8", "utf-8-sig", "latin1"):
        try:
            with open(path, "r", encoding=enc, errors="ignore") as f:
                return f.read()
        except Exception:
            continue
    return ""

def _safe_read_csv(path: str) -> Optional[pd.DataFrame]:
    if not _file_exists(path):
        return None
    for sep in (",", ";", "\t"):
        try:
            df = pd.read_csv(path, sep=sep, engine="python")
            if df is not None and not df.empty:
                return df
        except Exception:
            continue
    return None

def _flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df

def _force_series(x) -> pd.Series:
    if isinstance(x, pd.DataFrame):
        return x.iloc[:, 0]
    return x

def badge(text: str, color: str) -> str:
    cls = {"green": "badge-green", "red": "badge-red", "gray": "badge-gray", "blue": "badge-blue"}.get(color, "badge-gray")
    return f"<span class='{cls}'>{text}</span>"

def explain_box(title: str, meaning: str, practice: str, kind: str = "info"):
    """
    Hermenêutica didática: SEMPRE "Isso significa que..." / "Na prática..."
    """
    if not meaning.strip().lower().startswith("isso significa que"):
        meaning = "Isso significa que " + meaning
    if not practice.strip().lower().startswith("na prática"):
        practice = "Na prática, " + practice

    if kind == "success":
        st.success(f"**{title}**\n\n{meaning}\n\n{practice}")
    elif kind == "warning":
        st.warning(f"**{title}**\n\n{meaning}\n\n{practice}")
    elif kind == "error":
        st.error(f"**{title}**\n\n{meaning}\n\n{practice}")
    else:
        st.info(f"**{title}**\n\n{meaning}\n\n{practice}")

def safe_last(df: pd.DataFrame, col: str) -> Optional[float]:
    if df is None or df.empty or col not in df.columns:
        return None
    s = df[col].dropna()
    if s.empty:
        return None
    return float(s.iloc[-1])

# ======================================================================================
# TICKER EXTRACTION
# ======================================================================================
TICKER_PATTERNS = [
    re.compile(r"\b[A-Z]{4}\d{1,2}\.SA\b"),            # B3
    re.compile(r"\b[A-Z]{6}=X\b"),                     # FX
    re.compile(r"\b[A-Z]{1,3}=F\b"),                   # commodities
    re.compile(r"\^[A-Z0-9]{1,10}\b"),                 # indices
    re.compile(r"\b[A-Z0-9]{2,10}-(?:USD|BRL|EUR|USDT)\b"),  # crypto yahoo pairs
    re.compile(r"\b[A-Z]{1,5}(?:-[A-Z])?\b"),          # equities US (cuidado com falsos)
]

def _extract_tickers_from_text(blob: str) -> List[str]:
    if not blob:
        return []
    found = []
    for pat in TICKER_PATTERNS:
        found.extend(pat.findall(blob))

    cleaned = []
    for t in found:
        t = _norm_space(t)
        if len(t) <= 1:
            continue
        if t in {"OPEN", "HIGH", "LOW", "CLOSE", "VOLUME", "DATE", "TIME"}:
            continue
        # Evita palavras comuns em caps que o padrão equities poderia capturar
        if re.fullmatch(r"[A-Z]{2,5}", t) and t in {"DATA", "REAL", "BOLSA", "VALOR", "TAXA", "RISCO"}:
            continue
        cleaned.append(t)

    out = []
    seen = set()
    for t in cleaned:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out

def _extract_pairs_name_symbol_from_crypto_txt(blob: str, limit: int = 5000) -> Dict[str, str]:
    if not blob:
        return {}
    lines = [l.strip() for l in blob.splitlines() if l.strip()]
    out: Dict[str, str] = {}
    sym_re = re.compile(r"^[A-Z0-9]{2,10}$")

    i = 0
    while i < len(lines) - 1 and len(out) < limit:
        name = lines[i]
        sym = lines[i + 1]
        if sym_re.match(sym) and not sym.isdigit():
            y = f"{sym}-USD"
            out.setdefault(y, f"{_norm_space(name)} ({sym})")
            i += 2
        else:
            i += 1
    return out

def _parse_listaativos_format(blob: str) -> Dict[str, Dict[str, str]]:
    if not blob:
        return {}
    lines = [l.rstrip() for l in blob.splitlines()]
    lib: Dict[str, Dict[str, str]] = {}
    current_cat = None
    cat_re = re.compile(r"^\s*#\s*\d+\.\s*(.+?)(?:\(\d+\))?\s*$")

    for line in lines:
        l = line.strip()
        m = cat_re.match(l)
        if m:
            current_cat = _norm_space(m.group(1))
            lib.setdefault(current_cat, {})
            continue
        if not l or l.startswith("#") or l.startswith("="):
            continue

        if " — " in l:
            t, n = l.split(" — ", 1)
        elif " - " in l:
            t, n = l.split(" - ", 1)
        else:
            continue

        t = _norm_space(t)
        n = _norm_space(n)
        if current_cat is None:
            current_cat = "IMPORTADOS (SEM CATEGORIA)"
            lib.setdefault(current_cat, {})

        lib[current_cat][t] = n
    return {k: v for k, v in lib.items() if v}

def _infer_category_from_ticker(t: str) -> str:
    if t.endswith(".SA"):
        return "AÇÕES BRASILEIRAS – B3"
    if t.endswith("=X"):
        return "CÂMBIO & FX"
    if t.endswith("=F"):
        return "COMMODITIES & FUTUROS"
    if t.startswith("^"):
        return "ÍNDICES & TAXAS"
    if re.fullmatch(r"[A-Z0-9]{2,10}-(USD|BRL|EUR|USDT)", t):
        return "CRIPTOMOEDAS"
    if re.fullmatch(r"[A-Z]{1,5}(-[A-Z])?", t):
        return "AÇÕES AMÉRICAS & GLOBAIS"
    return "OUTROS"

# ======================================================================================
# BUILD ASSET LIBRARY (AUTO) + NORMALIZAÇÃO
# ======================================================================================
TARGET_8 = [
    ("CÂMBIO & FX", 63),
    ("AÇÕES BRASILEIRAS – B3", 63),
    ("AÇÕES AMÉRICAS & GLOBAIS", 63),
    ("CRIPTOMOEDAS", 63),
    ("COMMODITIES & FUTUROS", 62),
    ("ÍNDICES & TAXAS", 62),
    ("ETFs & FUNDOS", 62),
    ("OUTROS", 62),
]

def _map_to_macro(cat: str) -> str:
    c = (cat or "").upper()
    if "FX" in c or "CÂMBIO" in c or "=X" in c:
        return "CÂMBIO & FX"
    if "CRYPTO" in c or "CRIPTO" in c:
        return "CRIPTOMOEDAS"
    if "COMMOD" in c or "FUTURO" in c or "=F" in c:
        return "COMMODITIES & FUTUROS"
    if "ÍNDICE" in c or "INDEX" in c or "TAXA" in c or c.startswith("^"):
        return "ÍNDICES & TAXAS"
    if "ETF" in c or "FUND" in c or "FUNDO" in c:
        return "ETFs & FUNDOS"
    if "B3" in c or "BRASIL" in c or ".SA" in c:
        return "AÇÕES BRASILEIRAS – B3"
    if "AMÉRICAS" in c or "AMERICAS" in c or "US" in c or "NYSE" in c or "NASDAQ" in c:
        return "AÇÕES AMÉRICAS & GLOBAIS"
    if "AÇÕES" in c or "STOCK" in c or "EQUITY" in c:
        return "AÇÕES AMÉRICAS & GLOBAIS"
    return "OUTROS"

@st.cache_data(show_spinner=False)
def build_asset_library_auto(app_dir: str) -> Tuple[Dict[str, Dict[str, str]], List[str]]:
    missing = []
    for fn in EXPECTED_FILES:
        if not _file_exists(os.path.join(app_dir, fn)):
            missing.append(fn)

    lib: Dict[str, Dict[str, str]] = {}

    # Prioridade: listaativos*.txt
    for fn in ("listaativos_500_extremo.txt", "listaativos.txt"):
        path = os.path.join(app_dir, fn)
        if _file_exists(path):
            blob = _safe_read_text(path)
            parsed = _parse_listaativos_format(blob)
            for cat, items in parsed.items():
                lib.setdefault(cat, {}).update(items)

    # Criptos do TXT
    crypt_path = os.path.join(app_dir, "Todas criptos.txt")
    if _file_exists(crypt_path):
        blob = _safe_read_text(crypt_path)
        crypt = _extract_pairs_name_symbol_from_crypto_txt(blob, limit=5000)
        if crypt:
            lib.setdefault("CRIPTOMOEDAS – (ARQUIVO TXT)", {}).update(crypt)

    # CSVs: varredura textual
    csv_files = [
        "Mercados de Ações Américas.csv",
        "Tendências da Bolsa de Valores e Ações em Destaque.csv",
        "Preços de Commodities em Tempo Real_.csv",
        "Pares por Moeda.csv",
    ]
    for fn in csv_files:
        p = os.path.join(app_dir, fn)
        df = _safe_read_csv(p)
        if df is None or df.empty:
            continue
        blob = "\n".join([" ".join(map(str, row)) for row in df.head(500).values.tolist()])
        tickers = _extract_tickers_from_text(blob)
        for t in tickers:
            cat = _infer_category_from_ticker(t)
            lib.setdefault(cat, {}).setdefault(t, t)

    # Fallback mínimo
    if not lib:
        lib = {
            "CÂMBIO & FX": {"USDBRL=X": "USD/BRL"},
            "AÇÕES BRASILEIRAS – B3": {"PETR4.SA": "Petrobras PN"},
            "ÍNDICES & TAXAS": {"^BVSP": "Ibovespa"},
            "COMMODITIES & FUTUROS": {"GC=F": "Gold"},
            "CRIPTOMOEDAS": {"BTC-USD": "Bitcoin"},
            "AÇÕES AMÉRICAS & GLOBAIS": {"AAPL": "Apple"},
        }

    lib = {k: v for k, v in lib.items() if v}
    return lib, missing

def normalize_to_macro(library: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    macro: Dict[str, Dict[str, str]] = {k: {} for k, _ in TARGET_8}
    for cat, items in library.items():
        m = _map_to_macro(cat)
        for t, name in items.items():
            macro.setdefault(m, {}).setdefault(t, name)

    # Ordena determinístico
    out: Dict[str, Dict[str, str]] = {}
    for cat, _ in TARGET_8:
        items = list(macro.get(cat, {}).items())
        items.sort(key=lambda x: x[0])
        out[cat] = dict(items)
    return {k: v for k, v in out.items() if v}

def normalize_to_500_balanced(macro: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    # Corta para targets por macro
    out: Dict[str, Dict[str, str]] = {}
    for cat, target in TARGET_8:
        items = list(macro.get(cat, {}).items())
        items.sort(key=lambda x: x[0])
        out[cat] = dict(items[:target])

    # Completa faltas com pool global (sem duplicar)
    pool = []
    for cat, _ in TARGET_8:
        pool.extend(list(macro.get(cat, {}).items()))
    seen = set()
    pool2 = []
    for t, n in pool:
        if t not in seen:
            pool2.append((t, n))
            seen.add(t)

    used = set()
    for cat, _ in TARGET_8:
        used.update(out.get(cat, {}).keys())

    for cat, target in TARGET_8:
        missing = target - len(out[cat])
        if missing > 0:
            for t, n in pool2:
                if missing <= 0:
                    break
                if t in used:
                    continue
                out[cat][t] = n
                used.add(t)
                missing -= 1

    return {k: v for k, v in out.items() if v}

# ======================================================================================
# CIRCUIT BREAKER + SAFE DOWNLOAD
# ======================================================================================
@dataclass
class CircuitBreaker:
    fail_threshold: int = 2
    cooldown_seconds: int = 45
    fails: int = 0
    open_until: float = 0.0

    def is_open(self) -> bool:
        return time.time() < self.open_until

    def record_success(self):
        self.fails = 0
        self.open_until = 0.0

    def record_failure(self):
        self.fails += 1
        if self.fails >= self.fail_threshold:
            self.open_until = time.time() + self.cooldown_seconds

def _get_cb() -> CircuitBreaker:
    if "cb" not in st.session_state:
        st.session_state.cb = CircuitBreaker()
    return st.session_state.cb

@st.cache_data(show_spinner=False, ttl=60 * 10)
def yf_download_safe_cached(ticker: str, period: str, interval: str) -> pd.DataFrame:
    for attempt in range(1, 4):
        try:
            df = yf.download(
                ticker,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=False,
                threads=False,
                group_by="column",
            )
            if df is None or df.empty:
                time.sleep(0.7 * attempt)
                continue
            df = _flatten_yf_columns(df)
            for col in ["Open", "High", "Low", "Close"]:
                if col not in df.columns:
                    return pd.DataFrame()
            df = df.dropna()
            return df
        except Exception:
            time.sleep(0.9 * attempt)
            continue
    return pd.DataFrame()

def yf_download_safe(ticker: str, period: str, interval: str) -> Tuple[pd.DataFrame, str]:
    """
    Wrapper com circuit breaker e motivo textual (não trava).
    """
    cb = _get_cb()
    if cb.is_open():
        return pd.DataFrame(), "CIRCUIT_BREAKER_OPEN"

    df = yf_download_safe_cached(ticker, period, interval)
    if df is None or df.empty:
        cb.record_failure()
        return pd.DataFrame(), "YFINANCE_EMPTY_OR_FAIL"

    cb.record_success()
    return df, "OK"

# ======================================================================================
# TECHNICAL ENGINE (robusto)
# ======================================================================================
def technical_engine(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    df = _flatten_yf_columns(df)

    close = pd.to_numeric(_force_series(df.get("Close", pd.Series(dtype=float))), errors="coerce")
    high  = pd.to_numeric(_force_series(df.get("High", pd.Series(dtype=float))), errors="coerce")
    low   = pd.to_numeric(_force_series(df.get("Low", pd.Series(dtype=float))), errors="coerce")
    open_ = pd.to_numeric(_force_series(df.get("Open", pd.Series(dtype=float))), errors="coerce")

    df["Close"] = close
    df["High"]  = high
    df["Low"]   = low
    df["Open"]  = open_

    df = df.dropna(subset=["Close", "High", "Low", "Open"])
    if df.empty:
        return pd.DataFrame()

    df["SMA20"]  = df["Close"].rolling(20, min_periods=20).mean()
    df["STD20"]  = df["Close"].rolling(20, min_periods=20).std(ddof=0)

    z = (df["Close"] - df["SMA20"]) / df["STD20"]
    df["ZScore"] = pd.to_numeric(_force_series(z), errors="coerce")

    df["SMA50"]  = df["Close"].rolling(50,  min_periods=50).mean()
    df["SMA200"] = df["Close"].rolling(200, min_periods=200).mean()

    delta = df["Close"].diff()
    gain  = delta.where(delta > 0, 0.0).rolling(2, min_periods=2).mean()
    loss  = (-delta.where(delta < 0, 0.0)).rolling(2, min_periods=2).mean()
    rs = gain / (loss.replace(0, np.nan))
    df["RSI2"] = 100 - (100 / (1 + rs))

    l_h, l_l, l_c = df["High"].shift(1), df["Low"].shift(1), df["Close"].shift(1)
    df["S3"] = l_c - (l_h - l_l) * 1.1 / 4

    df["Return"] = df["Close"].pct_change()
    df["Vol20_Ann"] = df["Return"].rolling(20, min_periods=20).std(ddof=0) * np.sqrt(252)

    df["Trigger"] = (df["ZScore"] <= -2.0) & (df["RSI2"] < 15) & (df["Close"] <= df["S3"])

    # Drawdown
    roll_max = df["Close"].cummax()
    df["Drawdown"] = (df["Close"] / roll_max) - 1.0

    return df

# ======================================================================================
# RISK OF RUIN
# ======================================================================================
def risk_of_ruin_sim(
    win_rate: float,
    payoff_r: float,
    risk_per_trade: float,
    ruin_limit: float,
    horizon_trades: int,
    n_sims: int = 10000,
    seed: int = 42
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    p = float(win_rate)
    b = float(payoff_r)
    f = float(risk_per_trade)
    L = float(ruin_limit)

    p = min(max(p, 0.0001), 0.9999)
    b = max(b, 0.0001)
    f = min(max(f, 0.0), 0.99)
    L = min(max(L, 0.01), 0.99)

    if f <= 0:
        return {"ror": 0.0, "exp_log_growth": 0.0, "median_end": 1.0, "p05_end": 1.0, "p95_end": 1.0}

    wins = rng.random((n_sims, horizon_trades)) < p
    mult_win  = 1.0 + f * b
    mult_loss = 1.0 - f
    mult = np.where(wins, mult_win, mult_loss).astype(np.float64)

    eq = np.cumprod(mult, axis=1)
    ruin_level = 1.0 - L
    ruined = (eq <= ruin_level).any(axis=1)
    ror = float(ruined.mean())

    end = eq[:, -1]
    median_end = float(np.median(end))
    p05_end = float(np.quantile(end, 0.05))
    p95_end = float(np.quantile(end, 0.95))

    exp_log = p * math.log(mult_win) + (1 - p) * math.log(mult_loss)

    return {
        "ror": ror,
        "exp_log_growth": float(exp_log),
        "median_end": median_end,
        "p05_end": p05_end,
        "p95_end": p95_end,
    }

# ======================================================================================
# MONTE CARLO (500 / 30)
# ======================================================================================
def monte_carlo_paths(close: pd.Series, n_paths: int = 500, days: int = 30, seed: int = 7) -> np.ndarray:
    close = close.dropna().astype(float)
    if close.shape[0] < 40:
        return np.array([])
    rets = close.pct_change().dropna()
    mu = float(rets.mean())
    sig = float(rets.std(ddof=0))
    if sig <= 0:
        return np.array([])
    rng = np.random.default_rng(seed)
    last = float(close.iloc[-1])
    shocks = rng.normal(loc=mu, scale=sig, size=(n_paths, days))
    paths = np.zeros((n_paths, days + 1), dtype=np.float64)
    paths[:, 0] = last
    for d in range(1, days + 1):
        paths[:, d] = paths[:, d - 1] * (1.0 + shocks[:, d - 1])
    return paths

# ======================================================================================
# REGIME / VERDICT / SCORING
# ======================================================================================
def regime_from_z(z: Optional[float]) -> str:
    if z is None or math.isnan(z):
        return "Indefinido"
    if z >= 2.0:
        return "Sobrecomprado (extremo)"
    if z >= 1.0:
        return "Sobrecomprado (moderado)"
    if z <= -2.0:
        return "Sobrevendido (extremo)"
    if z <= -1.0:
        return "Sobrevendido (moderado)"
    return "Neutro"

def compute_signal_pack(df: pd.DataFrame) -> Dict[str, Optional[float]]:
    return {
        "close": safe_last(df, "Close"),
        "z": safe_last(df, "ZScore"),
        "rsi2": safe_last(df, "RSI2"),
        "vol": safe_last(df, "Vol20_Ann"),
        "dd": safe_last(df, "Drawdown"),
        "trigger": bool(df["Trigger"].dropna().iloc[-1]) if ("Trigger" in df.columns and not df["Trigger"].dropna().empty) else False,
        "sma200": safe_last(df, "SMA200"),
        "sma20": safe_last(df, "SMA20"),
    }

def score_from_pack(p: Dict[str, Optional[float]]) -> float:
    """
    Score de “ataque” (0-100): combina estresse (Z baixo), exaustão (RSI2 baixo),
    estrutura (Close vs SMA200), e penaliza vol/ drawdown extremos.
    """
    z = p.get("z")
    rsi2 = p.get("rsi2")
    vol = p.get("vol")
    dd = p.get("dd")
    close = p.get("close")
    sma200 = p.get("sma200")

    score = 50.0

    # Estresse: quanto mais negativo o Z, mais “reversão”
    if z is not None and not math.isnan(z):
        score += max(0.0, min(25.0, (-z) * 8.0))   # z=-3 -> +24
        score -= max(0.0, min(25.0, (z) * 6.0))    # z=+3 -> -18

    # Exaustão: RSI2 baixo puxa score
    if rsi2 is not None and not math.isnan(rsi2):
        if rsi2 < 15:
            score += 18
        elif rsi2 < 30:
            score += 8
        elif rsi2 > 85:
            score -= 14

    # Estrutura: acima da SMA200 é “campo favorável”
    if close is not None and sma200 is not None and not math.isnan(sma200):
        if close >= sma200:
            score += 10
        else:
            score -= 10

    # Penalidade por volatilidade hostil
    if vol is not None and not math.isnan(vol):
        if vol > 0.80:
            score -= 18
        elif vol > 0.60:
            score -= 10

    # Penalidade por drawdown profundo (campo minado)
    if dd is not None and not math.isnan(dd):
        if dd < -0.35:
            score -= 14
        elif dd < -0.20:
            score -= 6

    return float(max(0.0, min(100.0, score)))

def verdict(trigger: bool, ror: float, score: float, vol: Optional[float]) -> Tuple[str, str, str]:
    """
    Veredito final com comando:
    - EXECUTAR se: trigger e ror <= 25% e score alto e vol não explosiva
    - DEFENDER caso contrário
    """
    ror_ok = ror <= 0.25
    score_ok = score >= 65
    vol_ok = (vol is None) or (vol <= 0.65)

    if trigger and ror_ok and score_ok and vol_ok:
        return ("EXECUÇÃO AUTORIZADA", "green", "ATAQUE")
    return ("OBSERVAÇÃO DEFENSIVA", "red", "DEFESA")

# ======================================================================================
# CHECKBOX SELECTOR (com busca + paginação)
# ======================================================================================
def checkbox_selector(
    category: str,
    items: List[Tuple[str, str]],
    key_prefix: str,
    default_selected: Optional[List[str]] = None,
    page_size: int = 50,
) -> List[str]:
    default_selected = default_selected or []
    st.sidebar.markdown(f"### {category}")

    q = st.sidebar.text_input(
        "Buscar ticker/nome",
        value="",
        key=f"{key_prefix}_search",
        help="Isso significa filtro local sem travar. Na prática, use parte do ticker (PETR) ou do nome (Petrobras)."
    ).strip().lower()

    filtered = items
    if q:
        filtered = [(t, n) for (t, n) in items if (q in t.lower()) or (q in (n or "").lower())]

    total = len(filtered)
    if total == 0:
        st.sidebar.caption("Na prática, nenhum ativo bate com esse filtro.")
        return []

    pages = max(1, math.ceil(total / page_size))
    page = st.sidebar.number_input(
        "Página",
        min_value=1,
        max_value=pages,
        value=1,
        step=1,
        key=f"{key_prefix}_page",
        help="Isso significa que o app não precisa renderizar centenas de checkboxes de uma vez. Na prática, isso evita travamentos no front-end."
    )
    start = (page - 1) * page_size
    end = min(start + page_size, total)
    visible = filtered[start:end]

    state_key = f"{key_prefix}_selected"
    if state_key not in st.session_state:
        st.session_state[state_key] = set(default_selected)

    c1, c2 = st.sidebar.columns(2)
    if c1.button("Selecionar visíveis", key=f"{key_prefix}_sel_all"):
        for t, _ in visible:
            st.session_state[state_key].add(t)
    if c2.button("Limpar visíveis", key=f"{key_prefix}_clr_all"):
        for t, _ in visible:
            st.session_state[state_key].discard(t)

    st.sidebar.caption(f"Visíveis: {start+1}-{end} de {total}")
    for t, n in visible:
        label = f"{t} — {n}" if n else t
        ck = st.sidebar.checkbox(
            label,
            value=(t in st.session_state[state_key]),
            key=f"{key_prefix}_ck_{t}",
            help="Na prática, marcar aqui adiciona o ativo ao Scanner e aos rankings."
        )
        if ck:
            st.session_state[state_key].add(t)
        else:
            st.session_state[state_key].discard(t)

    return sorted(list(st.session_state[state_key]))

# ======================================================================================
# SCANNER (varredura dos selecionados) com cache + limite + progresso
# ======================================================================================
@st.cache_data(show_spinner=False, ttl=60 * 10)
def scan_one_ticker(ticker: str) -> Dict[str, object]:
    """
    Scanner rápido: usa janela fixa 6mo/1d para não explodir custo.
    Retorna pacote de sinais para ranking.
    """
    df_raw, status = yf_download_safe_cached(ticker, "6mo", "1d"), "OK"
    if df_raw is None or df_raw.empty:
        return {"ticker": ticker, "status": "EMPTY", "score": np.nan}
    df = technical_engine(df_raw)
    if df is None or df.empty:
        return {"ticker": ticker, "status": "ENGINE_EMPTY", "score": np.nan}

    p = compute_signal_pack(df)
    sc = score_from_pack(p)
    return {
        "ticker": ticker,
        "status": "OK",
        "score": sc,
        "close": p["close"],
        "z": p["z"],
        "rsi2": p["rsi2"],
        "vol": p["vol"],
        "dd": p["dd"],
        "trigger": p["trigger"],
        "regime": regime_from_z(p["z"]),
    }

def run_scanner(tickers: List[str], max_per_cycle: int, diag: bool) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()

    # Limita por ciclo para não “travar” em redes/finance
    tickers = tickers[:max_per_cycle]

    prog = st.progress(0)
    box = st.empty()

    out = []
    n = len(tickers)
    for i, t in enumerate(tickers, start=1):
        box.info(f"Scanner: processando {t} ({i}/{n})...")
        try:
            out.append(scan_one_ticker(t))
        except Exception:
            out.append({"ticker": t, "status": "EXCEPTION", "score": np.nan})
        prog.progress(int(i / n * 100))

    box.success("Scanner finalizado.")
    prog.empty()

    df = pd.DataFrame(out)
    if not df.empty:
        df["score"] = pd.to_numeric(df["score"], errors="coerce")
        df = df.sort_values(by="score", ascending=False, na_position="last")
    return df

# ======================================================================================
# MAIN
# ======================================================================================
def main():
    st.markdown(
        f"""
        <div class="overlord-header">
          <div style="display:flex; align-items:center; justify-content:space-between; gap:12px;">
            <div>
              <div style="font-family:'JetBrains Mono'; color:#ff0000; font-weight:900; letter-spacing:6px; font-size:20px;">
                ATLAS | SOVEREIGN FINANCIAL INTELLIGENCE
              </div>
              <div class="small-mono">Scanner + Circuit Breaker + Command Center. Seleção por checkbox com universo automático.</div>
            </div>
            <div>{badge("UI ONLINE", "green")} {badge("INTEGRATED", "blue")} {badge("BOOT SAFE", "gray")}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # -------------------------------------
    # Sidebar: Diagnóstico e parâmetros (com balões)
    # -------------------------------------
    st.sidebar.markdown("## Diagnóstico")
    diag = st.sidebar.checkbox(
        "Modo diagnóstico",
        value=True,
        help="Isso significa checkpoints no sidebar e mensagens explícitas. Na prática, você sempre sabe onde travou."
    )

    def DIAG(msg: str):
        if diag:
            st.sidebar.write(f"• {msg}")

    DIAG("Boot: script carregado")
    DIAG(f"Python: {sys.version.split()[0]}")
    DIAG(f"Pasta do app: {APP_DIR}")

    st.sidebar.markdown("## Parâmetros de Dados")
    period = st.sidebar.selectbox(
        "Período",
        ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"],
        index=4,
        help="Isso significa o tamanho da janela histórica do ATIVO ATIVO. Na prática, 2y é equilíbrio (estrutura + estabilidade)."
    )
    interval = st.sidebar.selectbox(
        "Intervalo",
        ["1d", "1wk", "1mo"],
        index=0,
        help="Isso significa a granularidade do candle. Na prática, 1d é padrão para evitar retornos vazios/travamentos."
    )

    st.sidebar.markdown("## Risco de Ruína (inputs)")
    win_rate = st.sidebar.slider(
        "Win Rate estimado",
        0.10, 0.90, 0.55, 0.01,
        help="Isso significa taxa de acerto estimada. Na prática, superestimar aqui mascara risco real e te empurra para ruína."
    )
    payoff = st.sidebar.slider(
        "Payoff médio (R-múltiplo)",
        0.50, 5.00, 1.60, 0.05,
        help="Isso significa ganho médio por acerto em múltiplos de risco. Na prática, payoff baixo exige win-rate alto para sobreviver."
    )
    rpt = st.sidebar.slider(
        "Risk-per-Trade",
        0.00, 0.05, 0.01, 0.001,
        help="Isso significa fração do capital arriscada por trade. Na prática, 1% é agressivo; >2% acelera ruína via variância."
    )
    ruin_limit = st.sidebar.slider(
        "Limite de ruína (perda do capital)",
        0.10, 0.90, 0.30, 0.01,
        help="Isso significa a perda considerada 'quebra'. Na prática, 30% já compromete o psicológico e a curva de recuperação."
    )
    horizon = st.sidebar.number_input(
        "Horizonte (trades)",
        min_value=50, max_value=2000, value=200, step=10,
        help="Isso significa o número de trades no qual você quer sobreviver estatisticamente. Na prática, 200+ revela caudas (riscos raros)."
    )

    # Scanner settings
    st.sidebar.markdown("## Scanner (varredura)")
    scan_limit = st.sidebar.slider(
        "Máx. tickers por ciclo (estabilidade)",
        10, 200, 60, 5,
        help="Isso significa quantos tickers o Scanner processa por execução. Na prática, limite evita travar rede/Yahoo em lote."
    )

    # -------------------------------------
    # Biblioteca automática (expansão por arquivos locais)
    # -------------------------------------
    DIAG("Carregando biblioteca automática...")
    library_raw, missing = build_asset_library_auto(APP_DIR)
    macro_full = normalize_to_macro(library_raw)
    library = normalize_to_500_balanced(macro_full)  # 500 balanceados (quando houver material)
    total_assets = sum(len(v) for v in library.values())
    DIAG(f"Total ativos (500 balanceado): {total_assets}")

    if missing:
        st.warning(
            "Isso significa que alguns arquivos esperados não foram encontrados na pasta do projeto. "
            "Na prática, o universo pode ficar menor. Se você quiser o máximo, copie os arquivos faltantes para a pasta do app.\n\n"
            f"Arquivos não encontrados: {missing}"
        )

    # -------------------------------------
    # Seleção por checkbox
    # -------------------------------------
    st.sidebar.markdown("## Seleção de Ativos")
    categories = sorted(list(library.keys()))
    default_cat = "AÇÕES BRASILEIRAS – B3" if "AÇÕES BRASILEIRAS – B3" in categories else categories[0]
    active_cat = st.sidebar.selectbox(
        "Categoria",
        categories,
        index=categories.index(default_cat),
        help="Isso significa a classe operacional. Na prática, selecione a macro-categoria e marque tickers para varredura/monitoramento."
    )

    items = list(library[active_cat].items())
    selected = checkbox_selector(
        active_cat,
        items,
        key_prefix=f"atlas_{active_cat}",
        default_selected=(["PETR4.SA"] if "PETR4.SA" in dict(items) else []),
        page_size=50
    )

    active_ticker = st.sidebar.selectbox(
        "Ativo ativo (análise detalhada)",
        options=(selected if selected else [items[0][0]]),
        index=0,
        help="Isso significa o alvo de profundidade total. Na prática, o ATIVO ATIVO guia decisão final, enquanto o Scanner guia o funil."
    )

    # =====================================
    # Estado Operacional
    # =====================================
    st.markdown("<div class='section-title'>Estado Operacional</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)

    c1.markdown(f"<div class='card'>{badge('UNIVERSO', 'gray')}<div class='kpi-label'>Ativos balanceados</div><div class='kpi'>{total_assets}</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='card'>{badge('CATEGORIA', 'gray')}<div class='kpi-label'>Ativa</div><div style='font-size:18px; font-weight:900;'>{active_cat}</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='card'>{badge('SELECIONADOS', 'blue')}<div class='kpi-label'>Tickers marcados</div><div class='kpi'>{len(selected)}</div></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='card'>{badge('ATIVO ATIVO', 'gray')}<div class='kpi-label'>Foco</div><div class='kpi'>{active_ticker}</div></div>", unsafe_allow_html=True)

    explain_box(
        "Arquitetura de Decisão",
        "o sistema separa FUNIL (Scanner nos tickers marcados) e PROFUNDIDADE (Ativo ativo).",
        "use o Scanner para encontrar candidatos e o Ativo ativo para confirmar o regime, risco e comando final."
    )

    # =====================================
    # PIPELINE DO ATIVO ATIVO (com CB)
    # =====================================
    st.markdown("<div class='section-title'>Pipeline do Ativo Ativo</div>", unsafe_allow_html=True)
    status_box = st.empty()
    prog = st.progress(0)

    DIAG("Executando download do ativo ativo...")
    status_box.info("Carregando dados do Yahoo Finance (com circuit breaker)...")
    prog.progress(15)

    df_raw, st_code = yf_download_safe(active_ticker, period, interval)
    prog.progress(45)

    if st_code == "CIRCUIT_BREAKER_OPEN":
        prog.progress(100)
        explain_box(
            "Circuit Breaker Ativo",
            "o sistema detectou falhas repetidas e abriu um cooldown para evitar travar.",
            "aguarde alguns segundos e recarregue (tecla R no terminal do Streamlit), ou reduza o lote do Scanner.",
            kind="warning"
        )
        st.stop()

    if df_raw.empty:
        prog.progress(100)
        explain_box(
            "Falha de Dados (Yahoo vazio)",
            "o Yahoo não retornou OHLC para o ticker/período/intervalo escolhido.",
            "troque para intervalo 1d e período 2y, ou teste outro ticker. Isso não é visual: é retorno vazio de dados.",
            kind="error"
        )
        st.stop()

    DIAG(f"OHLC OK: {len(df_raw)} linhas")
    status_box.success("Dados carregados. Calculando indicadores...")
    prog.progress(60)

    df = technical_engine(df_raw)
    if df.empty:
        prog.progress(100)
        explain_box(
            "Engine Técnico ficou vazio",
            "após limpeza mínima, não sobrou série consistente para cálculo.",
            "aumente o período e mantenha 1d; esse erro ocorre em ativos sem histórico limpo.",
            kind="error"
        )
        st.stop()

    prog.progress(100)
    status_box.success("Pipeline concluído. Interface pronta.")

    # =====================================
    # SCANNER TAB + TABS PRINCIPAIS
    # =====================================
    tabs = st.tabs([
        "0) Scanner (selecionados)",
        "1) Diagnóstico Visual",
        "2) Indicadores & Hermenêutica",
        "3) Risco de Ruína",
        "4) Monte Carlo (30d / 500)",
        "5) Veredito (Comando Final)"
    ])

    # =============================================================================
    # TAB 0 — Scanner
    # =============================================================================
    with tabs[0]:
        st.markdown("<div class='section-title'>Scanner (Varredura Operacional)</div>", unsafe_allow_html=True)

        if not selected:
            explain_box(
                "Scanner sem universo",
                "nenhum ticker foi marcado no checklist.",
                "marque alguns tickers na sidebar para o Scanner gerar ranking e oportunidades."
            )
            st.stop()

        explain_box(
            "Como o Scanner funciona",
            "o Scanner calcula sinais rápidos (6mo/1d) para cada ticker marcado e produz um SCORE 0–100.",
            "use o ranking para escolher o próximo Ativo ativo. O objetivo é reduzir o espaço de busca e concentrar fogo no edge."
        )

        # Rodar scanner automaticamente (como você pediu), mas limitado por ciclo para estabilidade
        st.caption("Na prática, o Scanner roda com limite por ciclo para evitar travar rede/Yahoo em lote.")

        df_scan = run_scanner(selected, max_per_cycle=int(scan_limit), diag=diag)

        if df_scan.empty:
            explain_box(
                "Scanner não retornou dados",
                "os tickers do ciclo retornaram vazio ou falharam no Yahoo.",
                "reduza o lote, teste outra categoria e confirme conexão.",
                kind="warning"
            )
            st.stop()

        # Destaques visuais
        top_ok = df_scan[df_scan["status"] == "OK"].head(10).copy()
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"{badge('TOP 10 SCORE', 'green')} <span class='small-mono'>alvos mais promissores do ciclo</span>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.dataframe(
            top_ok[["ticker", "score", "regime", "z", "rsi2", "vol", "dd", "trigger", "status"]],
            use_container_width=True
        )

        explain_box(
            "Leitura do SCORE",
            "score alto tende a indicar convergência de estresse + estrutura favorável, mas não substitui o veredito do Ativo ativo.",
            "pegue os Top 3 e teste como Ativo ativo para confirmar risco de ruína, regime e comando final."
        )

    # =============================================================================
    # TAB 1 — Diagnóstico Visual
    # =============================================================================
    with tabs[1]:
        st.markdown("<div class='section-title'>Diagnóstico Visual</div>", unsafe_allow_html=True)

        pack = compute_signal_pack(df)
        last_close = pack["close"]
        last_z = pack["z"]
        last_vol = pack["vol"]
        last_dd = pack["dd"]
        trig = pack["trigger"]

        r = regime_from_z(last_z)

        a, b, c, d = st.columns(4)
        a.metric("Preço Atual", f"{last_close:,.4f}" if last_close is not None else "—",
                 help="Isso significa o último fechamento. Na prática, é a referência central de níveis e risco.")
        b.metric("Z-Score (20)", f"{last_z:,.2f}" if last_z is not None else "—",
                 help="Isso significa desvios-padrão da SMA20. Na prática, |Z| alto = estiramento (tensão de reversão).")
        c.metric("Vol (20) a.a.", f"{last_vol*100:,.1f}%" if last_vol is not None else "—",
                 help="Isso significa risco de dispersão. Na prática, vol alta exige reduzir tamanho para sobreviver.")
        d.metric("Trigger", "ATIVO" if trig else "INATIVO",
                 help="Isso significa convergência de condições de reversão. Na prática, trigger ativo = atenção máxima.")

        explain_box(
            "Leitura do Regime",
            f"o regime atual está classificado como: {r}.",
            "use SMA200 para estrutura e Z/RSI2 para timing. Estrutura manda; timing dispara."
        )

        fig = make_subplots(
            rows=4, cols=1, shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.55, 0.15, 0.15, 0.15]
        )

        fig.add_trace(go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
            name="Preço"
        ), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"], name="SMA20"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"], name="SMA50"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA200"], name="SMA200"), row=1, col=1)

        fig.add_trace(go.Scatter(x=df.index, y=df["ZScore"], name="ZScore"), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI2"], name="RSI2"), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["Drawdown"], name="Drawdown"), row=4, col=1)

        fig.update_layout(
            height=880,
            paper_bgcolor="#020202",
            plot_bgcolor="#020202",
            font=dict(color="#e2e8f0", family="Inter"),
            margin=dict(l=10, r=10, t=30, b=10),
            title=f"{active_ticker} | Preço + Médias + ZScore + RSI2 + Drawdown"
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridcolor="#111")

        st.plotly_chart(fig, use_container_width=True)

        explain_box(
            "Interpretação do Painel",
            "a primeira faixa (Preço + SMAs) define estrutura; as faixas seguintes definem tensão (ZScore), exaustão (RSI2) e dano acumulado (Drawdown).",
            "se Z e RSI2 apontam estresse, mas a estrutura (SMA200) é contrária, trate como contra-tendência e reduza risco."
        )

        cols_show = ["Open", "High", "Low", "Close", "SMA20", "STD20", "ZScore", "RSI2", "Vol20_Ann", "Drawdown", "Trigger"]
        st.dataframe(df[cols_show].tail(14), use_container_width=True)

    # =============================================================================
    # TAB 2 — Indicadores & Hermenêutica
    # =============================================================================
    with tabs[2]:
        st.markdown("<div class='section-title'>Indicadores & Hermenêutica</div>", unsafe_allow_html=True)

        pack = compute_signal_pack(df)
        z = pack["z"]
        rsi2 = pack["rsi2"]
        vol = pack["vol"]
        dd = pack["dd"]
        trig = pack["trigger"]

        sc = score_from_pack(pack)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"{badge('SCORE', 'blue')} <span class='emph-blue'>{sc:,.1f}/100</span>", unsafe_allow_html=True)
        st.markdown(f"{badge('REGIME', 'gray')} <span class='emph-gray'>{regime_from_z(z)}</span>", unsafe_allow_html=True)
        st.markdown(f"{badge('TRIGGER', 'green' if trig else 'red')} <span class='emph-green'>ATIVO</span>" if trig else f"{badge('TRIGGER', 'red')} <span class='emph-red'>INATIVO</span>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # ZScore
        if z is None or math.isnan(z):
            explain_box("ZScore", "o ZScore não pôde ser calculado (histórico insuficiente).", "aumente período e use 1d.", kind="warning")
        else:
            if z <= -2:
                explain_box("ZScore (estresse vendedor)", f"o preço está {abs(z):.2f} desvios ABAIXO da SMA20.", "isso eleva probabilidade de reversão tática, mas exige validação de estrutura.", kind="success")
            elif z >= 2:
                explain_box("ZScore (estresse comprador)", f"o preço está {abs(z):.2f} desvios ACIMA da SMA20.", "isso aumenta risco de correção; evite perseguir preço sem pullback.", kind="warning")
            else:
                explain_box("ZScore (faixa normal)", "o preço está em faixa estatisticamente comum ao redor da SMA20.", "o edge tende a vir mais de estrutura (SMA200) do que de reversão imediata.")

        # RSI2
        if rsi2 is None or math.isnan(rsi2):
            explain_box("RSI2", "o RSI2 não pôde ser calculado.", "aumente período e use 1d.", kind="warning")
        else:
            if rsi2 < 15:
                explain_box("RSI2 (exaustão)", f"RSI2 em {rsi2:.1f} indica EXAUSTÃO de curtíssimo prazo.", "quando combinado com ZScore baixo, o setup de reversão fica mais plausível.", kind="success")
            elif rsi2 > 85:
                explain_box("RSI2 (superaquecimento)", f"RSI2 em {rsi2:.1f} indica SUPERCOMPRA extrema.", "entrar aqui sem proteção vira aposta contra a média.", kind="warning")
            else:
                explain_box("RSI2 (neutro)", f"RSI2 em {rsi2:.1f} não aponta extremo.", "o timing deve vir de estrutura e confirmação.")

        # Vol
        if vol is None or math.isnan(vol):
            explain_box("Volatilidade", "a volatilidade não pôde ser calculada.", "aumente período e use 1d.", kind="warning")
        else:
            if vol > 0.70:
                explain_box("Volatilidade (hostil)", f"vol anualizada ~{vol*100:.1f}% indica regime hostil.", "reduza tamanho e exija triggers mais fortes; variância destrói capital.", kind="warning")
            elif vol < 0.25:
                explain_box("Volatilidade (contida)", f"vol anualizada ~{vol*100:.1f}% indica regime mais contido.", "stops podem ser mais curtos e sizing mais eficiente.", kind="success")
            else:
                explain_box("Volatilidade (intermediária)", f"vol anualizada ~{vol*100:.1f}%.", "use sizing e stops proporcionais ao regime.")

        # Drawdown
        if dd is not None and not math.isnan(dd):
            if dd < -0.35:
                explain_box("Drawdown (campo minado)", f"drawdown atual ~{dd*100:.1f}% sugere dano profundo.", "evite agressão sem confirmação: estrutura pode estar quebrada.", kind="warning")
            else:
                explain_box("Drawdown", f"drawdown atual ~{dd*100:.1f}%.", "use drawdown como filtro para não comprar euforia/queda estrutural.")

        # Trigger
        if trig:
            explain_box("Trigger", "o gatilho técnico está ATIVO (convergência de estresse).", "isso aumenta o potencial de entrada tática, desde que risco e estrutura estejam aceitáveis.", kind="success")
        else:
            explain_box("Trigger", "o gatilho técnico está INATIVO.", "sem convergência, a chance de entrar sem edge aumenta. Prefira esperar o alinhamento.", kind="warning")

    # =============================================================================
    # TAB 3 — Risk of Ruin
    # =============================================================================
    with tabs[3]:
        st.markdown("<div class='section-title'>Risco de Ruína</div>", unsafe_allow_html=True)

        explain_box(
            "O que é Ruína",
            "aqui calculamos a probabilidade de sua equity cruzar o limite de quebra ao longo de N trades, dadas suas premissas.",
            "se a ruína for alta, o sistema deve priorizar DEFESA mesmo que o setup pareça bonito."
        )

        res = risk_of_ruin_sim(
            win_rate=win_rate, payoff_r=payoff,
            risk_per_trade=rpt, ruin_limit=ruin_limit,
            horizon_trades=int(horizon),
            n_sims=10000, seed=42
        )

        ror = res["ror"]
        a, b, c, d = st.columns(4)
        a.metric("Prob. de Ruína", f"{ror*100:,.1f}%")
        b.metric("Crescimento log/trade", f"{res['exp_log_growth']:.5f}")
        c.metric("Equity final (mediana)", f"{res['median_end']:.2f}x")
        d.metric("Faixa 5%-95%", f"{res['p05_end']:.2f}x → {res['p95_end']:.2f}x")

        if ror >= 0.40:
            explain_box("Ruína CRÍTICA", f"ruína em {ror*100:.1f}% indica cenário estatisticamente letal.", "reduza risk-per-trade e exija edge superior; operar assim é sangrar.", kind="error")
        elif ror >= 0.25:
            explain_box("Ruína ALTA", f"ruína em {ror*100:.1f}% indica risco elevado.", "reduza risco e aumente seletividade; sobreviver aqui depende de sorte.", kind="warning")
        else:
            explain_box("Ruína CONTROLADA", f"ruína em {ror*100:.1f}% sob as premissas.", "mantenha disciplina: stops e sizing são obrigatórios.", kind="success")

    # =============================================================================
    # TAB 4 — Monte Carlo
    # =============================================================================
    with tabs[4]:
        st.markdown("<div class='section-title'>Monte Carlo (30 dias / 500 trajetórias)</div>", unsafe_allow_html=True)

        paths = monte_carlo_paths(df["Close"], n_paths=500, days=30, seed=7)
        if paths.size == 0:
            explain_box(
                "Monte Carlo indisponível",
                "não há histórico suficiente ou variância para simular caminhos.",
                "aumente período e mantenha 1d para obter retornos estáveis.",
                kind="warning"
            )
            st.stop()

        p10 = np.quantile(paths, 0.10, axis=0)
        p50 = np.quantile(paths, 0.50, axis=0)
        p90 = np.quantile(paths, 0.90, axis=0)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(paths.shape[1])), y=p90, line=dict(width=0), showlegend=False, name="P90"))
        fig.add_trace(go.Scatter(x=list(range(paths.shape[1])), y=p10, fill="tonexty", line=dict(width=0), name="P10–P90"))
        fig.add_trace(go.Scatter(x=list(range(paths.shape[1])), y=p50, name="Mediana"))

        fig.update_layout(
            height=560,
            paper_bgcolor="#020202",
            plot_bgcolor="#020202",
            font=dict(color="#e2e8f0", family="Inter"),
            margin=dict(l=10, r=10, t=30, b=10),
            title=f"{active_ticker} | Faixa Monte Carlo 30d"
        )
        fig.update_xaxes(title="Dias à frente", showgrid=False)
        fig.update_yaxes(title="Preço", showgrid=True, gridcolor="#111")

        st.plotly_chart(fig, use_container_width=True)

        explain_box(
            "Leitura da Banda P10–P90",
            "a banda representa dispersão estatística estimada a partir do passado; não é promessa, é risco quantificado.",
            "se seu plano exige atravessar P10 com risco alto, o sizing deve cair: sobrevivência vence convicção."
        )

    # =============================================================================
    # TAB 5 — Veredito / Comando Final (ênfase visual)
    # =============================================================================
    with tabs[5]:
        st.markdown("<div class='section-title'>Veredito (Comando Final)</div>", unsafe_allow_html=True)

        pack = compute_signal_pack(df)
        sc = score_from_pack(pack)

        ror = risk_of_ruin_sim(
            win_rate=win_rate, payoff_r=payoff,
            risk_per_trade=rpt, ruin_limit=ruin_limit,
            horizon_trades=int(horizon),
            n_sims=8000, seed=11
        )["ror"]

        verdict_text, verdict_color, posture = verdict(pack["trigger"], ror, sc, pack["vol"])

        # Command Center visual
        if verdict_text == "EXECUÇÃO AUTORIZADA":
            st.markdown(
                f"""
                <div class="cmd-box">
                  <div class="cmd-title">{badge("COMANDO", "green")} <span class="emph-green">ATAQUE</span></div>
                  <div class="cmd-main"><span class="emph-green">EXECUTAR</span></div>
                  <div class="cmd-sub">
                    <span class="emph-gray">Critérios:</span>
                    Trigger <span class="emph-green">ATIVO</span> • Ruína <span class="emph-green">{ror*100:.1f}%</span> • Score <span class="emph-blue">{sc:.1f}</span>
                  </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            explain_box(
                "Diretriz",
                "a convergência de edge + risco estatístico aceitável permite execução tática.",
                "execute apenas com sizing consistente com Risk-per-Trade e stop obrigatório; sem disciplina, o edge vira ruído.",
                kind="success"
            )
        else:
            st.markdown(
                f"""
                <div class="cmd-box cmd-box-def">
                  <div class="cmd-title">{badge("COMANDO", "red")} <span class="emph-red">DEFESA</span></div>
                  <div class="cmd-main"><span class="emph-red">DEFENDER</span></div>
                  <div class="cmd-sub">
                    <span class="emph-gray">Gatilhos de defesa:</span>
                    Trigger <span class="emph-red">INCOMPLETO</span> ou Ruína <span class="emph-red">{ror*100:.1f}%</span> ou Volatilidade <span class="emph-red">HOSTIL</span> ou Score <span class="emph-red">{sc:.1f}</span>
                  </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # FRASE CORRIGIDA + COM ÊNFASE VISUAL (pedido seu)
            st.markdown(
                f"""
                <div class="card">
                  {badge("OBSERVAÇÃO DEFENSIVA", "red")}
                  <div style="margin-top:10px; line-height:1.9;">
                    <span class="emph-gray">Na prática</span>, a estrutura atual recomenda <span class="emph-red">DEFESA</span>:
                    ou o <span class="emph-red">TRIGGER</span> não está completo, ou o <span class="emph-red">RISCO ESTATÍSTICO</span> está alto,
                    ou a <span class="emph-red">VOLATILIDADE</span> está em regime hostil.
                    <br/>
                    Operar aqui é <span class="emph-red">pagar spread</span> para comprar <span class="emph-red">incerteza</span>.
                  </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            explain_box(
                "Diretriz",
                "o sistema não enxerga convergência suficiente entre edge e sobrevivência.",
                "não force entrada: aguarde alinhamento ou reduza risco/tamanho até o ambiente deixar de ser hostil.",
                kind="error"
            )

        # Painel resumido de decisão
        st.markdown("<div class='section-title'>Resumo Quantitativo</div>", unsafe_allow_html=True)
        cols = st.columns(5)
        cols[0].metric("Score", f"{sc:.1f}/100")
        cols[1].metric("Ruína", f"{ror*100:.1f}%")
        cols[2].metric("ZScore", f"{pack['z']:.2f}" if pack["z"] is not None else "—")
        cols[3].metric("RSI2", f"{pack['rsi2']:.1f}" if pack["rsi2"] is not None else "—")
        cols[4].metric("Vol a.a.", f"{pack['vol']*100:.1f}%" if pack["vol"] is not None else "—")

        explain_box(
            "Como usar o comando",
            "o comando final é uma síntese de EDGE (Trigger/Score) e SOBREVIVÊNCIA (Ruína/Vol).",
            "se DEFENDER: preserve capital e espere. Se EXECUTAR: execute com sizing e stop; o sistema não tolera indisciplina."
        )

# ======================================================================================
# ENTRYPOINT
# ======================================================================================
if __name__ == "__main__":
    main()
