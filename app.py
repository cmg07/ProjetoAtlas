from __future__ import annotations

import csv
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf


APP_TITLE = "ATLAS v5"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
UNIVERSE_PATH = os.path.join(DATA_DIR, "universe.csv")
BRIDGE_PATH = os.path.join(DATA_DIR, "bridge_prices.csv")
LOCAL_PRICE_PATH = os.path.join(DATA_DIR, "local_prices.csv")
LEDGER_PATH = os.path.join(os.path.dirname(__file__), ".atlas_ledger.csv")

OFFICIAL_CATEGORIES = [
    "Ações Brasil",
    "Ações Globais",
    "Mercado de Câmbio",
    "Cripto",
    "Commodities",
    "Índices",
    "ETFs",
    "Juros Brasil",
    "Renda Fixa",
]

CSS = """
<style>
:root {
  color-scheme: dark;
}
html, body, [class*="css"]  {
  background: #0b0e14 !important;
  color: #e6edf3 !important;
  font-family: "Roboto Mono", "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
section[data-testid="stSidebar"]{
  background: #05070b !important;
  border-right: 1px solid rgba(255,255,255,0.08);
}
.atlas-title {
  font-size: 38px;
  font-weight: 800;
  letter-spacing: 1px;
  margin-bottom: 2px;
}
.atlas-subtitle {
  color: rgba(230,237,243,0.7);
  font-size: 12px;
  letter-spacing: 1px;
  text-transform: uppercase;
}
.ticker-tape {
  display: flex;
  gap: 22px;
  overflow: hidden;
  white-space: nowrap;
  background: #020308;
  padding: 6px 10px;
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 8px;
  margin-bottom: 12px;
}
.ticker-item {
  color: #5efc8d;
  font-size: 12px;
}
.atlas-card {
  background: linear-gradient(160deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 12px;
  padding: 12px;
  box-shadow: 0 12px 30px rgba(0,0,0,0.35);
}
.atlas-kpi-title {
  text-transform: uppercase;
  font-size: 12px;
  color: rgba(230,237,243,0.6);
  margin-bottom: 6px;
}
.atlas-kpi-value {
  font-size: 24px;
  font-weight: 700;
}
.atlas-kpi-foot {
  font-size: 11px;
  color: rgba(230,237,243,0.5);
}
.atlas-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 999px;
  padding: 4px 10px;
  font-size: 11px;
  text-transform: uppercase;
}
.atlas-alert {
  border-left: 4px solid #ff6b6b;
  background: rgba(255,107,107,0.08);
  padding: 12px;
  border-radius: 8px;
}
.atlas-ok {
  border-left: 4px solid #00d084;
  background: rgba(0,208,132,0.08);
  padding: 12px;
  border-radius: 8px;
}
</style>
"""


@dataclass
class UniverseEntry:
    name: str
    category: str
    ticker_yahoo: str
    currency_code: str
    base: str
    country: str = ""
    exchange: str = ""
    priority_source: str = "yahoo"
    bridge_key: str = ""


class Utils:
    @staticmethod
    def safe_float(val: object, default: float = 0.0) -> float:
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def ensure_data_dir() -> None:
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR, exist_ok=True)


class UniverseEngine:
    @staticmethod
    def ensure_default_universe() -> None:
        Utils.ensure_data_dir()
        if os.path.exists(UNIVERSE_PATH):
            return

        default_rows = [
            ["PETROBRAS PN", "Ações Brasil", "PETR4", "BRL", "BR"],
            ["VALE ON", "Ações Brasil", "VALE3", "BRL", "BR"],
            ["ITAU PN", "Ações Brasil", "ITUB4", "BRL", "BR"],
            ["B3 ON", "Ações Brasil", "B3SA3", "BRL", "BR"],
            ["USD/BRL", "Mercado de Câmbio", "BRL=X", "BRL", "BR"],
            ["EUR/BRL", "Mercado de Câmbio", "EURBRL=X", "BRL", "BR"],
            ["S&P 500", "Índices", "^GSPC", "USD", "US"],
            ["IBOV", "Índices", "^BVSP", "BRL", "BR"],
            ["BTC", "Cripto", "BTC-USD", "USD", "GLOBAL"],
            ["ETH", "Cripto", "ETH-USD", "USD", "GLOBAL"],
            ["Ouro", "Commodities", "GC=F", "USD", "GLOBAL"],
            ["IVVB11", "ETFs", "IVVB11", "BRL", "BR"],
            ["BOVA11", "ETFs", "BOVA11", "BRL", "BR"],
            ["SELIC", "Juros Brasil", "SELIC", "BRL", "BR"],
            ["CDI", "Juros Brasil", "CDI", "BRL", "BR"],
        ]

        header = [
            "name",
            "category",
            "ticker_yahoo",
            "currency_code",
            "base",
            "country",
            "exchange",
            "priority_source",
            "bridge_key",
        ]
        df = pd.DataFrame(default_rows, columns=header[:5])
        df["country"] = ""
        df["exchange"] = ""
        df["priority_source"] = "yahoo"
        df["bridge_key"] = ""
        df.to_csv(UNIVERSE_PATH, index=False, sep=";")

    @staticmethod
    def _detect_encoding(file_path: str) -> str:
        for enc in ("utf-8", "cp1252", "latin1"):
            try:
                with open(file_path, "r", encoding=enc) as handle:
                    handle.read(2048)
                return enc
            except UnicodeDecodeError:
                continue
        return "latin1"

    @staticmethod
    def _detect_separator(sample: str) -> str:
        candidates = [";", ",", "\t"]
        scores = {c: sample.count(c) for c in candidates}
        return max(scores, key=scores.get) if scores else ";"

    @staticmethod
    def _normalize_categories(series: pd.Series) -> pd.Series:
        mapping = {
            "AÃ‡Ã•ES BRASIL": "Ações Brasil",
            "Acoes Brasil": "Ações Brasil",
            "ACOES BRASIL": "Ações Brasil",
            "Ações Brasileiras": "Ações Brasil",
            "Acoes Globais": "Ações Globais",
            "AÃ‡Ã•ES GLOBAIS": "Ações Globais",
            "Mercado de Cambio": "Mercado de Câmbio",
            "Cambio": "Mercado de Câmbio",
            "Criptomoedas": "Cripto",
            "Indices": "Índices",
        }
        cleaned = series.fillna("").astype(str).str.strip()
        cleaned = cleaned.replace(mapping)
        return cleaned

    @staticmethod
    def _split_single_column(df: pd.DataFrame) -> pd.DataFrame:
        if df.shape[1] > 1:
            return df
        col = df.columns[0]
        expanded = df[col].astype(str).str.split(";", expand=True)
        return expanded

    @staticmethod
    def _sanitize_universe(df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [str(c).strip().lower() for c in df.columns]
        if "ticker" in df.columns and "ticker_yahoo" not in df.columns:
            df = df.rename(columns={"ticker": "ticker_yahoo"})

        required = ["name", "category", "ticker_yahoo", "currency_code", "base"]
        for col in required:
            if col not in df.columns:
                df[col] = ""

        df["ticker_yahoo"] = df["ticker_yahoo"].astype(str).str.strip()
        df = df[~df["ticker_yahoo"].isin(["0", "000000", "", "nan", "11299"])]

        df["category"] = UniverseEngine._normalize_categories(df["category"])
        df["category"] = df["category"].replace({"": "Ações Brasil"})

        if "priority_source" not in df.columns:
            df["priority_source"] = "yahoo"
        if "bridge_key" not in df.columns:
            df["bridge_key"] = ""
        if "country" not in df.columns:
            df["country"] = ""
        if "exchange" not in df.columns:
            df["exchange"] = ""

        df["name"] = df["name"].replace({"": "Ativo"})

        df = df.dropna(subset=["ticker_yahoo"]).copy()
        df = df[df["ticker_yahoo"].apply(TickerResolver.is_valid_ticker_syntax)]

        df["category"] = df["category"].apply(lambda x: x if x in OFFICIAL_CATEGORIES else "Ações Brasil")
        df["display"] = df["name"].astype(str) + " — " + df["ticker_yahoo"].astype(str)
        return df.reset_index(drop=True)

    @staticmethod
    @st.cache_data(show_spinner=False)
    def load_universe() -> pd.DataFrame:
        UniverseEngine.ensure_default_universe()
        encoding = UniverseEngine._detect_encoding(UNIVERSE_PATH)
        with open(UNIVERSE_PATH, "r", encoding=encoding) as handle:
            sample = handle.read(2048)
        separator = UniverseEngine._detect_separator(sample)

        df = pd.read_csv(UNIVERSE_PATH, encoding=encoding, sep=separator)
        df = UniverseEngine._split_single_column(df)
        df = UniverseEngine._sanitize_universe(df)
        return df


class TickerResolver:
    @staticmethod
    def is_valid_ticker_syntax(ticker: str) -> bool:
        if ticker is None:
            return False
        value = str(ticker).strip()
        if not value:
            return False
        if value in {"0", "000000", "nan"}:
            return False
        if re.fullmatch(r"\d+", value):
            return False
        if ";" in value:
            return False
        if " " in value:
            return False
        if len(value) > 20:
            return False
        return True

    @staticmethod
    def resolve(ticker: str, category: str) -> str:
        raw = str(ticker).strip().upper()
        if category == "Ações Brasil":
            if raw.endswith(".SA"):
                return raw
            if raw.endswith("=X") or raw.startswith("^") or raw.endswith("-USD"):
                return raw
            if re.fullmatch(r"[A-Z]{4}\d{1,2}", raw):
                return f"{raw}.SA"
        return raw


class DataRouter:
    BCB_SERIES = {
        "USDBRL": "10813",
        "EURBRL": "21619",
        "SELIC": "432",
        "CDI": "12",
    }

    @staticmethod
    def _format_attempt(source: str, info: str, rows: int = 0, ok: bool = False) -> Dict:
        return {
            "source": source,
            "info": info,
            "rows": rows,
            "ok": ok,
            "ts": datetime.utcnow().isoformat(),
        }

    @staticmethod
    @st.cache_data(show_spinner=False, ttl=60 * 30)
    def fetch_yahoo(ticker: str, period: str, interval: str) -> pd.DataFrame:
        df = yf.download(
            tickers=ticker,
            period=period,
            interval=interval,
            auto_adjust=False,
            threads=False,
            progress=False,
            group_by="column",
        )
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            if ticker in df.columns.get_level_values(-1):
                df = df.xs(ticker, axis=1, level=-1)
            else:
                df = df.droplevel(-1, axis=1)
        df = df.rename(columns={c: c.lower() for c in df.columns})
        if "adj close" in df.columns:
            df["close_final"] = df["adj close"]
        elif "close" in df.columns:
            df["close_final"] = df["close"]
        return df.dropna(subset=["close_final"]).copy()

    @staticmethod
    def fetch_bcb_series(series_id: str) -> pd.DataFrame:
        url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{series_id}/dados?formato=json"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        payload = response.json()
        if not payload:
            return pd.DataFrame()
        df = pd.DataFrame(payload)
        df["data"] = pd.to_datetime(df["data"], dayfirst=True)
        df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
        df = df.dropna(subset=["valor"]).set_index("data").sort_index()
        df["open"] = df["valor"]
        df["high"] = df["valor"]
        df["low"] = df["valor"]
        df["close_final"] = df["valor"]
        df["volume"] = np.nan
        return df[["open", "high", "low", "close_final", "volume"]]

    @staticmethod
    def fetch_bridge_prices(ticker: str, bridge_key: str) -> pd.DataFrame:
        if not os.path.exists(BRIDGE_PATH):
            return pd.DataFrame()
        df = pd.read_csv(BRIDGE_PATH)
        df.columns = [str(c).strip().lower() for c in df.columns]
        key = bridge_key.strip().lower() if bridge_key else ""
        if key:
            subset = df[df.get("bridge_key", "").astype(str).str.lower() == key]
        else:
            subset = df[df.get("ticker", "").astype(str).str.lower() == ticker.lower()]
        if subset.empty:
            return pd.DataFrame()
        subset["date"] = pd.to_datetime(subset["date"])
        subset = subset.set_index("date").sort_index()
        subset = subset.rename(columns={"close": "close_final"})
        return subset[[c for c in ["open", "high", "low", "close_final", "volume"] if c in subset.columns]]

    @staticmethod
    def fetch_local_prices(ticker: str) -> pd.DataFrame:
        if not os.path.exists(LOCAL_PRICE_PATH):
            return pd.DataFrame()
        df = pd.read_csv(LOCAL_PRICE_PATH)
        df.columns = [str(c).strip().lower() for c in df.columns]
        subset = df[df.get("ticker", "").astype(str).str.upper() == ticker.upper()]
        if subset.empty:
            return pd.DataFrame()
        subset["date"] = pd.to_datetime(subset["date"])
        subset = subset.set_index("date").sort_index()
        subset = subset.rename(columns={"close": "close_final"})
        return subset[[c for c in ["open", "high", "low", "close_final", "volume"] if c in subset.columns]]

    @staticmethod
    def fetch_data(entry: pd.Series, period: str = "2y", interval: str = "1d") -> Tuple[pd.DataFrame, Dict]:
        attempts: List[Dict] = []
        ticker = entry["ticker_yahoo"]
        bridge_key = str(entry.get("bridge_key", ""))

        combos = [(period, interval), ("1y", "1d"), ("6mo", "1d")]
        for p, i in combos:
            df = DataRouter.fetch_yahoo(ticker, p, i)
            ok = len(df) >= 20
            attempts.append(DataRouter._format_attempt("yahoo", f"{p}/{i}", len(df), ok))
            if ok:
                return df, {"attempts": attempts, "source": "yahoo", "reason": "ok"}

        if entry["category"] == "Mercado de Câmbio" or ticker in {"SELIC", "CDI"}:
            series_key = ticker.replace("=X", "").replace("/", "").replace("-", "").upper()
            if series_key in {"BRL", "USDBRL"}:
                series_key = "USDBRL"
            if series_key in {"EURBRL", "EUR"}:
                series_key = "EURBRL"
            series_id = DataRouter.BCB_SERIES.get(series_key)
            if series_id:
                try:
                    df = DataRouter.fetch_bcb_series(series_id)
                    attempts.append(DataRouter._format_attempt("bcb", f"series {series_id}", len(df), len(df) > 5))
                    if len(df) > 5:
                        return df, {"attempts": attempts, "source": "bcb", "reason": "ok"}
                except requests.RequestException as exc:
                    attempts.append(DataRouter._format_attempt("bcb", f"error {exc}", 0, False))

        df = DataRouter.fetch_bridge_prices(ticker, bridge_key)
        attempts.append(DataRouter._format_attempt("bridge", "bridge_prices.csv", len(df), len(df) > 5))
        if len(df) > 5:
            return df, {"attempts": attempts, "source": "bridge", "reason": "ok"}

        df = DataRouter.fetch_local_prices(ticker)
        attempts.append(DataRouter._format_attempt("local", "local_prices.csv", len(df), len(df) > 5))
        if len(df) > 5:
            return df, {"attempts": attempts, "source": "local", "reason": "ok"}

        return pd.DataFrame(), {"attempts": attempts, "source": "none", "reason": "no_data"}


class IndicatorEngine:
    @staticmethod
    def calculate(df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        if "close_final" not in data.columns:
            return pd.DataFrame()
        close = data["close_final"]
        data["sma20"] = close.rolling(20).mean()
        data["sma50"] = close.rolling(50).mean()
        data["ema20"] = close.ewm(span=20, adjust=False).mean()
        data["std20"] = close.rolling(20).std(ddof=0)
        data["zscore"] = (close - data["sma20"]) / (data["std20"] + 1e-9)

        data["return"] = close.pct_change()
        data["log_return"] = np.log(close / close.shift(1))
        data["vol_ann"] = data["log_return"].rolling(20).std(ddof=0) * np.sqrt(252)

        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        roll_up = gain.rolling(14).mean()
        roll_down = loss.rolling(14).mean()
        rs = roll_up / (roll_down + 1e-9)
        data["rsi14"] = 100 - (100 / (1 + rs))

        high = data["high"] if "high" in data.columns else close
        low = data["low"] if "low" in data.columns else close
        prev_close = close.shift(1)
        tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        data["atr14"] = tr.rolling(14).mean()

        data["bb_mid"] = data["sma20"]
        data["bb_up"] = data["sma20"] + 2 * data["std20"]
        data["bb_dn"] = data["sma20"] - 2 * data["std20"]

        data["equity"] = (1 + data["return"].fillna(0)).cumprod()
        data["peak"] = data["equity"].cummax()
        data["drawdown"] = data["equity"] / data["peak"] - 1

        data["rolling_sharpe"] = data["return"].rolling(20).mean() / (data["return"].rolling(20).std(ddof=0) + 1e-9)
        data["skew"] = data["return"].rolling(60).skew()
        data["kurt"] = data["return"].rolling(60).kurt()

        var_window = data["return"].rolling(60)
        data["var_95"] = var_window.quantile(0.05)
        data["cvar_95"] = var_window.apply(lambda x: x[x <= x.quantile(0.05)].mean(), raw=False)

        if "volume" in data.columns:
            vol_mean = data["volume"].rolling(20).mean()
            vol_std = data["volume"].rolling(20).std(ddof=0)
            data["z_volume"] = (data["volume"] - vol_mean) / (vol_std + 1e-9)

        return data.dropna().copy()


class RegimeEngine:
    @staticmethod
    def analyze(df: pd.DataFrame) -> Dict:
        last = df.iloc[-1]
        trend_up = last["sma20"] > last["sma50"] and last["close_final"] > last["sma20"]
        trend_down = last["sma20"] < last["sma50"] and last["close_final"] < last["sma20"]
        z = last["zscore"]
        vol = last["vol_ann"]
        dd = last["drawdown"]

        regime = "Neutro/Transição"
        if dd < -0.15:
            regime = "Stress"
        elif vol > 0.45:
            regime = "Volatilidade Hostil"
        elif trend_up:
            regime = "Tendência de Alta"
        elif trend_down:
            regime = "Tendência de Baixa"

        if z > 2.0:
            regime += " (Esticado)"
        elif z < -2.0:
            regime += " (Descontado)"

        return {
            "label": regime,
            "trend_up": trend_up,
            "trend_down": trend_down,
            "zscore": z,
            "vol": vol,
            "drawdown": dd,
        }

    @staticmethod
    def operability_score(regime: Dict, cost_bps: float) -> Tuple[int, List[str]]:
        score = 100
        reasons = []
        if "Stress" in regime["label"]:
            score -= 40
            reasons.append("Stress detectado")
        if "Volatilidade Hostil" in regime["label"]:
            score -= 25
            reasons.append("Volatilidade hostil")
        if abs(regime["zscore"]) > 2.2:
            score -= 15
            reasons.append("Z-score extremo")
        if cost_bps >= 20:
            score -= 30
            reasons.append("Custo elevado")
        score = max(0, min(100, score))
        return score, reasons


class CostEngine:
    @staticmethod
    def compute(spread_bps: float, slippage_bps: float, commission_bps: float, price: float) -> Dict:
        total_bps = spread_bps + slippage_bps + commission_bps
        break_even = (total_bps / 10000) * price
        block = total_bps >= 20
        return {
            "total_bps": total_bps,
            "break_even_move": break_even,
            "block": block,
        }


class RiskEngine:
    @staticmethod
    def monte_carlo_ruin(win_rate: float, payoff: float, risk_per_trade: float, horizon: int, paths: int, ruin_limit: float) -> float:
        rng = np.random.default_rng(42)
        outcomes = rng.random((paths, horizon))
        wins = outcomes < win_rate
        returns = np.where(wins, risk_per_trade * payoff, -risk_per_trade)
        equity = returns.cumsum(axis=1)
        ruin = np.any(equity <= -ruin_limit, axis=1)
        return float(ruin.mean() * 100)

    @staticmethod
    def sizing_from_atr(capital: float, risk_per_trade: float, atr: float, price: float) -> Dict:
        if atr <= 0:
            return {"qty": 0.0, "stop": price}
        stop = price - 2 * atr
        risk_amount = capital * risk_per_trade
        qty = risk_amount / (price - stop)
        return {"qty": max(qty, 0.0), "stop": stop}

    @staticmethod
    def stress_test(df: pd.DataFrame) -> float:
        if df.empty:
            return 0.0
        return float(df["drawdown"].min() * 100)


class VerdictEngine:
    @staticmethod
    def evaluate(regime: Dict, cost: Dict, ruin_prob: float) -> Dict:
        status = "AUTORIZADO"
        command = "EXECUÇÃO AUTORIZADA"
        reasons = []
        if cost["block"]:
            status = "BLOQUEIO"
            command = "NÃO OPERAR"
            reasons.append("Custo alto")
        if "Stress" in regime["label"] or "Volatilidade Hostil" in regime["label"]:
            status = "DEFENSIVO" if status == "AUTORIZADO" else status
            command = "AGUARDE" if status == "DEFENSIVO" else command
            reasons.append("Regime adverso")
        if ruin_prob > 5:
            status = "DEFENSIVO" if status != "BLOQUEIO" else status
            reasons.append("Risco de ruína elevado")
        return {
            "status": status,
            "command": command,
            "reasons": reasons or ["Sem bloqueios críticos"],
        }


class ExecutionSimEngine:
    @staticmethod
    def init_ledger() -> None:
        if not os.path.exists(LEDGER_PATH):
            pd.DataFrame(columns=["ts", "ticker", "side", "qty", "price", "note"]).to_csv(LEDGER_PATH, index=False)

    @staticmethod
    def append(ticker: str, side: str, qty: float, price: float, note: str) -> None:
        ExecutionSimEngine.init_ledger()
        df = pd.read_csv(LEDGER_PATH)
        row = pd.DataFrame(
            [
                {
                    "ts": datetime.utcnow().isoformat(),
                    "ticker": ticker,
                    "side": side,
                    "qty": qty,
                    "price": price,
                    "note": note,
                }
            ]
        )
        df = pd.concat([df, row], ignore_index=True)
        df.to_csv(LEDGER_PATH, index=False)

    @staticmethod
    def read() -> pd.DataFrame:
        ExecutionSimEngine.init_ledger()
        return pd.read_csv(LEDGER_PATH)


class NarrativeAIEngine:
    @staticmethod
    def llm_generate(prompt: str) -> str:
        return prompt

    @staticmethod
    def narrative(ticker: str, regime: Dict, last: pd.Series, cost: Dict) -> str:
        text = [f"**{ticker}** negocia a **{last['close_final']:.2f}**."]
        text.append(f"Regime dominante: **{regime['label']}**.")
        if abs(regime["zscore"]) > 2:
            text.append("O preço está estatisticamente esticado (alto risco de reversão).")
        else:
            text.append("O preço está dentro da faixa normal (movimento orgânico).")
        if regime["vol"] > 0.4:
            text.append("Volatilidade elevada exige stops largos e menor size.")
        if cost["total_bps"] > 15:
            text.append("Custo de execução está significativo: edge precisa ser maior para compensar.")
        return "\n\n".join(text)

    @staticmethod
    def plan(regime: Dict, last: pd.Series) -> Dict:
        atr = last["atr14"]
        price = last["close_final"]
        setup = "tendência" if "Tendência" in regime["label"] else "reversão"
        if "Tendência de Alta" in regime["label"]:
            direction = "COMPRA"
            stop = price - 2 * atr
            target = price + 3 * atr
        elif "Tendência de Baixa" in regime["label"]:
            direction = "VENDA"
            stop = price + 2 * atr
            target = price - 3 * atr
        else:
            direction = "AGUARDE"
            stop = price - 2 * atr
            target = price + 2 * atr
        return {
            "setup": setup,
            "direction": direction,
            "entry": price,
            "stop": stop,
            "target": target,
            "risk_per_trade": 0.01,
            "invalidate": "Quebra de regime ou custo alto.",
        }

    @staticmethod
    def checklist(regime: Dict, cost: Dict, ruin_prob: float, volume_ok: bool) -> Dict[str, bool]:
        return {
            "regime_ok": "Stress" not in regime["label"],
            "custo_ok": cost["total_bps"] < 20,
            "vol_ok": regime["vol"] < 0.5,
            "ruina_ok": ruin_prob < 5,
            "liquidez_ok": volume_ok,
            "macro_ok": True,
        }

    @staticmethod
    def benchmark_compare(df: pd.DataFrame, benchmark: Optional[pd.DataFrame]) -> Dict:
        if benchmark is None or benchmark.empty:
            return {}
        aligned = pd.concat([df["return"], benchmark["return"]], axis=1).dropna()
        if aligned.empty:
            return {}
        aligned.columns = ["asset", "bench"]
        beta = aligned.cov().iloc[0, 1] / (aligned["bench"].var() + 1e-9)
        corr = aligned["asset"].corr(aligned["bench"])
        perf = (1 + aligned["asset"]).prod() - (1 + aligned["bench"]).prod()
        return {"beta": beta, "corr": corr, "relative_perf": perf}


class VisualizationEngine:
    @staticmethod
    def plot_price(df: pd.DataFrame) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df.get("open"),
                high=df.get("high"),
                low=df.get("low"),
                close=df.get("close_final"),
                name="Preço",
            )
        )
        fig.add_trace(go.Scatter(x=df.index, y=df["sma20"], name="SMA20"))
        fig.add_trace(go.Scatter(x=df.index, y=df["sma50"], name="SMA50"))
        fig.add_trace(go.Scatter(x=df.index, y=df["bb_up"], name="BB Up", line=dict(width=1)))
        fig.add_trace(go.Scatter(x=df.index, y=df["bb_dn"], name="BB Down", line=dict(width=1)))
        fig.update_layout(template="plotly_dark", height=520, margin=dict(l=10, r=10, t=40, b=10))
        fig.update_xaxes(rangeslider_visible=False)
        return fig

    @staticmethod
    def plot_metric(df: pd.DataFrame, column: str, title: str) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df[column], name=column))
        fig.update_layout(template="plotly_dark", height=240, margin=dict(l=10, r=10, t=40, b=10), title=title)
        return fig


class ScreenerEngine:
    @staticmethod
    def run(universe: pd.DataFrame, limit: int, period: str, interval: str) -> pd.DataFrame:
        results = []
        sample = universe.head(limit)
        for _, row in sample.iterrows():
            entry = row.copy()
            entry["ticker_yahoo"] = TickerResolver.resolve(entry["ticker_yahoo"], entry["category"])
            df_raw, _ = DataRouter.fetch_data(entry, period, interval)
            if df_raw.empty:
                continue
            df = IndicatorEngine.calculate(df_raw)
            if df.empty:
                continue
            last = df.iloc[-1]
            results.append(
                {
                    "Ticker": entry["ticker_yahoo"],
                    "Categoria": entry["category"],
                    "Close": last["close_final"],
                    "ZScore": last["zscore"],
                    "RSI": last["rsi14"],
                    "Vol": last["vol_ann"],
                    "Drawdown": last["drawdown"],
                }
            )
        return pd.DataFrame(results)


class DiagnosticsEngine:
    @staticmethod
    def suspicious_tickers(universe: pd.DataFrame) -> List[str]:
        return [t for t in universe["ticker_yahoo"] if not TickerResolver.is_valid_ticker_syntax(t)]


def render_header(entry: pd.Series) -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="ticker-tape">
            <div class="ticker-item">USD/BRL 5.XX</div>
            <div class="ticker-item">IBOV 12X.XXX</div>
            <div class="ticker-item">DI1F27 10.XX%</div>
            <div class="ticker-item">S&P 500 4XXX</div>
            <div class="ticker-item">BTC 6X.XXX</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(f"<div class=\"atlas-title\">{APP_TITLE}</div>", unsafe_allow_html=True)
    st.markdown("<div class=\"atlas-subtitle\">Premium Financial Intelligence Terminal (BR-First)</div>", unsafe_allow_html=True)
    st.markdown(
        f"<span class=\"atlas-badge\">Ativo</span> {entry['name']} &nbsp;"
        f"<span class=\"atlas-badge\">Ticker</span> {entry['ticker_yahoo']} &nbsp;"
        f"<span class=\"atlas-badge\">Categoria</span> {entry['category']}",
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, foot: str) -> None:
    st.markdown(
        f"""
        <div class="atlas-card">
          <div class="atlas-kpi-title">{label}</div>
          <div class="atlas-kpi-value">{value}</div>
          <div class="atlas-kpi-foot">{foot}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")

    universe = UniverseEngine.load_universe()
    if universe.empty:
        st.error("Universo vazio. Verifique data/universe.csv")
        return

    st.sidebar.header("📡 Universo")
    category = st.sidebar.selectbox("Classe de Ativo", universe["category"].unique())
    subset = universe[universe["category"] == category]
    asset_display = st.sidebar.selectbox("Ativo", subset["display"].tolist())
    entry = subset[subset["display"] == asset_display].iloc[0].copy()
    entry["ticker_yahoo"] = TickerResolver.resolve(entry["ticker_yahoo"], entry["category"])

    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Parâmetros")
    period = st.sidebar.selectbox("Período", ["2y", "1y", "6mo"], index=0)
    interval = st.sidebar.selectbox("Intervalo", ["1d"], index=0)

    st.sidebar.markdown("---")
    spread = st.sidebar.number_input("Spread (bps)", value=6, min_value=0)
    slippage = st.sidebar.number_input("Slippage (bps)", value=4, min_value=0)
    commission = st.sidebar.number_input("Comissão (bps)", value=2, min_value=0)

    st.sidebar.markdown("---")
    analyze = st.sidebar.button("Analisar Ativo", type="primary")

    render_header(entry)

    if "analysis" not in st.session_state:
        st.session_state["analysis"] = {}

    if analyze:
        with st.spinner("DataRouter em execução..."):
            df_raw, debug = DataRouter.fetch_data(entry, period, interval)
        st.session_state["analysis"] = {"df_raw": df_raw, "debug": debug}

    data = st.session_state.get("analysis", {})
    df_raw = data.get("df_raw", pd.DataFrame())
    debug = data.get("debug", {})

    tabs = st.tabs(
        [
            "📊 Análise",
            "🔎 Screener",
            "🌍 Macro Map",
            "📚 Fundamentalista",
            "🤖 IA Interna",
            "🧾 Execução",
            "🧪 Diagnóstico",
        ]
    )

    with tabs[0]:
        if df_raw.empty:
            st.markdown("<div class=\"atlas-alert\">Sem dados para este ativo. Verifique o ticker ou tente outro período.</div>", unsafe_allow_html=True)
            if debug:
                with st.expander("Debug"):
                    st.json(debug)
            return

        df = IndicatorEngine.calculate(df_raw)
        last = df.iloc[-1]
        regime = RegimeEngine.analyze(df)
        cost = CostEngine.compute(spread, slippage, commission, last["close_final"])
        oper_score, oper_reasons = RegimeEngine.operability_score(regime, cost["total_bps"])

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kpi_card("Preço", f"{last['close_final']:.2f}", "Último fechamento")
        with c2:
            kpi_card("Regime", regime["label"], "Classificação do regime")
        with c3:
            kpi_card("Vol Anual", f"{last['vol_ann']*100:.1f}%", "Ruído do ativo")
        with c4:
            kpi_card("Operability", f"{oper_score}/100", ", ".join(oper_reasons) or "Operável")

        st.plotly_chart(VisualizationEngine.plot_price(df.tail(220)), use_container_width=True)

        c5, c6, c7 = st.columns(3)
        with c5:
            st.plotly_chart(VisualizationEngine.plot_metric(df, "rsi14", "RSI (14)"), use_container_width=True)
        with c6:
            st.plotly_chart(VisualizationEngine.plot_metric(df, "drawdown", "Drawdown"), use_container_width=True)
        with c7:
            st.plotly_chart(VisualizationEngine.plot_metric(df, "vol_ann", "Volatilidade"), use_container_width=True)

        st.markdown("### Custos e Edge")
        st.write(
            {
                "total_bps": cost["total_bps"],
                "break_even_move": round(cost["break_even_move"], 4),
                "bloqueio": cost["block"],
            }
        )

    with tabs[1]:
        st.subheader("Screener (amostra)")
        limit = st.slider("Qtd ativos", 20, 50, 20)
        run_batch = st.checkbox("Rodar lote (cuidado)", value=False)
        if st.button("Executar Screener"):
            if run_batch:
                with st.spinner("Rodando screener..."):
                    result = ScreenerEngine.run(subset, limit, "6mo", "1d")
                if result.empty:
                    st.warning("Sem resultados.")
                else:
                    st.dataframe(result.style.background_gradient(cmap="RdYlGn", subset=["ZScore"]), use_container_width=True)
            else:
                st.info("Ative 'rodar lote' para executar a amostra.")

    with tabs[2]:
        st.subheader("Macro Map")
        st.markdown(
            "- **FX:** juros, inflação, risco país, DXY.\n"
            "- **Ações BR:** Selic, commodities, fiscal.\n"
            "- **Cripto:** liquidez global, yields, risk-on/off."
        )
        st.markdown("### Links rápidos")
        for name, url in [
            ("Reuters", "https://www.reuters.com"),
            ("Bloomberg", "https://www.bloomberg.com"),
            ("Valor", "https://valor.globo.com"),
            ("InfoMoney", "https://www.infomoney.com.br"),
        ]:
            st.markdown(f"- [{name}]({url})")

    with tabs[3]:
        st.subheader("Fundamentalista (Yahoo)")
        if st.button("Carregar dados fundamentais"):
            with st.spinner("Buscando..."):
                info = yf.Ticker(entry["ticker_yahoo"]).get_info()
            st.json(info)

    with tabs[4]:
        st.subheader("IA Interna")
        if df_raw.empty:
            st.info("Sem dados para IA interna.")
        else:
            df = IndicatorEngine.calculate(df_raw)
            last = df.iloc[-1]
            regime = RegimeEngine.analyze(df)
            cost = CostEngine.compute(spread, slippage, commission, last["close_final"])
            narrative = NarrativeAIEngine.narrative(entry["ticker_yahoo"], regime, last, cost)
            st.info(narrative)

            plan = NarrativeAIEngine.plan(regime, last)
            st.markdown("### Plano de Trade")
            st.write(plan)

            st.markdown("### Checklist Operacional")
            ruin_prob = RiskEngine.monte_carlo_ruin(0.45, 2.0, 0.02, 200, 2000, 0.3)
            volume_ok = "volume" in df.columns and df["volume"].iloc[-1] > 0
            checklist = NarrativeAIEngine.checklist(regime, cost, ruin_prob, volume_ok=volume_ok)
            st.json(checklist)

            benchmark_ticker = None
            if entry["category"] == "Ações Brasil":
                benchmark_ticker = "^BVSP"
            elif entry["category"] == "Ações Globais":
                benchmark_ticker = "^GSPC"
            elif entry["category"] == "Cripto":
                benchmark_ticker = "BTC-USD"
            if benchmark_ticker:
                bench_entry = entry.copy()
                bench_entry["ticker_yahoo"] = benchmark_ticker
                bench_raw, _ = DataRouter.fetch_data(bench_entry, "6mo", "1d")
                if not bench_raw.empty:
                    bench_ind = IndicatorEngine.calculate(bench_raw)
                    comparison = NarrativeAIEngine.benchmark_compare(df, bench_ind)
                    st.markdown("### Benchmark")
                    st.write(comparison)

    with tabs[5]:
        st.subheader("Execução simulada")
        if df_raw.empty:
            st.info("Carregue dados para execução.")
        else:
            last_price = float(IndicatorEngine.calculate(df_raw).iloc[-1]["close_final"])
            side = st.selectbox("Side", ["BUY", "SELL"])
            qty = st.number_input("Quantidade", min_value=1.0, value=1.0)
            note = st.text_input("Nota", value="Simulação")
            if st.button("Registrar no ledger"):
                ExecutionSimEngine.append(entry["ticker_yahoo"], side, qty, last_price, note)
                st.success("Registrado.")

            st.markdown("### Ledger")
            st.dataframe(ExecutionSimEngine.read().tail(200), use_container_width=True)

    with tabs[6]:
        st.subheader("Diagnóstico")
        st.json(
            {
                "universe_path": UNIVERSE_PATH,
                "total_assets": len(universe),
                "suspicious_tickers": DiagnosticsEngine.suspicious_tickers(universe),
                "last_debug": debug,
            }
        )


if __name__ == "__main__":
    main()
