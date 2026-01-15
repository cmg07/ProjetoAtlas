import os
import re
import math
import time
import json
import random
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import requests
import yfinance as yf
from bs4 import BeautifulSoup

warnings.filterwarnings("ignore")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_FILE = os.path.join(APP_DIR, "listaativos_500_extremo.txt")

# ============================================================
# CONFIG: distribuição equivalente (500 / 8)
# ============================================================
CATEGORIES = [
    ("CÂMBIO & FX (YAHOO)", 63),
    ("CRIPTOMOEDAS (TOP MARKET CAP)", 63),
    ("AÇÕES BRASILEIRAS – B3 (IBOV OFICIAL)", 63),
    ("AÇÕES EUA – US (NASDAQ/NYSE OFICIAL)", 63),
    ("ETFs GLOBAIS (CURADOS + VALIDADOS)", 62),
    ("COMMODITIES & FUTUROS (YAHOO)", 62),
    ("ÍNDICES & TAXAS (YAHOO)", 62),
    ("AÇÕES GLOBAIS EM DESTAQUE (ADR/MEGA)", 62),
]

# ============================================================
# UTIL
# ============================================================
def http_get(url: str, timeout: int = 20) -> str:
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return r.text

def safe_norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def yf_is_valid_ticker(ticker: str) -> bool:
    """
    Validação rápida: tenta baixar 5 dias de 1d.
    Se vier OHLC com Close, é aceito.
    """
    try:
        df = yf.download(ticker, period="5d", interval="1d", progress=False, auto_adjust=False, threads=True)
        if df is None or df.empty:
            return False
        if isinstance(df.columns, pd.MultiIndex):
            # tenta reduzir
            df.columns = df.columns.get_level_values(0)
        return ("Close" in df.columns) and (df["Close"].dropna().shape[0] > 0)
    except Exception:
        return False

def pick_valid(candidates: List[Tuple[str, str]], target: int, hard_limit: int = 4000) -> Dict[str, str]:
    """
    Seleciona 'target' tickers validados.
    candidates: list of (ticker, label)
    """
    out: Dict[str, str] = {}
    tries = 0
    for t, label in candidates:
        if len(out) >= target:
            break
        if t in out:
            continue
        tries += 1
        if tries > hard_limit:
            break
        if yf_is_valid_ticker(t):
            out[t] = label

    return out

# ============================================================
# 1) B3 – IBOV (oficial): parse HTML e transforma em .SA
# Fonte: página oficial de composição (B3)
# ============================================================
def fetch_b3_ibov_constituents() -> List[Tuple[str, str]]:
    """
    Retorna lista (ticker_yahoo, label)
    Ex.: ("PETR4.SA", "PETROBRAS PN (IBOV)")
    """
    # Versão BR+ costuma estar atualizada; fallback para outra página se necessário
    urls = [
        "https://www.b3.com.br/pt_br/market-data-e-indices/indices/indices-amplos/indice-bovespa-b3-br-ibovespa-b3-br-composicao-carteira.htm",
        "https://www.b3.com.br/pt_br/market-data-e-indices/indices/indices-amplos/indice-ibovespa-ibovespa-composicao-da-carteira.htm",
        "https://www.b3.com.br/en_us/market-data-and-indices/indexes/broad-indexes/bovespa-b3-br-index-ibovespa-b3-br-composition-index-portfolio.htm",
        "https://www.b3.com.br/en_us/market-data-and-indices/indices/broad-indices/indice-ibovespa-ibovespa-composition-index-portfolio.htm",
    ]

    html = None
    for u in urls:
        try:
            html = http_get(u)
            if html and "Composição" in html or "Composition" in html:
                break
        except Exception:
            continue

    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    # A B3 publica uma tabela com colunas (Código/Stock etc). Vamos extrair códigos em células.
    text = soup.get_text("\n", strip=True)
    # códigos B3 típicos: 4 letras + número (ex: PETR4, VALE3, ITUB4, BBDC4, etc.)
    codes = re.findall(r"\b[A-Z]{4}\d{1,2}\b", text)

    # remove duplicatas preservando ordem
    seen = set()
    unique = []
    for c in codes:
        if c not in seen:
            unique.append(c)
            seen.add(c)

    out = []
    for code in unique:
        out.append((f"{code}.SA", f"{code} (IBOV)"))
    return out

# ============================================================
# 2) NASDAQ Trader (oficial): nasdaqlisted + otherlisted
# Fonte: arquivos oficiais em texto (pipe-delimited)
# ============================================================
def fetch_nasdaq_trader_listed(url: str, is_otherlisted: bool) -> List[Tuple[str, str]]:
    raw = http_get(url)
    lines = raw.splitlines()
    # header
    header = lines[0].split("|")
    data = lines[1:]
    out = []
    for ln in data:
        if not ln.strip():
            continue
        parts = ln.split("|")
        if len(parts) < 2:
            continue
        symbol = parts[0].strip()
        name = parts[1].strip()

        # linhas de trailer (File Creation Time / etc) costumam vir com "File Creation Time"
        if symbol.upper().startswith("FILE"):
            continue

        # filtra símbolos estranhos e testes
        if symbol in ("Symbol", "ACT Symbol"):
            continue
        if "^" in symbol or "/" in symbol:
            continue

        # outroslisted: tem coluna ETF indicando Y/N
        # nasdaqlisted: também tem coluna ETF (posição varia mas existe)
        out.append((symbol, name))
    return out

def fetch_us_equities_candidates() -> List[Tuple[str, str]]:
    nasdaq_url = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
    other_url  = "https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt"

    nasdaq = fetch_nasdaq_trader_listed(nasdaq_url, is_otherlisted=False)
    other  = fetch_nasdaq_trader_listed(other_url, is_otherlisted=True)

    # Monta candidatos: vamos priorizar "Common Stock" / "Common Shares" no nome,
    # e evitar "Warrant", "Unit", "Rights", "Preferred", etc.
    bad_kw = ["Warrant", "Unit", "Rights", "Preferred", "Notes", "Bond", "Depositary", "Trust", "ETF", "ETN"]
    good_kw = ["Common Stock", "Common Shares", "Ordinary Shares", "Class A", "Class B"]

    def score(name: str) -> int:
        s = 0
        for g in good_kw:
            if g.lower() in name.lower():
                s += 2
        for b in bad_kw:
            if b.lower() in name.lower():
                s -= 5
        return s

    combined = nasdaq + other
    combined = [(sym, nm) for sym, nm in combined if re.fullmatch(r"[A-Z]{1,5}", sym or "")]

    combined.sort(key=lambda x: score(x[1]), reverse=True)
    # mantém um pool grande para validação final
    return [(sym, f"{nm} (US)") for sym, nm in combined[:1500]]

# ============================================================
# 3) CoinGecko (documentação oficial): top coins by market cap
# Observação: API pública pode ter rate limit; faremos backoff simples.
# ============================================================
def fetch_coingecko_top_symbols(vs_currency="usd", per_page=250, pages=2) -> List[Tuple[str, str]]:
    """
    Retorna lista (ticker_yahoo, label) convertendo symbol -> SYMBOL-USD (Yahoo)
    """
    out = []
    base = "https://api.coingecko.com/api/v3/coins/markets"
    for page in range(1, pages + 1):
        params = {
            "vs_currency": vs_currency,
            "order": "market_cap_desc",
            "per_page": per_page,
            "page": page,
            "sparkline": "false",
        }
        ok = False
        for attempt in range(1, 6):
            try:
                r = requests.get(base, params=params, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
                if r.status_code == 429:
                    time.sleep(1.5 * attempt)
                    continue
                r.raise_for_status()
                data = r.json()
                for it in data:
                    sym = (it.get("symbol") or "").upper().strip()
                    name = safe_norm(it.get("name") or sym)
                    if not sym or not re.fullmatch(r"[A-Z0-9]{2,10}", sym):
                        continue
                    # Yahoo cripto: SYMBOL-USD
                    out.append((f"{sym}-USD", f"{name} ({sym})"))
                ok = True
                break
            except Exception:
                time.sleep(1.2 * attempt)
        if not ok:
            break
    return out

# ============================================================
# 4) Listas curadas (para completar com qualidade e variedade)
# - ETFs globais (top conhecidos)
# - Commodities/Futuros (tickers padrão Yahoo)
# - Índices & taxas (tickers padrão Yahoo)
# - FX (pares relevantes, BRL + majors)
# - Ações globais em destaque (ADRs / mega caps)
# ============================================================
def curated_fx_candidates() -> List[Tuple[str, str]]:
    # BRL + majors + crosses altamente líquidos
    pairs = [
        ("USDBRL=X", "USD/BRL"),
        ("EURBRL=X", "EUR/BRL"),
        ("GBPBRL=X", "GBP/BRL"),
        ("JPYBRL=X", "JPY/BRL"),
        ("AUDBRL=X", "AUD/BRL"),
        ("CADBRL=X", "CAD/BRL"),
        ("CHFBRL=X", "CHF/BRL"),
        ("CNYBRL=X", "CNY/BRL"),
        ("ARSBRL=X", "ARS/BRL"),
        ("MXNBRL=X", "MXN/BRL"),
        ("CLPBRL=X", "CLP/BRL"),
        ("PENBRL=X", "PEN/BRL"),
        ("COPBRL=X", "COP/BRL"),
        ("UYUBRL=X", "UYU/BRL"),
        ("TRYBRL=X", "TRY/BRL"),
        ("ZARBRL=X", "ZAR/BRL"),
        ("INRBRL=X", "INR/BRL"),
        ("KRWBRL=X", "KRW/BRL"),
        ("HKDBRL=X", "HKD/BRL"),
        ("SGDBRL=X", "SGD/BRL"),
        # majors
        ("EURUSD=X", "EUR/USD"),
        ("GBPUSD=X", "GBP/USD"),
        ("USDJPY=X", "USD/JPY"),
        ("AUDUSD=X", "AUD/USD"),
        ("USDCAD=X", "USD/CAD"),
        ("USDCHF=X", "USD/CHF"),
        ("NZDUSD=X", "NZD/USD"),
        ("EURJPY=X", "EUR/JPY"),
        ("EURGBP=X", "EUR/GBP"),
        ("EURAUD=X", "EUR/AUD"),
        ("GBPJPY=X", "GBP/JPY"),
        ("AUDJPY=X", "AUD/JPY"),
        ("CADJPY=X", "CAD/JPY"),
        ("CHFJPY=X", "CHF/JPY"),
        # emergentes líquidos
        ("USDMXN=X", "USD/MXN"),
        ("USDCLP=X", "USD/CLP"),
        ("USDCNH=X", "USD/CNH"),
        ("USDZAR=X", "USD/ZAR"),
        ("USDTRY=X", "USD/TRY"),
        ("USDINR=X", "USD/INR"),
        ("USDKRW=X", "USD/KRW"),
        ("USDHKD=X", "USD/HKD"),
        ("USDSGD=X", "USD/SGD"),
    ]
    # completar até ~80; pick_valid vai selecionar até target
    # adições para robustez
    extras = [
        ("EURCHF=X", "EUR/CHF"), ("EURNZD=X", "EUR/NZD"), ("EURCAD=X", "EUR/CAD"),
        ("GBPAUD=X", "GBP/AUD"), ("GBPCAD=X", "GBP/CAD"), ("GBPCHF=X", "GBP/CHF"),
        ("AUDCAD=X", "AUD/CAD"), ("AUDCHF=X", "AUD/CHF"), ("AUDNZD=X", "AUD/NZD"),
        ("CADCHF=X", "CAD/CHF"), ("NZDJPY=X", "NZD/JPY"), ("NZDCAD=X", "NZD/CAD"),
        ("NZDCHF=X", "NZD/CHF"), ("SGDJPY=X", "SGD/JPY"), ("HKDJPY=X", "HKD/JPY"),
        ("CNYJPY=X", "CNY/JPY"), ("INRJPY=X", "INR/JPY"), ("ZARJPY=X", "ZAR/JPY"),
    ]
    pairs.extend(extras)
    return [(t, f"{n} (FX)") for t, n in pairs]

def curated_commodities_candidates() -> List[Tuple[str, str]]:
    # Futuros/benchmarks padrão Yahoo (alta cobertura)
    fut = [
        ("CL=F", "Crude Oil WTI"),
        ("BZ=F", "Brent Crude"),
        ("NG=F", "Natural Gas"),
        ("GC=F", "Gold"),
        ("SI=F", "Silver"),
        ("HG=F", "Copper"),
        ("PL=F", "Platinum"),
        ("PA=F", "Palladium"),
        ("ZC=F", "Corn"),
        ("ZW=F", "Wheat"),
        ("ZS=F", "Soybeans"),
        ("KC=F", "Coffee"),
        ("SB=F", "Sugar"),
        ("CT=F", "Cotton"),
        ("OJ=F", "Orange Juice"),
        ("LE=F", "Live Cattle"),
        ("HE=F", "Lean Hogs"),
        ("GF=F", "Feeder Cattle"),
        ("LBS=F", "Lumber"),
    ]
    # ETFs de commodities para ampliar até 62 sem “inventar”
    etf = [
        ("GLD", "SPDR Gold Shares"),
        ("SLV", "iShares Silver Trust"),
        ("USO", "United States Oil Fund"),
        ("UNG", "United States Natural Gas Fund"),
        ("DBC", "Invesco DB Commodity Index"),
        ("PDBC", "Invesco Optimum Yield Diversified"),
        ("DBA", "Invesco DB Agriculture"),
        ("CORN", "Teucrium Corn Fund"),
        ("WEAT", "Teucrium Wheat Fund"),
        ("SOYB", "Teucrium Soybean Fund"),
        ("COPX", "Global X Copper Miners"),
        ("XME", "SPDR S&P Metals & Mining"),
        ("XOP", "SPDR S&P Oil & Gas Exploration"),
        ("OIH", "VanEck Oil Services"),
    ]
    cand = [(t, f"{n} (COMMOD)") for t, n in fut + etf]
    return cand

def curated_indices_rates_candidates() -> List[Tuple[str, str]]:
    idx = [
        ("^BVSP", "IBOVESPA"),
        ("^GSPC", "S&P 500"),
        ("^IXIC", "NASDAQ Composite"),
        ("^DJI", "Dow Jones"),
        ("^RUT", "Russell 2000"),
        ("^VIX", "VIX"),
        ("^FTSE", "FTSE 100"),
        ("^GDAXI", "DAX"),
        ("^FCHI", "CAC 40"),
        ("^STOXX50E", "Euro Stoxx 50"),
        ("^N225", "Nikkei 225"),
        ("^HSI", "Hang Seng"),
        ("000001.SS", "Shanghai Composite"),
        ("^AXJO", "ASX 200"),
        ("^GSPTSE", "TSX Composite"),
        ("^MXX", "IPC Mexico"),
        # taxas/yields (Yahoo)
        ("^TNX", "US 10Y Yield"),
        ("^IRX", "US 13W Yield"),
        ("^FVX", "US 5Y Yield"),
        ("^TYX", "US 30Y Yield"),
        # dólar index / proxies
        ("DX-Y.NYB", "US Dollar Index (DXY)"),
    ]
    # ETFs de taxas/inflação para completar
    etf = [
        ("TLT", "iShares 20+ Year Treasury"),
        ("IEF", "iShares 7-10 Year Treasury"),
        ("SHY", "iShares 1-3 Year Treasury"),
        ("TIP", "iShares TIPS Bond"),
        ("LQD", "iShares iBoxx IG Corporate"),
        ("HYG", "iShares High Yield Corporate"),
        ("EMB", "iShares JPM USD EM Bond"),
    ]
    cand = [(t, f"{n} (INDEX/RATES)") for t, n in idx + etf]
    return cand

def curated_etfs_candidates() -> List[Tuple[str, str]]:
    # ETFs mega líquidos e representativos (EUA e globais)
    etfs = [
        ("SPY", "SPDR S&P 500"),
        ("VOO", "Vanguard S&P 500"),
        ("IVV", "iShares Core S&P 500"),
        ("QQQ", "Invesco QQQ"),
        ("VTI", "Vanguard Total Stock Market"),
        ("VT", "Vanguard Total World Stock"),
        ("VEA", "Vanguard FTSE Developed ex-US"),
        ("VWO", "Vanguard FTSE Emerging Markets"),
        ("EEM", "iShares MSCI Emerging Markets"),
        ("IWM", "iShares Russell 2000"),
        ("DIA", "SPDR Dow Jones"),
        ("XLK", "Technology Select Sector"),
        ("XLF", "Financial Select Sector"),
        ("XLE", "Energy Select Sector"),
        ("XLV", "Health Care Select Sector"),
        ("XLY", "Consumer Discretionary"),
        ("XLP", "Consumer Staples"),
        ("XLI", "Industrials"),
        ("XLB", "Materials"),
        ("XLU", "Utilities"),
        ("VNQ", "Vanguard REIT"),
        ("SCHD", "Schwab US Dividend Equity"),
        ("DGRO", "iShares Core Dividend Growth"),
        ("ARKK", "ARK Innovation"),
        ("SMH", "VanEck Semiconductor"),
        ("SOXX", "iShares Semiconductor"),
        ("KWEB", "KraneShares CSI China Internet"),
        ("EWZ", "iShares MSCI Brazil"),
        ("EFA", "iShares MSCI EAFE"),
        ("ACWI", "iShares MSCI ACWI"),
        ("GLD", "SPDR Gold Shares"),
        ("SLV", "iShares Silver Trust"),
        ("USO", "United States Oil Fund"),
        ("TLT", "iShares 20+ Year Treasury"),
        ("TIP", "iShares TIPS Bond"),
    ]
    # duplicatas removidas por pick_valid
    return [(t, f"{n} (ETF)") for t, n in etfs]

def curated_global_highlights_candidates() -> List[Tuple[str, str]]:
    # Mega caps + ADRs/representativos globais (alta cobertura no Yahoo)
    stocks = [
        ("AAPL", "Apple"),
        ("MSFT", "Microsoft"),
        ("NVDA", "NVIDIA"),
        ("AMZN", "Amazon"),
        ("GOOGL", "Alphabet"),
        ("META", "Meta Platforms"),
        ("TSLA", "Tesla"),
        ("BRK-B", "Berkshire Hathaway"),
        ("JPM", "JPMorgan"),
        ("V", "Visa"),
        ("MA", "Mastercard"),
        ("UNH", "UnitedHealth"),
        ("XOM", "Exxon Mobil"),
        ("LLY", "Eli Lilly"),
        ("AVGO", "Broadcom"),
        ("COST", "Costco"),
        ("KO", "Coca-Cola"),
        ("PEP", "PepsiCo"),
        ("MCD", "McDonald's"),
        ("WMT", "Walmart"),
        ("ORCL", "Oracle"),
        # ADR / internacionais em NY
        ("TSM", "TSMC"),
        ("ASML", "ASML"),
        ("NVO", "Novo Nordisk"),
        ("SAP", "SAP"),
        ("TM", "Toyota"),
        ("SONY", "Sony"),
        ("BABA", "Alibaba"),
        ("TCEHY", "Tencent ADR"),
        ("PDD", "PDD Holdings"),
        ("MELI", "MercadoLibre"),
        ("ITUB", "Itaú ADR"),
        ("VALE", "Vale ADR"),
        ("NU", "Nubank"),
        ("RBLX", "Roblox"),
        ("NFLX", "Netflix"),
    ]
    return [(t, f"{n} (GLOBAL)") for t, n in stocks]

# ============================================================
# Geração final
# ============================================================
def build_universe_500() -> Dict[str, Dict[str, str]]:
    universe: Dict[str, Dict[str, str]] = {}

    # --- 1) FX
    fx_cand = curated_fx_candidates()
    universe["CÂMBIO & FX (YAHOO)"] = pick_valid(fx_cand, target=63)

    # --- 2) Cripto (CoinGecko top cap -> Yahoo SYMBOL-USD)
    cg = fetch_coingecko_top_symbols(per_page=250, pages=3)  # 750 candidatos
    # shuffle leve para reduzir viés de falhas repetidas
    # (mantém top cap, mas validação filtra)
    universe["CRIPTOMOEDAS (TOP MARKET CAP)"] = pick_valid(cg, target=63)

    # --- 3) B3 (Ibov oficial)
    b3 = fetch_b3_ibov_constituents()
    universe["AÇÕES BRASILEIRAS – B3 (IBOV OFICIAL)"] = pick_valid(b3, target=63)

    # --- 4) US equities (NASDAQ/NYSE oficial)
    us = fetch_us_equities_candidates()
    universe["AÇÕES EUA – US (NASDAQ/NYSE OFICIAL)"] = pick_valid(us, target=63)

    # --- 5) ETFs (curados)
    etf = curated_etfs_candidates()
    universe["ETFs GLOBAIS (CURADOS + VALIDADOS)"] = pick_valid(etf, target=62)

    # --- 6) Commodities & futuros
    com = curated_commodities_candidates()
    universe["COMMODITIES & FUTUROS (YAHOO)"] = pick_valid(com, target=62)

    # --- 7) Índices & taxas
    idx = curated_indices_rates_candidates()
    universe["ÍNDICES & TAXAS (YAHOO)"] = pick_valid(idx, target=62)

    # --- 8) Global highlights
    gh = curated_global_highlights_candidates()
    universe["AÇÕES GLOBAIS EM DESTAQUE (ADR/MEGA)"] = pick_valid(gh, target=62)

    # --- sanity: se alguma categoria não bateu o target, completar com pool US (validados)
    # para garantir 500 totais sem quebrar execução
    pool = us + etf + gh
    for cat, target in CATEGORIES:
        if cat not in universe:
            universe[cat] = {}
        missing = target - len(universe[cat])
        if missing > 0:
            fill = pick_valid(pool, target=missing)
            # evita duplicatas globais
            for t, n in fill.items():
                universe[cat].setdefault(t, n)

            # se ainda faltar, tenta mais candidatos do CoinGecko
            missing2 = target - len(universe[cat])
            if missing2 > 0:
                fill2 = pick_valid(cg, target=missing2)
                for t, n in fill2.items():
                    universe[cat].setdefault(t, n)

    return universe

def export_listaativos(universe: Dict[str, Dict[str, str]], out_path: str):
    total = sum(len(v) for v in universe.values())
    lines = []
    lines.append("# ==============================================================")
    lines.append("# ATLAS | LISTA EXTREMA (500 ATIVOS) – GERADA AUTOMATICAMENTE")
    lines.append("# Fontes: B3 (Ibovespa), NASDAQ Trader (listed/other), CoinGecko (top market cap)")
    lines.append("# ==============================================================")
    lines.append("")

    # ordem fixa
    for cat, target in CATEGORIES:
        items = universe.get(cat, {})
        # garante target
        items_list = list(items.items())[:target]
        lines.append("# ==============================")
        lines.append(f"# {cat} ({len(items_list)})")
        lines.append("# ==============================")
        for t, label in items_list:
            lines.append(f"{t} — {label}")
        lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[OK] Exportado: {out_path}")
    print(f"[OK] Total (soma das categorias): {sum(min(len(universe.get(cat, {})), tgt) for cat, tgt in CATEGORIES)}")

if __name__ == "__main__":
    print("[ATLAS] Construindo universo extremo (500 ativos) com validação Yahoo...")
    uni = build_universe_500()
    export_listaativos(uni, OUT_FILE)
