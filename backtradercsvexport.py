"""
backtrader_csv_export.py — batch‑test several strategies and export a CSV summary
-------------------------------------------------------------------------------
This script iterates over a list of symbols *and* a list of strategies, runs
Backtrader on each pair, collects a handful of key performance statistics, and
finally saves everything to **resulta.csv** (in the current working directory).

Run it as a normal Python script:
    $ python backtrader_csv_export.py
The code will download historical daily data via *yfinance* (make sure you have
an internet connection the first time) and will take a couple of minutes,
depending on the time‑range and strategy count.

Dependencies (same as the original script):
    pip install backtrader yfinance pandas matplotlib
"""

from __future__ import annotations
import backtrader as bt
import pandas as pd
import yfinance as yf
from typing import Dict, Any, List
import math
import numpy as np
from pathlib import Path
from DataManagement.av_downloader import av_doawnloader_main
from DataManagement.fetch_stooq_daily import import_stooq
from strats import *
from myTools import *

# ─────────────────────────── CONFIGURATION ────────────────────────────
#START = datetime(2023, 7, 7),END   = datetime(2025, 6, 6)

CONFIG_FILE = "config2.yaml"

SYMBOLS = ["SPY", "QQQ", "MTUM","EEM","TLT","GLD","IEF","MSFT", "AAPL", "NVDA","META"]
SYMBOLS = load_symbols(CONFIG_FILE)


STRATEGIES = retall()
STRATEGIES = load_strats(CONFIG_FILE)

CSV_PATH = "results.csv"
CSV_PATH = load_output_csv(CONFIG_FILE)

MINBARS = 252

DATA_DIRS = [
    Path("DataManagement/data/alpha"),        # av_downloader.py output
    Path("DataManagement/data/stooq"),        # fetch_stooq_daily.py output
]

# ────────────────────────── HELPER FUNCTIONS ──────────────────────────


# backtradercsvexport.py   (same place as before)
def make_feed(symbol: str,
              start="2005-01-01",
              end=None,
              auto_adjust=False) -> bt.feeds.PandasData | None:
    """
    1) Try to load a local CSV from DataManagement/data/*
    2) Otherwise fall back to yfinance daily data.
       Returns None when Yahoo sends back an empty DataFrame.
    """
    #import data using other thatn yfiannce
    av_doawnloader_main("config2.yaml")
    import_stooq()

    # -------- 1. local CSV ---------------------------------------------------
    for root in DATA_DIRS:
        for fname in root.glob(f"{symbol}*.csv"):
            tf   = bt.TimeFrame.Minutes if "_m" in fname.stem else bt.TimeFrame.Days
            comp = int(fname.stem.split("_")[-1][:-1]) if "_m" in fname.stem else 1
            fmt  = "%Y-%m-%d %H:%M:%S" if tf is bt.TimeFrame.Minutes else "%Y-%m-%d"
            print("used ",root,f"{symbol}*.csv")
            return bt.feeds.GenericCSVData(
                dataname     = str(fname),
                dtformat     = fmt,
                timeframe    = tf,
                compression  = comp,
                datetime     = 0,
                open         = 1,
                high         = 2,
                low          = 3,
                close        = 4,
                volume       = 5,
                openinterest = -1,
            )

    # -------- 2. yfinance fallback ------------------------------------------
    ysym = symbol.replace(".", "-").upper()            # BRK.B → BRK-B
    try:
        df = yf.download(
            ysym,
            start=start,
            end=end,
            progress=False,
            threads=False,
            auto_adjust=auto_adjust,        # be explicit!
        )
    except Exception as err:
        print(f"[skip] {symbol}: yfinance error → {err}")
        return None

    if df.empty:
        print(f"[skip] {symbol}: no data from Yahoo")
        return None
    elif len(df) < MINBARS:                 # ← ➊ NEW guard
        print(f"[skip] {symbol}: only {len(df)} bars (< {MINBARS})")
        return None

    if isinstance(df.columns, pd.MultiIndex):          # 1-hour bugs etc.
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.title() for c in df.columns]       # Open, High, …
    print("used yfiannce for:",symbol)
    return bt.feeds.PandasData(dataname=df)




def _safe_add(cerebro: bt.Cerebro, ancls, alias: str | None = None, **kwargs):
    """Attach an analyzer if available / if the current Backtrader build
    supports the given kwargs. Silently skip otherwise."""
    try:
        cerebro.addanalyzer(ancls, _name=alias or ancls.__name__.lower(), **kwargs)
    except Exception as err:
        print(f"[skip] {ancls.__name__}: {err}")


def _extract_tdrawdown(r: Any) -> float:
    """Normalise TimeDrawDown results across Backtrader versions."""
    if not isinstance(r, dict):
        return float("nan")
    if "max" in r:  # modern shape
        sub = r["max"]
        return sub.get("tdrawdown", sub.get("len", float("nan")))
    # very old shape
    return r.get("tdrawdown", r.get("len", float("nan")))


# ────────────────────────── CORE BACKTEST RUN ─────────────────────────

def run_one(symbol: str, strat_cls) -> Dict[str, Any]:
    
    
    cerebro = bt.Cerebro(stdstats=True)
    cerebro.broker.setcash(load_capital())
    cerebro.broker.setcommission(commission=load_comission())
    cerebro.broker.setslippage_perc(perc=load_slippage(), slip_open=True, slip_limit=True, slip_match=True, slip_out=False)

    cerebro.addsizer(bt.sizers.PercentSizer, percents=95)#added for sharpeRatio
    
    feed = make_feed(symbol)
    if feed is None:                   # << new
            return {} 

    cerebro.adddata(feed, name=symbol)
    
    
    cerebro.addstrategy(strat_cls)

    # ʜᴜɢᴇ list of analyzers — use what fits your BT version
# ───── inside run_one() ─────
    _safe_add(cerebro, bt.analyzers.SharpeRatio,"sharpe",timeframe=bt.TimeFrame.Days)          # ← change
    _safe_add(cerebro, bt.analyzers.SharpeRatio_A,"sharpe_ann",timeframe=bt.TimeFrame.Days)          # ← change
    _safe_add(cerebro, bt.analyzers.DrawDown,         "dd")
    _safe_add(cerebro, bt.analyzers.TimeDrawDown,     "tdd")
    _safe_add(cerebro, bt.analyzers.Calmar,           "calmar")
    _safe_add(cerebro, bt.analyzers.Returns,          "returns")
    _safe_add(cerebro, bt.analyzers.VWR,              "vwr")
    _safe_add(cerebro, bt.analyzers.SQN,              "sqn")
    _safe_add(cerebro, bt.analyzers.TradeAnalyzer,    "trades")
    _safe_add(cerebro, bt.analyzers.TimeReturn, "trets",timeframe=bt.TimeFrame.Days)
    strat = cerebro.run()[0]

    # small helper: fetch analyzer results if present
    def get(name: str, path=None, default=float("nan")):
        if hasattr(strat.analyzers, name):
            res = getattr(strat.analyzers, name).get_analysis()
            return res if path is None else path(res)
        return default


    # performance metrics
    total_ret   = get("returns",    lambda r: r.get("rtot", float("nan")) )
    rnorm100    = get("returns",    lambda r: r.get("rnorm100", float("nan"))/100)  # already %
    sharpe_d    = get("sharpe",     lambda r: r.get("sharperatio", float("nan")))
    sharpe_ann  = get("sharpe_ann", lambda r: r.get("sharperatio", float("nan")))
    max_dd      = get("dd",         lambda r: r.get("max", {}).get("drawdown", float("nan"))/100*-1)
    td_dd       = get("tdd",        _extract_tdrawdown)
    vwr         = get("vwr",        lambda r: r.get("vwr",         float("nan")))
    sqn         = get("sqn",        lambda r: r.get("sqn",         float("nan")))
    calmar = (rnorm100 / abs(max_dd)
        if not math.isnan(rnorm100) and not math.isnan(max_dd) and max_dd
        else float("nan")
    )
    # fallback ⇒ use draw-down length from the regular DD analyzer
    if math.isnan(td_dd):
        td_dd = get("dd", lambda r: r.get("max", {}).get("len", float("nan")))

    #sortino
    rets = get("trets")         # dict {datetime: return}
    if isinstance(rets, dict) and rets:
        rvals   = list(rets.values())
        target  = 0.0                         # MAR / risk-free per period
        downside = [r for r in rvals if r < target]
        if downside:                          # avoid div-by-zero
            sortino = (np.mean(rvals) - target) / np.std(downside, ddof=1)
            sortino *= math.sqrt(252)         # annualise (daily → year)
        else:
            sortino = float("inf")            # no losses → perfect
    else:
        sortino = float("nan")
    

    # trade stats
    trades_dict = get("trades")
    def _safe(d, *keys, default=float("nan")):
        for k in keys:
            if isinstance(d, dict) and k in d:
                d = d[k]
            else:
                return default
        return d

    if isinstance(trades_dict, dict):
        wins_total   = _safe(trades_dict, "won",   "total")
        trades_total = _safe(trades_dict, "total", "total")
        win_rate     = (wins_total / trades_total ) if trades_total else float("nan")

        pnl_win  = _safe(trades_dict, "won",  "pnl", "total", default=0)
        pnl_los  = _safe(trades_dict, "lost", "pnl", "total", default=0)
        profit_factor = (pnl_win / abs(pnl_los)) if pnl_los else float("nan")
        avg_trade_pl  = _safe(trades_dict, "pnl", "net", "average")
    else:
        win_rate = profit_factor = avg_trade_pl = float("nan")

    return {
        "Symbol": symbol,
        "Strategy": strat_cls.__name__,
        "TotalReturn_%": total_ret,
        "rnorm100_%": rnorm100,#its not CAGR anymore, have to change name
        "SharpeDaily": sharpe_d,
        "SharpeAnnual": sharpe_ann,
        "Calmar": calmar,
        "MaxDrawdown_%": max_dd,
        "TimeDD_bars": td_dd,
        "VWR": vwr,
        "SQN": sqn,
        "WinRate_%": win_rate,
        "ProfitFactor": profit_factor,
        "AvgTradePL": avg_trade_pl,
        "trades_total":trades_total,
        "Sortino": sortino
    }


# ────────────────────────────── MAIN ─────────────────────────────────

def main():
    rows: List[Dict[str, Any]] = []
    
    for symbol in SYMBOLS:
        print(f"▶ {symbol}")
        for strat_cls in STRATEGIES:
            print(f"    ▶ {strat_cls.__name__}")
            row = run_one(symbol, strat_cls)
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False)
    print("\nSaved results to", CSV_PATH)
    print(df.head())


if __name__ == "__main__":
    main()
