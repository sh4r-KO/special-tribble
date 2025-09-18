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
    (-m) pip install backtrader yfinance pandas matplotlib yaml mplfinance

"""

from __future__ import annotations

import os


os.environ["MPLBACKEND"] = "Agg"  # must be set before pyplot is imported
import matplotlib
matplotlib.use("Agg")             
import matplotlib.pyplot as plt
plt.ioff()


import backtrader as bt
import pandas as pd
import yfinance as yf
from typing import Dict, Any, List
import math
import numpy as np
from pathlib import Path
from DataManagement.av_downloader import av_doawnloader_main
from DataManagement.fetch_stooq_daily import import_stooq
from datetime import datetime

from strats import *
from myTools import *
from myTools import _find_local_csv

import mplfinance as mpf

# ─────────────────────────── CONFIGURATION ────────────────────────────

CONFIG_FILE = "config.yaml"

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
DOWNLOADED_ONCE: set[str] = set()   # symbols we already tried to fetch locally




def make_feed(symbol: str,
              start: str | None = None,
              end: str | None = None,
              auto_adjust: bool = False) -> bt.feeds.PandasData | None:
    """
    1) Try local CSVs (DataManagement/data/*) with fromdate/todate.
    2) Else download via yfinance using start/end.
    """
    # normalize dates
    start_dt = datetime.fromisoformat(start) if start else None
    end_dt   = datetime.fromisoformat(end)   if end   else None

    cand = _find_local_csv(symbol, DATA_DIRS)
    if cand:
        tf   = bt.TimeFrame.Minutes if "_m" in cand.stem else bt.TimeFrame.Days
        comp = int(cand.stem.split("_")[-1][:-1]) if "_m" in cand.stem else 1
        fmt  = "%Y-%m-%d %H:%M:%S" if tf is bt.TimeFrame.Minutes else "%Y-%m-%d"
        print("using local file:", cand)
        return bt.feeds.GenericCSVData(
            dataname     = str(cand),
            dtformat     = fmt,
            timeframe    = tf,
            compression  = comp,
            datetime     = 0, open=1, high=2, low=3, close=4, volume=5,
            openinterest = -1,
            fromdate     = start_dt,
            todate       = end_dt,
        )

    # one-time try to populate local store
    if symbol not in DOWNLOADED_ONCE:
        try:
            av_doawnloader_main(CONFIG_FILE)
            import_stooq()
        except Exception as err:
            print(f"[warn] local data fetchers failed for {symbol}: {err}")
        finally:
            DOWNLOADED_ONCE.add(symbol)

        cand = _find_local_csv(symbol, DATA_DIRS)
        if cand:
            tf   = bt.TimeFrame.Minutes if "_m" in cand.stem else bt.TimeFrame.Days
            comp = int(cand.stem.split("_")[-1][:-1]) if "_m" in cand.stem else 1
            fmt  = "%Y-%m-%d %H:%M:%S" if tf is bt.TimeFrame.Minutes else "%Y-%m-%d"
            print("using local file after fetch:", cand)
            return bt.feeds.GenericCSVData(
                dataname     = str(cand),
                dtformat     = fmt,
                timeframe    = tf,
                compression  = comp,
                datetime     = 0, open=1, high=2, low=3, close=4, volume=5,
                openinterest = -1,
                fromdate     = start_dt,
                todate       = end_dt,
            )

    # fallback: yfinance (dates applied here too)
    ysym = symbol.replace(".", "-").upper()
    try:
        df = yf.download(
            ysym,
            start=start,       # strings are fine for yfinance
            end=end,
            progress=False,
            threads=False,
            auto_adjust=auto_adjust
        )
    except Exception as err:
        print(f"[skip] {symbol}: yfinance error → {err}")
        return None

    if df.empty:
        print(f"[skip] {symbol}: no data from Yahoo")
        return None
    elif len(df) < MINBARS:
        print(f"[skip] {symbol}: only {len(df)} bars (< {MINBARS})")
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.title() for c in df.columns]
    print("used yfinance for:", symbol)
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
    cerebro.broker.set_slippage_perc(perc=load_slippage(), slip_open=True, slip_limit=True, slip_match=True, slip_out=False)

    cerebro.addsizer(bt.sizers.PercentSizer, percents=95)#added for sharpeRatio
    
    feed = make_feed(symbol, start=load_startDate(), end=load_endDate())
    if feed is None:             
            return {} 

    cerebro.adddata(feed, name=symbol)
    cerebro.addstrategy(strat_cls)

    _safe_add(cerebro, bt.analyzers.SharpeRatio,"sharpe",timeframe=bt.TimeFrame.Days)         
    _safe_add(cerebro, bt.analyzers.SharpeRatio_A,"sharpe_ann",timeframe=bt.TimeFrame.Days)        
    _safe_add(cerebro, bt.analyzers.DrawDown,         "dd")
    _safe_add(cerebro, bt.analyzers.TimeDrawDown,     "tdd")
    _safe_add(cerebro, bt.analyzers.Calmar,           "calmar")
    _safe_add(cerebro, bt.analyzers.Returns,          "returns")
    _safe_add(cerebro, bt.analyzers.VWR,              "vwr")
    _safe_add(cerebro, bt.analyzers.SQN,              "sqn")
    _safe_add(cerebro, bt.analyzers.TradeAnalyzer,    "trades")
    _safe_add(cerebro, bt.analyzers.TimeReturn, "trets",timeframe=bt.TimeFrame.Days)
    _safe_add(cerebro, EntryExitMarks, "marks")     

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

    # save png plot chart(s)
    outdir = Path("output/plots")
    outdir.mkdir(exist_ok=True)

    
    # pretty mplfinance chart with buy/sell markers
    try:
        # collect markers
        recs = getattr(strat.analyzers, "marks").get_analysis() if hasattr(strat.analyzers, "marks") else []
        buys  = [dt for kind, dt, px in recs if kind == 'buy']
        sells = [dt for kind, dt, px in recs if kind == 'sell']

        # price dataframe
        df = _price_df_for(symbol)

        # clip to backtest window if you want (optional)
        df = df.loc[pd.to_datetime(load_startDate()): pd.to_datetime(load_endDate())]

        # build addplots
        def _marker_series(df, event_dts):
            if not event_dts:
                return None
            # normalize both to dates
            ev_idx = pd.to_datetime(event_dts).tz_localize(None).normalize()
            idx_norm = df.index.tz_localize(None).normalize() if df.index.tz is not None else df.index.normalize()
            mask = idx_norm.isin(ev_idx)
            # series with NaN except where we have an event (use Close for y)
            return pd.Series(np.where(mask, df['Close'].values, np.nan), index=df.index)

        buy_ser  = _marker_series(df, buys)
        sell_ser = _marker_series(df, sells)

        aps = []
        if buy_ser is not None:
            aps.append(mpf.make_addplot(buy_ser, type='scatter', marker='^', markersize=80))
        if sell_ser is not None:
            aps.append(mpf.make_addplot(sell_ser, type='scatter', marker='v', markersize=80))

        pretty_dir = Path("output/pretty"); pretty_dir.mkdir(parents=True, exist_ok=True)
        pretty_png = pretty_dir / f"{symbol}_{strat_cls.__name__}.png"

        mpf.plot(
            df, type='line', volume=True, mav=(12, 26),
            addplot=aps, style='yahoo',
            figratio=(16,9), figsize=(12,6),
            title=f"{symbol} · {strat_cls.__name__}",
            savefig=dict(fname=str(pretty_png), dpi=180, bbox_inches="tight"),
        )
        print("Saved clean chart:", pretty_png)
    except Exception as e:
        print("[warn] pretty plot failed:", e)

    

    return {
        "Symbol": symbol,
        "Strategy": strat_cls.__name__,
        "TotalReturn_%": total_ret,
        "rnorm100_%": rnorm100,
        "SharpeDaily": sharpe_d,
        "SharpeAnnual": sharpe_ann,
        "Calmar": calmar,
        "MaxDrawdown_%": max_dd,
        "TimeDD_bars": -td_dd,
        "VWR": vwr,
        "SQN": sqn,
        "WinRate_%": win_rate,
        "ProfitFactor": profit_factor,
        "AvgTradePL": avg_trade_pl,
        "trades_total":trades_total,
        "Sortino": sortino
    }


def creategraph(row: dict, thresholds: dict) -> None:
    outdir = Path("output/graphs")
    outdir.mkdir(exist_ok=True)

    symbol = row.get("Symbol")
    strat  = row.get("Strategy")

    color = ["#19183B","#708993","#A1C2BD","#E7F2EF"]
    #doing the gaugeqs one first
    for i, (metric_name, metric_value) in enumerate(row.items()):

        if not isinstance(metric_value, (int, float)):
            continue
        if math.isnan(metric_value):
            continue

        # Match thresholds row
        th = thresholds.get(metric_name.replace("_%", "").replace("_", " "), None)
        if th is None:
            print(f"No thresholds for {metric_name}, skipping")
            continue

        vmin = th["min"]
        vmax = th["best"]
        marker = th["good"]


        fig, ax = gauge_percent(
            metric_value,
            vmin=vmin, vmax=vmax,
            marker=marker,
            title=f"{symbol}_{strat}_{metric_name}",
            as_percent=("_%" in metric_name),
            color=color
        )

        #outfile = outdir / f"{symbol}_{strat}_{metric_name}.png"
        outfile = outdir / f"{metric_name}.png"
        fig.savefig(outfile, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {outfile}")
    
    #other kpi than gauge : 

 # ────────────────────────── PLOTTING HELPERS ──────────────────────────

class EntryExitMarks(bt.Analyzer):
    """Record buy/sell moments based on position changes."""
    def start(self):
        self.prev_size = 0
        self.recs = []  # list of tuples: ('buy'|'sell', datetime, price)
    def next(self):
        size = self.strategy.position.size
        if size > 0 and self.prev_size <= 0:                # entry long
            self.recs.append(('buy',  self.strategy.datetime.datetime(0), float(self.data.close[0])))
        if size == 0 and self.prev_size != 0:               # flat out
            self.recs.append(('sell', self.strategy.datetime.datetime(0), float(self.data.close[0])))
        self.prev_size = size
    def get_analysis(self):
        return self.recs


def _price_df_for(symbol: str) -> pd.DataFrame:
    """
    Return a Pandas OHLCV DataFrame for mplfinance, regardless of data source.
    Index is DatetimeIndex; columns: Open,High,Low,Close,Volume.
    """
    cand = _find_local_csv(symbol, DATA_DIRS)
    if cand:
        df = pd.read_csv(cand, parse_dates=['Date'])
        df = df.rename(columns=str.title).set_index('Date')[['Open','High','Low','Close','Volume']]
        return df.sort_index()

    # fallback: yfinance (same as make_feed)
    ysym = symbol.replace('.', '-').upper()
    df = yf.download(ysym, start=load_startDate(), progress=False, threads=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.title() for c in df.columns]
    return df[['Open','High','Low','Close','Volume']].sort_index()

# ────────────────────────────── MAIN ─────────────────────────────────

def main():
    rows: List[Dict[str, Any]] = []
    
    for symbol in SYMBOLS:
        print(f"▶ {symbol}")
        for strat_cls in STRATEGIES:
            print(f"    ▶ {strat_cls.__name__}")
            row = run_one(symbol, strat_cls)
            thresholds = load_thresholds()


            creategraph(row, thresholds)
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False)
    df.to_csv("output/results.csv", index=False)
    print("\nSaved results to", CSV_PATH)
    print(df.head())

if __name__ == "__main__":
    main()
