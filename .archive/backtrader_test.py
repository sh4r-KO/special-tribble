import yfinance as yf
import backtrader as bt
from datetime import datetime
import pandas as pd
import matplotlib
from typing import Dict, Any, List

from strats import *
#pip install yfinance backtrader pandas matplotlib
#https://www.backtrader.com/docu/quickstart/quickstart/
# Step 1: Download historical data
#start_date = datetime(2013, 1, 1)
#end_date = datetime(2023, 1, 1)
#df = yf.download("MSFT", start=start_date, end=end_date)

#df.reset_index(inplace=True)
#AAPL:Date,Open,High,Low,Close,Volume
#MSFT:Date,Open,High,Low,Close,Volume

START = datetime(2013, 1, 1)
END   = datetime(2025, 1, 1)          # or date.today()
SYMBOLS = ['MSFT', 'AAPL', 'NVDA']    # quick list you can edit on the fly
CSV_PATH = "resulta.csv"

def make_feed(symbol: str) -> bt.feeds.PandasData:
    df = yf.download(symbol,
                     start=START,
                     end=END,
                     progress=False,
                     group_by="column",   # keep or drop, both now work
                     auto_adjust=False)   # quiets the new yfinance warning

    # 1️⃣  flatten: keep the field (c[0]) if it’s a tuple
    # flatten first level if MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # normalise the names
    df.columns = [str(c).title() for c in df.columns]   # 'open' → 'Open', etc.

    needed = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing = set(needed) - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    df = df[needed].copy()

    return bt.feeds.PandasData(dataname=df)





cerebro = bt.Cerebro(stdstats=False)



def _safe_add(cerebro, ancls, alias=None, **kwargs):
    """
    Try to attach `ancls` as an analyzer.  If the installed Backtrader
    version doesn’t accept the kwargs (or the class is missing), just
    skip it and carry on.
    """
    try:
        cerebro.addanalyzer(ancls, _name=alias or ancls.__name__.lower(), **kwargs)
    except Exception as err:
        print(f"[skip] {ancls.__name__}: {err}")

def _extract_tdrawdown(r):
    """
    Normalize TimeDrawDown output across Backtrader versions.
    Returns #bars in deepest draw-down or NaN if unavailable.
    Known shapes:
        {'max': {'tdrawdown': 87, 'len': 87, ...}, ...}          # ≥ v1.9
        {'max': {'len': 87, ...}, ...}                           # ≤ v1.8
        {'len': 87, 'drawdown': -12.3, ...}                      # very old
    """
    if not isinstance(r, dict):
        return float('nan')

    if 'max' in r:                       # modern releases
        sub = r['max']
        return sub.get('tdrawdown', sub.get('len', float('nan')))

    # fallback for ancient versions
    return r.get('tdrawdown', r.get('len', float('nan')))

# -------------------------------------------------------------------
def run_one(symbol: str, strat_cls, tag=None, **kwargs):
    cerebro = bt.Cerebro(stdstats=True)
    cerebro.broker.setcash(10_000)
    cerebro.broker.setcommission(commission=0.001)

    cerebro.adddata(make_feed(symbol), name=symbol)   # <── NEW
    cerebro.addstrategy(strat_cls, **kwargs)

    # ── kitchen-sink analyzers, now wrapped in _safe_add ────────────
    _safe_add(cerebro, bt.analyzers.SharpeRatio,      'sharpe',
              timeframe=bt.TimeFrame.Days)
    _safe_add(cerebro, bt.analyzers.SharpeRatio_A,    'sharpe_ann',
              timeframe=bt.TimeFrame.Days)

    _safe_add(cerebro, bt.analyzers.DrawDown,         'dd')
    _safe_add(cerebro, bt.analyzers.TimeDrawDown,     'tdd')
    _safe_add(cerebro, bt.analyzers.Calmar,           'calmar')
    _safe_add(cerebro, bt.analyzers.Returns,          'returns')
    _safe_add(cerebro, bt.analyzers.TimeReturn,       'timeret',
              timeframe=bt.TimeFrame.Months)
    _safe_add(cerebro, bt.analyzers.AnnualReturn,     'annual')
    _safe_add(cerebro, bt.analyzers.LogReturnsRolling,'logroll')   # ← no kwargs
    _safe_add(cerebro, bt.analyzers.VWR,              'vwr')
    _safe_add(cerebro, bt.analyzers.SQN,              'sqn')
    _safe_add(cerebro, bt.analyzers.TradeAnalyzer,    'trades')
    _safe_add(cerebro, bt.analyzers.PeriodStats,      'period')

    # PyFolio is optional
    try:
        _safe_add(cerebro, bt.analyzers.PyFolio, 'pyfolio')
    except AttributeError:
        pass

    strat = cerebro.run()[0]

    # ─────────────────── helper to fetch if present ────────────────
    def get(name, path=None, default=float('nan')):
        if hasattr(strat.analyzers, name):
            res = getattr(strat.analyzers, name).get_analysis()
            return res if path is None else path(res)
        return default

    sharpe      = get('sharpe',      lambda r: r.get('sharperatio', float('nan')))
    sharpe_ann  = get('sharpe_ann',  lambda r: r.get('sharperatio', float('nan')))
    calmar      = get('calmar',      lambda r: r.get('calmar',       float('nan')))
    max_dd      = get('dd',          lambda r: r.get('max',  {}).get('drawdown',  float('nan')))
    td_dd       = get('tdd',         _extract_tdrawdown)   # helper already handles NaN
    vwr         = get('vwr',         lambda r: r.get('vwr',          float('nan')))
    sqn         = get('sqn',         lambda r: r.get('sqn',          float('nan')))
    total_ret   = get('returns',     lambda r: r.get('rtot',         float('nan')) * 100)
    cagr        = get('returns',     lambda r: r.get('ravg',         float('nan')) * 100)
    ann_ret     = get('annual')      # this one can stay as-is (dict or None)

    # -------------------  pull trade statistics safely  -----------------
    trades_dict = get('trades')  # OrderedDict or float('nan')

    def _safe(d, *keys, default=float('nan')):
        """Nested dict getter that never throws KeyError."""
        for k in keys:
            if isinstance(d, dict) and k in d:
                d = d[k]
            else:
                return default
        return d

    if isinstance(trades_dict, dict):
        wins_total   = _safe(trades_dict, 'won',   'total')
        trades_total = _safe(trades_dict, 'total', 'total')
        win_rate     = wins_total / trades_total if trades_total else float('nan')

        pnl_win = _safe(trades_dict, 'won',  'pnl', 'total', default=0)
        pnl_los = _safe(trades_dict, 'lost', 'pnl', 'total', default=0)
        profit_factor = (pnl_win / abs(pnl_los)) if pnl_los else float('nan')

        avg_trade_pl  = _safe(trades_dict, 'pnl', 'net', 'average')
    else:
        win_rate = profit_factor = avg_trade_pl = float('nan')

   

    print(f"\n=== {(tag or strat_cls.__name__).upper()} ANALYZER SUMMARY ===")
    print(f"Total Return      : {total_ret:7.2f}%")
    print(f"CAGR (avg return) : {cagr:7.2f}%")
    print(f"Sharpe (daily)    : {sharpe:7.2f}")
    print(f"Sharpe (annual)   : {sharpe_ann:7.2f}")
    print(f"Calmar            : {calmar:7.2f}")
    print(f"Max DrawDown      : {max_dd:7.2f}%   (time DD: {td_dd:5.0f} bars)")
    print(f"Vol-Wt. Return    : {vwr:7.2f}")
    print(f"SQN               : {sqn:7.2f}")
    print(f"Win Rate          : {win_rate*100:7.2f}%")
    print(f"Profit Factor     : {profit_factor:7.2f}")
    print(f"Avg Trade P/L     : ${avg_trade_pl:,.2f}")
    if ann_ret and isinstance(ann_ret, dict):
        yr_line = ", ".join(f"{yr}:{ret*100:4.1f}%" for yr, ret in sorted(ann_ret.items()))
        print(f"Annual Returns    : {yr_line}")
    print("-" * 55)

    return strat




from strats import (GoldenCross,Rando,
    BuyAndHold, SmaCross, EmaCross, Rsi2, BollingerBreakout,
    MacdSignal, DonchianTurtle, MaEnvelope, TwelveMonthMomentum,
    

)

strategy_list = [GoldenCross,Rando,
    BuyAndHold, SmaCross, EmaCross, Rsi2, BollingerBreakout,
    MacdSignal, DonchianTurtle, MaEnvelope, TwelveMonthMomentum,
    
]

# --- quick batch ---------------------------------------------------
results = []
for sym in SYMBOLS:
    for strat in strategy_list:
        print(f"\n▶ {strat.__name__} on {sym}")
        summary = run_one(sym, strat)
        results.append(summary)       # keep if you want to build a DF later

