import yfinance as yf
import backtrader as bt
from datetime import datetime
import pandas as pd
import matplotlib

from strats import *
#pip install yfinance backtrader pandas matplotlib
#https://www.backtrader.com/docu/quickstart/quickstart/
# Step 1: Download historical data
start_date = datetime(2013, 1, 1)
end_date = datetime(2023, 1, 1)
df = yf.download("MSFT", start=start_date, end=end_date)

df.reset_index(inplace=True)
#AAPL:Date,Open,High,Low,Close,Volume
#MSFT:Date,Open,High,Low,Close,Volume

df.columns = ['Date','Close','High','Low','Open','Volume']

df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]

df.to_csv("yf_data.csv", index=False)

# Step 2: Define strategy


# Step 3: Load into Backtrader
datafeed = bt.feeds.GenericCSVData(
    dataname='yf_data.csv',
    dtformat='%Y-%m-%d',
    datetime=0,
    open=1,
    high=2,
    low=3,
    close=4,
    volume=5,
    openinterest=-1,
    skiprows=1
)

def run_one(sym):
    cerebro = bt.Cerebro()
    data = bt.feeds.YahooFinanceData(dataname=sym,
                                     fromdate=datetime(2015, 1, 1))
    cerebro.adddata(data)
    cerebro.addstrategy(GoldenCross)
    return cerebro.run()


cerebro = bt.Cerebro(stdstats=False)


# -------------------------------------------------------------------
# replacement run_one()  – paste this into your driver script
# -------------------------------------------------------------------
# -------------------------------------------------------------------
# bullet-proof helper: try to add, fall back gracefully
# -------------------------------------------------------------------
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
def run_one(strat_cls, tag=None, **kwargs):
    cerebro = bt.Cerebro(stdstats=True)
    cerebro.broker.setcash(10_000)
    cerebro.broker.setcommission(commission=0.001)

    cerebro.adddata(datafeed)
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

strats = [GoldenCross,Rando]



from strats import (
    BuyAndHold, SmaCross, EmaCross, Rsi2, BollingerBreakout,
    MacdSignal, DonchianTurtle, MaEnvelope, TwelveMonthMomentum,
    
)

strategy_list = [
    BuyAndHold, SmaCross, EmaCross, Rsi2, BollingerBreakout,
    MacdSignal, DonchianTurtle, MaEnvelope, TwelveMonthMomentum,
    
]

for i in strategy_list :
    run_one(i)  

