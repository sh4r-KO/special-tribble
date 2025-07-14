# strat.py  ──────────────────────────────────────────────────────────
import backtrader as bt
import random
import backtrader as bt
import math


import inspect, sys, backtrader as bt

def retall():
    """
    Return a list of all Strategy subclasses defined in this module.
    """
    this_module = sys.modules[__name__]

    return [
        cls
        for _, cls in inspect.getmembers(this_module, inspect.isclass)
        if issubclass(cls, bt.Strategy)           # only Backtrader strategies
        and cls is not bt.Strategy                # exclude the base class itself
        and cls.__module__ == __name__            # defined here, not imported
    ]


# ── 1. “Golden-Cross” SMA trend-follower ───────────────────────────
class GoldenCross(bt.Strategy):
    """
    Buy when SMA-50 crosses above SMA-200; exit when it crosses back.
    """
    params = dict(fast=50, slow=200)

    def __init__(self):
        sma_fast = bt.ind.SMA(period=self.p.fast)
        sma_slow = bt.ind.SMA(period=self.p.slow)
        self.crossover = bt.ind.CrossOver(sma_fast, sma_slow)

    def next(self):
        if not self.position and self.crossover > 0:
            self.buy()
        elif self.position and self.crossover < 0:
            self.close()


# ── 2. “Rando” coin-toss strategy (for demonstration only) ─────────
#class Rando(bt.Strategy):
#    """
#    Opens a random long/short position each bar, then has a 10 % chance
#    Opens a random long/short position each bar, then has a 10 % chance
#    per bar to close the position.
#    """
#    params = dict(exit_chance=0.10)
#
#    def next(self):
#        if not self.position:
#            (self.buy if random.random() < 0.5 else self.sell)()
#        elif random.random() < self.p.exit_chance:
#            self.close()


#  classic_strats.py – 10 popular strategies       
#  all positions are closed in .stop()             


# 1) Buy & Hold … but exit at the end
class BuyAndHold(bt.Strategy):
    def next(self):
        if not self.position:
            self.buy()#self.buy(size=1)                     # 100 % allocation
    def stop(self):
        if self.position:                        # flatten on the last bar
            self.close()


# 2) Simple-Moving-Average Cross (50 / 200) — Golden-Cross trend
class SmaCross(bt.Strategy):
    params = dict(fast=50, slow=200)
    def __init__(self):
        fast, slow = bt.ind.SMA(period=self.p.fast), bt.ind.SMA(period=self.p.slow)
        self.cross = bt.ind.CrossOver(fast, slow)
    def next(self):
        if not self.position and self.cross > 0:
            self.buy()
        elif self.position and self.cross < 0:
            self.close()
    def stop(self):
        if self.position:
            self.close()


# 3) Exponential-MA Cross (12 / 26) — faster variant
class EmaCross(bt.Strategy):
    params = dict(fast=12, slow=26)
    def __init__(self):
        fast, slow = bt.ind.EMA(period=self.p.fast), bt.ind.EMA(period=self.p.slow)
        self.cross = bt.ind.CrossOver(fast, slow)
        
    def next(self):
        if not self.position and self.cross > 0:
            self.buy()
        elif self.position and self.cross < 0:
            self.close()
    def stop(self):
        if self.position:
            self.close()


# 4) 2-period RSI Mean-Reversion (Connors RSI-2)
class Rsi2(bt.Strategy):
    params = dict(
        rsi_len=2,
        oversold=10,
        overbought=70,          # faster exit
        ma_len=200,             # trend filter
    )

    def __init__(self):
        self.rsi = bt.ind.RSI(self.data.close, period=self.p.rsi_len, safediv=True)
        self.ma  = bt.ind.SMA(self.data.close, period=self.p.ma_len)

        # Crossing helpers
        self.cross_under = bt.ind.CrossDown(self.rsi, self.p.oversold)
        self.cross_over  = bt.ind.CrossOver(self.rsi, self.p.overbought)

    def next(self):
        if not self.position:
            if self.data.close[0] > self.ma[0] and self.cross_under[0]:
                self.buy()                       # optionally size=...
        else:
            if self.cross_over[0]:
                self.close()                     # take profit

    def stop(self):
        if self.position:
            self.close()

# 5) Bollinger Band Breakout (buy upper-band break, sell mid-band)
class BollingerBreakout(bt.Strategy):
    params = dict(period=20, devfactor=2.0)
    def __init__(self):
        self.bbands = bt.ind.BollingerBands(period=self.p.period, devfactor=self.p.devfactor)
    def next(self):
        if not self.position and self.data.close > self.bbands.lines.top:
            self.buy()
        elif self.position and self.data.close < self.bbands.lines.mid:
            self.close()
    def stop(self):
        if self.position:
            self.close()


# 6) MACD Trend-Follower (signal cross)
class MacdSignal(bt.Strategy):
    def __init__(self):
        macd = bt.ind.MACD()
        self.cross = bt.ind.CrossOver(macd.macd, macd.signal)
    def next(self):
        if not self.position and self.cross > 0:
            self.buy()
        elif self.position and self.cross < 0:
            self.close()
    def stop(self):
        if self.position:
            self.close()


# 7) 20-day Donchian / Turtle Channel Breakout
class DonchianTurtle(bt.Strategy):
    params = dict(period=20)
    def __init__(self):
        self.highest = bt.ind.Highest(self.data.high,  period=self.p.period)
        self.lowest  = bt.ind.Lowest (self.data.low,   period=self.p.period)
    def next(self):
        if not self.position and self.data.close >= self.highest[-1]:
            self.buy()
        elif self.position and self.data.close <= self.lowest[-1]:
            self.close()
    def stop(self):
        if self.position:
            self.close()


# 8) Dual Moving-Average Envelope (buy when price > upper band, sell mid)
class MaEnvelope(bt.Strategy):
    params = dict(period=50, perc=0.02)        # 2 % envelope
    def __init__(self):
        sma = bt.ind.SMA(period=self.p.period)
        self.upper = sma * (1 + self.p.perc)
        self.lower = sma * (1 - self.p.perc)
    def next(self):
        if not self.position and self.data.close > self.upper:
            self.buy()
        elif self.position and self.data.close < self.lower:
            self.close()
    def stop(self):
        if self.position:
            self.close()


# 9) 12-month Momentum (monthly data recommended)
class TwelveMonthMomentum(bt.Strategy):
    params = dict(lookback=252)                # ≈ 12-month trading days
    def next(self):
        if len(self.data) <= self.p.lookback:
            return
        pct = (self.data.close[0] / self.data.close[-self.p.lookback]) - 1
        if not self.position and pct > 0:
            self.buy()
        elif self.position and pct <= 0:
            self.close()
    def stop(self):
        if self.position:
            self.close()


# 10) Volatility-Weighted Average Price (VWAP) Reversion intraday demo
class VwapReversion(bt.Strategy):
    """
    Fade excursions beyond ±devfactor × rolling σ around VWAP.
    Closes at the VWAP touch.
    """
    params = dict(period=30, devfactor=1.5)

    def __init__(self):
        # -------- reference our own indicator, NOT bt.ind.VWAP ------------
        self.vwap = VWAP()                                # cumulative VWAP
        # If you prefer a rolling VWAP use: self.vwap = VWAP(period=self.p.period)
        # ------------------------------------------------------------------

        self.std = bt.ind.StdDev(self.data.close, period=self.p.period)

    def next(self):
        upper = self.vwap + self.p.devfactor * self.std
        lower = self.vwap - self.p.devfactor * self.std

        if not self.position and self.data.close < lower:
            self.buy()
        elif self.position and self.data.close > self.vwap:
            self.close()

    def stop(self):
        # safety-net: flatten if something slipped through to the final bar
        if self.position:
            self.close()


class AtrTrailingTrend(bt.Strategy):
    params = dict(period=20, atr_mult=3.0)

    def __init__(self):
        self.atr   = bt.ind.ATR(period=self.p.period)
        self.highs = bt.ind.Highest(self.data.high, period=self.p.period)
        self.stop  = self.highs - self.p.atr_mult * self.atr

    def next(self):
        if not self.position and self.data.close[0] > self.stop[0]:
            self.buy()
        elif self.position and self.data.close[0] < self.stop[0]:
            self.close()

class KeltnerBreakout(bt.Strategy):
    params = dict(period=20, atr_mult=1.5)

    def __init__(self):
        ema  = bt.ind.EMA(period=self.p.period)
        atr  = bt.ind.ATR(period=self.p.period)
        self.upper = ema + self.p.atr_mult * atr
        self.middle = ema

    def next(self):
        if not self.position and self.data.close > self.upper:
            self.buy()
        elif self.position and self.data.close < self.middle:
            self.close()

class IchimokuTrend(bt.Strategy):
    params = dict(tenkan=9, kijun=26, senkou=52)

    def __init__(self):
        ichimoku = bt.ind.Ichimoku(
            tenkan=self.p.tenkan, kijun=self.p.kijun, senkou=self.p.senkou
        )
        self.tenkan = ichimoku.tenkan_sen
        self.kijun  = ichimoku.kijun_sen
        self.cross  = bt.ind.CrossOver(self.tenkan, self.kijun)
        self.cloud_top = ichimoku.senkou_span_a
        self.cloud_bot = ichimoku.senkou_span_b

    def next(self):
        in_cloud_uptrend = self.data.close[0] > max(self.cloud_top[0], self.cloud_bot[0])
        if not self.position and in_cloud_uptrend and self.cross > 0:
            self.buy()
        elif self.position and (self.cross < 0 or not in_cloud_uptrend):
            self.close()

class RsiStochCombo(bt.Strategy):
    params = dict(rsi_len=2, sto_k=14, sto_d=3, oversold=15, overbought=85)

    def __init__(self):
        self.rsi = bt.ind.RSI(period=self.p.rsi_len, safediv=True)
        stoch = bt.ind.Stochastic(self.data, period=self.p.sto_k, period_dfast=self.p.sto_d)
        self.sto_k = stoch.percK
        self.cross_over = bt.ind.CrossOver(self.rsi, self.p.overbought)
        self.cross_under = bt.ind.CrossDown(self.rsi, self.p.oversold)

    def next(self):
        if not self.position:
            if self.rsi < self.p.oversold and self.sto_k < self.p.oversold:
                self.buy()
        else:
            if self.rsi > self.p.overbought or self.cross_over:
                self.close()

class MonthlyRotator(bt.Strategy):
    params = dict(long_lookback=252, short_lookback=126)

    def __init__(self):
        self.last_month = -1

    def next(self):
        # run logic only at month change
        if self.data.datetime.date(0).month == self.last_month:
            return
        self.last_month = self.data.datetime.date(0).month

        llb, slb = self.p.long_lookback, self.p.short_lookback
        if len(self.data) < llb:
            return
        long_ret = (self.data.close[0] / self.data.close[-llb]) - 1
        short_ret = (self.data.close[0] / self.data.close[-slb]) - 1

        if not self.position and long_ret > 0 and short_ret > 0:
            self.buy()
        elif self.position and (long_ret <= 0 or short_ret <= 0):
            self.close()

class AdxBreakout(bt.Strategy):
    params = dict(chan_period=20, adx_period=14, adx_thresh=25)

    def __init__(self):
        self.highest = bt.ind.Highest(self.data.high, period=self.p.chan_period)
        self.lowest  = bt.ind.Lowest (self.data.low,  period=self.p.chan_period)
        self.adx = bt.ind.ADX(period=self.p.adx_period)

    def next(self):
        if not self.position:
            if self.data.close >= self.highest[-1] and self.adx > self.p.adx_thresh:
                self.buy()
        else:
            if self.data.close <= self.lowest[-1]:
                self.close()

class AtrChannelReversion(bt.Strategy):
    params = dict(period=30, atr_mult=1.8)

    def __init__(self):
        self.vwap  = VWAP()                                # ← our local VWAP
        atr        = bt.ind.ATR(period=self.p.period)
        self.upper = self.vwap + self.p.atr_mult * atr
        self.lower = self.vwap - self.p.atr_mult * atr

    def next(self):
        if not self.position and self.data.close < self.lower:
            self.buy()
        elif self.position and self.data.close >= self.vwap:
            self.close()

class VWAP(bt.Indicator):
    """
    Volume-Weighted Average Price.

    • If period == 0 (default) we keep a **cumulative VWAP** from the first bar.  
    • Set period = N to roll over a **look-back VWAP** (N-bar window).

    Plots on the main chart (subplot=False) and falls back to the close price if
    the volume for a bar is zero.
    """
    lines = ('vwap',)
    params = dict(period=0)          # 0 = cumulative
    plotinfo = dict(subplot=False)

    def __init__(self):
        pv = self.data.close * self.data.volume            # price × volume

        if self.p.period:                                  # rolling window
            cum_pv  = bt.ind.SumN(pv,              period=self.p.period)
            cum_vol = bt.ind.SumN(self.data.volume, period=self.p.period)
        else:                                              # cumulative
            cum_pv  = bt.ind.Accum(pv)
            cum_vol = bt.ind.Accum(self.data.volume)

        # Safe divide — avoids ZeroDivisionError on zero-volume bars
        # 1) do the division safely, but put a *scalar* in zero=
        raw_vwap = bt.ind.DivByZero(cum_pv, cum_vol, zero=float('nan'))

        # 2) where volume is zero, fall back to the *close* price
        self.l.vwap = bt.ind.If(cum_vol == 0, self.data.close, raw_vwap)

