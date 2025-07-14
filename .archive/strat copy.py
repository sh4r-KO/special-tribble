import backtrader as bt
import random

import backtrader as bt
import numpy as np

import backtrader as bt
import numpy as np

# --- Reusable Mixin for Performance Metrics ---
class PerformanceMixin:
    def __init__(self):
        self.starting_cash = None
        self.trades = []
        self.equity_curve = []

    def start(self):
        self.starting_cash = self.broker.getvalue()

    def next(self):
        self.equity_curve.append(self.broker.getvalue())
        
    def notify_trade(self, trade):
        if trade.isclosed:
            pnl = trade.pnlcomm             # profit net of commission
            self.trades.append(pnl)

    def report_performance(self):
        final_cash = self.broker.getvalue()
        net_profit = final_cash - self.starting_cash
        roi = (net_profit / self.starting_cash) * 100

        eq  = np.array(self.equity_curve)
        if len(eq) >= 2:                # ← **only** check for 2+ points
            returns = np.diff(eq) / eq[:-1]
            annual_return = (eq[-1] / eq[0]) ** (252 / (len(eq)-1)) - 1
            volatility    = returns.std(ddof=0) * np.sqrt(252)
            sharpe        = returns.mean() / returns.std(ddof=0) * np.sqrt(252)
            running_max   = np.maximum.accumulate(eq)
            max_dd        = ((eq - running_max) / running_max).min()
        else:
            annual_return = volatility = sharpe = max_dd = np.nan

        wins   = [p for p in self.trades if p > 0]
        losses = [p for p in self.trades if p < 0]

        if wins or losses:
            win_rate      = len(wins) / (len(wins) + len(losses))
            profit_factor = sum(wins) / abs(sum(losses)) if losses else np.inf
        else:
            win_rate = profit_factor = np.nan


        #win_rate = len(wins) / len(self.trades) * 100 if self.trades else 0
        #profit_factor = sum(wins) / abs(sum(losses)) if losses else float('inf')

        print(f"\nPerformance Summary:")
        print(f"  - Starting Cash   : ${self.starting_cash:.2f}")
        print(f"  - Final Cash      : ${final_cash:.2f}")
        print(f"  - Net Profit      : ${net_profit:.2f}")
        print(f"  - ROI             : {roi:.2f}%")
        print(f"  - Annual Return   : {annual_return:.2f}%")
        print(f"  - Volatility      : {volatility:.2f}%")
        print(f"  - Sharpe Ratio    : {sharpe:.2f}")
        print(f"  - Max Drawdown    : {max_dd:.2f}%")
        print(f"  - Win Rate        : {win_rate:.2f}%")
        print(f"  - Profit Factor   : {profit_factor:.2f}")

# --- Strategy: Golden Cross ---
class GoldenCross(bt.Strategy, PerformanceMixin):
    def __init__(self):
        PerformanceMixin.__init__(self)
        self.starting_cash = self.broker.getvalue()

        sma50 = bt.ind.SMA(period=50)
        sma200 = bt.ind.SMA(period=200)
        self.crossover = bt.ind.CrossOver(sma50, sma200)

    def next(self):

        if not self.position:
            if self.crossover > 0:
                self.buy()
        elif self.crossover < 0:
            self.close()
        super().next()  # collect equity data


    def notify_trade(self, trade):
        super().notify_trade(trade)

    def start(self):
        super().start()

    def stop(self):
        self.report_performance()




class SmaCross(bt.SignalStrategy):
    def __init__(self):
        
        sma1 = bt.ind.SMA(period=10)
        sma2 = bt.ind.SMA(period=30)
        crossover = bt.ind.CrossOver(sma1, sma2)
        self.signal_add(bt.SIGNAL_LONG, crossover)

    def start(self):
        self.starting_cash = self.broker.getvalue()
        print(f"{self.__class__.__name__} - Starting Cash: {self.starting_cash:.2f}")

    def stop(self):
        final_cash = self.broker.getvalue()
        print(f"{self.__class__.__name__} - Final Cash: {final_cash:.2f}")

class RsiStrategy(bt.Strategy):
    def __init__(self):
        self.starting_cash = 10000
        self.rsi = bt.ind.RSI(period=14)

    def start(self):
        self.starting_cash = self.broker.getvalue()
        print(f"{self.__class__.__name__} - Starting Cash: {self.starting_cash:.2f}")

    def next(self):
        if not self.position:
            if self.rsi < 30:
                self.buy()
        elif self.rsi > 70:
            self.sell()

    def stop(self):
        final_cash = self.broker.getvalue()
        print(f"{self.__class__.__name__} - Final Cash: {final_cash:.2f}")

class MacdStrategy(bt.Strategy):
    def __init__(self):
        self.starting_cash = 10000
        self.macd = bt.ind.MACD()
        self.crossover = bt.ind.CrossOver(self.macd.macd, self.macd.signal)

    def start(self):
        self.starting_cash = self.broker.getvalue()
        print(f"{self.__class__.__name__} - Starting Cash: {self.starting_cash:.2f}")

    def next(self):
        if self.crossover > 0:
            self.buy()
        elif self.crossover < 0:
            self.sell()

    def stop(self):
        final_cash = self.broker.getvalue()
        print(f"{self.__class__.__name__} - Final Cash: {final_cash:.2f}")

class BollingerStrategy(bt.Strategy):
    def __init__(self):
        self.starting_cash = 10000
        self.bb = bt.ind.BollingerBands(period=20)

    def start(self):
        self.starting_cash = self.broker.getvalue()
        print(f"{self.__class__.__name__} - Starting Cash: {self.starting_cash:.2f}")

    def next(self):
        if not self.position:
            if self.data.close < self.bb.lines.bot:
                self.buy()
        elif self.data.close > self.bb.lines.top:
            self.sell()

    def stop(self):
        final_cash = self.broker.getvalue()
        print(f"{self.__class__.__name__} - Final Cash: {final_cash:.2f}")

class BuyHold(bt.Strategy):
    def __init__(self):
        self.starting_cash = 10000
        self.buy_order = None

    def start(self):
        self.starting_cash = self.broker.getvalue()
        print(f"{self.__class__.__name__} - Starting Cash: {self.starting_cash:.2f}")

    def next(self):
        if not self.position and self.buy_order is None:
            self.buy_order = self.buy()

    def stop(self):
        final_cash = self.broker.getvalue()
        print(f"{self.__class__.__name__} - Final Cash: {final_cash:.2f}")

class MomentumStrategy(bt.Strategy):
    def __init__(self):
        self.starting_cash = 10000
        self.sma10 = bt.ind.SMA(period=10)
        self.sma30 = bt.ind.SMA(period=30)
        self.rsi = bt.ind.RelativeStrengthIndex(period=14)
        #self.crossover = bt.ind.CrossOver(sma10, sma30)

    def start(self):
        self.starting_cash = self.broker.getvalue()
        print(f"{self.__class__.__name__} - Starting Cash: {self.starting_cash:.2f}")

    def next(self):
        if self.sma10[0] > self.sma30[0] and self.data.close[0] <self.sma10[0] and self.rsi[0] < 50:
            self.buy()
        elif self.data.close[0] >self.sma10[0] and self.rsi[0] > 60:
            self.sell()

    def stop(self):
        final_cash = self.broker.getvalue()
        print(f"{self.__class__.__name__} - Final Cash: {final_cash:.2f}")

class RandomStrategy(bt.Strategy):
    def __init__(self):
        self.starting_cash = 10000

    def start(self):
        self.starting_cash = self.broker.getvalue()
        print(f"{self.__class__.__name__} - Starting Cash: {self.starting_cash:.2f}")

    def next(self):
    #if not self.position:
        if random.randint(0, 1):  # 50% chance
            self.buy()
        else:
            self.sell()
    #else:
        # Randomly decide to close the position
        if random.random() < 0.1:  # 10% chance to exit
            self.close()

    def stop(self):
        if self.position : self.close()
        final_cash = self.broker.getvalue()
        print(f"{self.__class__.__name__} - Final Cash: {final_cash:.2f}")
