import yfinance as yf
import backtrader as bt
from datetime import datetime
import pandas as pd
import matplotlib

class TestStrategy(bt.Strategy):
    def __init__(self):
        self.sma = bt.indicators.SimpleMovingAverage(self.data.close, period=20)
        self.rsi = bt.indicators.RelativeStrengthIndex(period=14)
        self.macd = bt.indicators.MACD()
        self.bollinger = bt.indicators.BollingerBands(period=20)

    def next(self):
        if not self.position:
            if self.data.close < self.bollinger.lines.bot:
                self.buy()
        elif self.data.close > self.bollinger.lines.top:
            self.sell()

    def log(self, txt, dt=None):
        dt = dt or self.data.datetime.date(0)
        print(f'{dt.isoformat()} {txt}')
    