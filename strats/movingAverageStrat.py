import pandas as pd
import numpy as np
from collections import deque
from backtesterClass.orderClass import orders
from backtesterClass.orderBookClass import OBData
from backtesterClass.tradingStratClass import autoTrader
from utils.debug import logger
import sys


MAX_INVENTORY = 10000  # Example max inventory per asset

class movingAverageStrat(autoTrader):

    def __init__(self, name, short_window, long_window):
        super().__init__(name)
        self.short_window = short_window
        self.long_window = long_window

        self.asset_list = OBData.assets
        asset_count = len(self.asset_list)
        self.prices = [deque(maxlen=self.long_window+1) for _ in range(asset_count)] # Store only up to `long_window` prices
        self.historical_short_ma = [[] for _ in range(asset_count)]
        self.historical_long_ma = [[] for _ in range(asset_count)]
        self.short_sums = np.zeros(asset_count)
        self.long_sums = np.zeros(asset_count)

    def calculate_moving_averages(self, asset):
        idx = OBData.assetIdx[asset]-1
        new_price = OBData.currentPrice(asset)
        price_queue = self.prices[idx]
        price_queue.append(new_price)

        if len(price_queue) <= self.short_window:
            self.short_sums[idx] += new_price
            self.long_sums[idx] += new_price
            self.historical_short_ma[idx].append(None)
            self.historical_long_ma[idx].append(None)
            return None, None

        elif len(price_queue) < self.long_window:
            self.short_sums[idx] += new_price - price_queue[-self.short_window-1]
            self.long_sums[idx] += new_price
            self.historical_short_ma[idx].append(None)
            self.historical_long_ma[idx].append(None)
            return None, None

        # Fast rolling sums for short and long window
        self.short_sums[idx] += new_price - price_queue[-self.short_window-1]
        self.long_sums[idx] += new_price - price_queue[0]

        short_ma = self.short_sums[idx] / self.short_window
        long_ma = self.long_sums[idx] / self.long_window

        self.historical_short_ma[idx].append(short_ma)
        self.historical_long_ma[idx].append(long_ma)

        return short_ma, long_ma

        
    def strategy(self, orderClass):

        for asset in self.asset_list:
            short_ma, long_ma = self.calculate_moving_averages(asset)
            if short_ma is None or long_ma is None:
                continue

            current_price = OBData.currentPrice(asset)

            # If the signal is bullish, go long (buy)
            if short_ma > long_ma:
                if self.inventory[asset]["quantity"] < MAX_INVENTORY and self.AUM_available>0:
                    price, quantity = current_price, min(1000, self.AUM_available)
                    orderClass.send_order(self, asset, price, quantity)
                    self.AUM_available -= quantity
                    self.orderID += 1

            # If the signal is bearish, go short (sell)
            elif short_ma < long_ma:
                if self.inventory[asset]["quantity"] > -MAX_INVENTORY:
                    price, quantity = current_price, 1000
                    orderClass.send_order(self, asset, price, -quantity)
                    self.AUM_available += quantity
                    self.orderID += 1

        # Process filled orders
        self.historical_AUM.append(self.AUM_available)
        orderClass.filled_order(self)

