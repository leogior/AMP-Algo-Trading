import pandas as pd
import numpy as np
from collections import deque
from backtesterClass.orderClass import orders
from backtesterClass.orderBookClass import OBData
from backtesterClass.tradingStratClass import autoTrader
from debug import logger
import sys


MAX_INVENT = 5

class movingAverageStrat(autoTrader):

    def __init__(self, name, short_window, long_window, z_threshold = 1, z_threshold_exit = 0.5):
        super().__init__(name)
        self.short_window = short_window
        self.long_window = long_window
        self.z_threshold = z_threshold
        self.z_threshold_exit = z_threshold_exit

        self.asset_list = OBData.assets
        asset_count = len(self.asset_list)
        self.prices = [deque(maxlen=self.long_window) for _ in range(asset_count)] # Store only up to `long_window` prices
        self.historical_short_ma = [[] for _ in range(asset_count)]
        self.historical_long_ma = [[] for _ in range(asset_count)]

        self.historical_ma_diff = [[] for _ in range(asset_count)]
        self.historical_z_score = [[] for _ in range(asset_count)]

        self.ma_diff_mean = np.zeros(asset_count)
        self.ma_diff_M2 = np.zeros(asset_count)  # For variance calculation
        self.ma_diff_count = 0
        
        self.short_sums = np.zeros(asset_count)
        self.long_sums = np.zeros(asset_count)

    def calculate_moving_averages(self, asset):
        idx = OBData.assetIdx[asset]-1
        new_price = OBData.currentPrice(asset)
        price_queue = self.prices[idx]
        price_queue.append(new_price)

        if len(price_queue) < self.short_window:
            self.short_sums[idx] += new_price
            self.long_sums[idx] += new_price
            self.historical_short_ma[idx].append(None)
            self.historical_long_ma[idx].append(None)
            self.historical_ma_diff[idx].append(None)
            return None, None, None

        if len(price_queue) < self.long_window:
            self.short_sums[idx] += new_price - (price_queue[-self.short_window] if len(price_queue) >= self.short_window else 0)
            self.long_sums[idx] += new_price
            self.historical_short_ma[idx].append(None)
            self.historical_long_ma[idx].append(None)
            self.historical_ma_diff[idx].append(None)
            return None, None, None
      

        # Fast rolling sums for short and long window
        self.short_sums[idx] += new_price - price_queue[-self.short_window]
        self.long_sums[idx] += new_price - price_queue[0]

        short_ma = self.short_sums[idx] / self.short_window
        long_ma = self.long_sums[idx] / self.long_window

        ma_diff = long_ma - short_ma
        self.historical_ma_diff[idx].append(ma_diff)

        self.ma_diff_count += 1
        delta = ma_diff - self.ma_diff_mean[idx]
        self.ma_diff_mean[idx] += delta / self.ma_diff_count
        delta2 = ma_diff - self.ma_diff_mean[idx]
        self.ma_diff_M2[idx] += delta * delta2

        # Compute std and z-score only if count >= 2
        if self.ma_diff_count >= 2:
            variance = self.ma_diff_M2[idx] / (self.ma_diff_count - 1)
            std = np.sqrt(variance)
            if std > 1e-8:
                z = (ma_diff - self.ma_diff_mean[idx]) / std
            else:
                z = 0.0
        else:
            z = 0.0


        self.historical_short_ma[idx].append(short_ma)
        self.historical_long_ma[idx].append(long_ma)

        return short_ma, long_ma, z

        
    def strategy(self, orderClass):

        MAX_INVENTORY = 10000  # Example max inventory per asset

        for asset in self.asset_list:
            short_ma, long_ma, z = self.calculate_moving_averages(asset)
            if short_ma is None or long_ma is None:
                continue

            current_price = OBData.currentPrice(asset)

            # If the signal is bullish (ie short_ma > long_ma), go long (buy)
            if z > self.z_threshold:
                if self.inventory[asset]["quantity"] < MAX_INVENTORY:
                    quantity = 1000
                    orderClass.send_order(self, asset, current_price, quantity)
                    self.AUM_available -= 1000
                    self.orderID += 1
            
            elif z <= self.z_threshold_exit and self.inventory[asset]["quantity"]>0:
                # Exit long position
                quantity = self.inventory[asset]["quantity"]
                orderClass.send_order(self, asset, current_price, -quantity)
                self.AUM_available += (self.inventory[asset]["price"]-OBData.currentPrice(asset))*-quantity

            # If the signal is bearish, go short (sell)
            elif z < -self.z_threshold:
                if self.inventory[asset]["quantity"] > -MAX_INVENTORY:
                    quantity = 1000
                    orderClass.send_order(self, asset, current_price, -quantity)
                    self.AUM_available += 1000
                    self.orderID += 1

            elif z >= -self.z_threshold_exit and self.inventory[asset]["quantity"]<0:
                # Exit short position
                quantity = self.inventory[asset]["quantity"]
                orderClass.send_order(self, asset, current_price, quantity)
                self.AUM_available += (self.inventory[asset]["price"]-OBData.currentPrice(asset))*-quantity - quantity

        # Process filled orders
        orderClass.filled_order(self)

