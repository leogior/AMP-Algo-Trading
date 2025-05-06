import pandas as pd
import numpy as np
from collections import deque
from backtesterClass.orderClass import orders
from backtesterClass.orderBookClass import OBData
from backtesterClass.tradingStratClass import autoTrader
from debug import logger
import sys

MAX_INVENTORY = 10000  # Example max inventory per asset


class rsiStrat(autoTrader):
    
    def __init__(self, name, window=50, sellThreshold=70, buyThreshold=40, alpha=0.02):
        super().__init__(name)
        self.name = name
        self.asset_list = OBData.assets
        self.asset_count = len(self.asset_list)
        self.windowLengt = window
        self.windowRSI = [deque(maxlen=self.windowLengt) for _ in range(self.asset_count)]
        self.sellThreshold = sellThreshold
        self.buyThreshold = buyThreshold
        self.alpha = alpha  # Smoothing factor
        self.historical_RSI = [[] for _ in range(self.asset_count)]
    
    def compute_RSI(self, asset):
        
        idx = OBData.assetIdx[asset]-1
        self.windowRSI[idx].append(OBData.currentPrice(asset))

        if len(self.windowRSI[idx]) > 1:
            delta = self.windowRSI[idx][-1] - self.windowRSI[idx][-2]
        else:
            delta = 0  # No change for the first element

        # Initialize rolling averages if not already done
        if not hasattr(self, "avg_gain"):
            self.avg_gain = np.zeros(self.asset_count)
            self.avg_loss = np.zeros(self.asset_count)

        # Compute current gain and loss
        gain = max(delta, 0)
        loss = max(-delta, 0)

        # Update rolling averages using exponential moving average (EMA)

        self.avg_gain[idx] = (1 - self.alpha) * self.avg_gain[idx] + self.alpha * gain
        self.avg_loss[idx] = (1 - self.alpha) * self.avg_loss[idx] + self.alpha * loss


        # Avoid division by zero and compute RSI
        if self.avg_loss[idx] == 0:
            rsi = 100  # No losses mean RSI is maxed
        else:
            rs = self.avg_gain[idx] / self.avg_loss[idx]
            rsi = 100 - (100 / (1 + rs))


        if len(self.windowRSI[idx]) < self.windowLengt:
            # print(f"len(self.windowRSI[idx]): {len(self.windowRSI[idx])}, self.windowLengt:{self.windowLengt}")
            self.historical_RSI[idx].append(None)
            return None
        else:
            self.historical_RSI[idx].append(rsi)
            return rsi

    def strategy(self, orderClass):
        
        for asset in self.asset_list:
            rsi = self.compute_RSI(asset)
            current_price = OBData.currentPrice(asset)

            if rsi is None:
                pass
            
            else:

                
                if rsi <= self.buyThreshold:  # Buy Signal
                    if self.inventory[asset]["quantity"] < MAX_INVENTORY and self.AUM_available:
                        price, quantity = current_price, 1000
                        orderClass.send_order(self, asset, price, quantity)
                        self.AUM_available -= quantity
                        self.orderID += 1

                
                if rsi >= self.sellThreshold:  # Sell Signal
                    if self.inventory[asset]["quantity"] > -MAX_INVENTORY:
                        price, quantity = current_price, 1000 
                        orderClass.send_order(self, asset, price, -quantity)
                        self.AUM_available += quantity
                        self.orderID += 1
                
                if rsi <= self.sellThreshold/2 and self.inventory[asset]["quantity"]<0:
                    # Exit sell
                    price, quantity = current_price, abs(self.inventory[asset]["quantity"])
                    orderClass.send_order(self, asset, price, quantity)
                    self.AUM_available -= quantity
                    self.orderID += 1

                if rsi <= self.sellThreshold/2 and self.inventory[asset]["quantity"]>0:
                    # Exit buy
                    price, quantity = current_price, abs(self.inventory[asset]["quantity"])
                    orderClass.send_order(self, asset, price, -quantity)
                    self.AUM_available += quantity
                    self.orderID += 1


        # Update filled orders
        self.historical_AUM.append(self.AUM_available)
        orderClass.filled_order(self)