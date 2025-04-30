import pandas as pd
import numpy as np
from collections import deque
from backtesterClass.orderClass import orders
from backtesterClass.orderBookClass import OBData
from backtesterClass.tradingStratClass import autoTrader
from debug import logger


MAX_INVENT = 5

class momentumStrat(autoTrader):
    
    def __init__(self, name,
                short_window: int, long_window: int,
                RSI_window=1000, sellThreshold=70,
                buyThreshold=40, alpha=2):
        
        super().__init__(name)
        self.name = name

        self.RSIWindow = RSI_window 
        self.short_window = short_window
        self.long_window = long_window
        self.maxLen = max(self.RSIWindow,self.long_window)
        
        self.prices = deque(maxlen=self.maxLen) # Store prices only up to the max window necessary

        self.sellThreshold = sellThreshold
        self.buyThreshold = buyThreshold
        self.alpha = alpha/self.RSIWindow # Smoothing factor

        self.short_sum = 0
        self.long_sum = 0        
        
        self.historical_RSI = []
        self.historical_short_ma = []
        self.historical_long_ma = []

    def compute_RSI(self):

        if len(self.prices) > 1:
            delta = self.prices[-1] - self.prices[-2]
        else:
            delta = 0  # No change for the first element

        # Initialize rolling averages if not already done
        if not hasattr(self, "avg_gain"):
            self.avg_gain = 0
            self.avg_loss = 0

        # Compute current gain and loss
        gain = max(delta, 0)
        loss = max(-delta, 0)

        # Update rolling averages using exponential moving average (EMA)

        self.avg_gain = (1 - self.alpha) * self.avg_gain + self.alpha * gain
        self.avg_loss = (1 - self.alpha) * self.avg_loss + self.alpha * loss

        # Avoid division by zero and compute RSI
        if self.avg_loss == 0:
            rsi = 100  # No losses mean RSI is maxed
        else:
            rs = self.avg_gain / self.avg_loss
            rsi = 100 - (100 / (1 + rs))

    
        if len(self.prices) < self.RSIWindow:
            self.historical_RSI.append(None)
            return None
        else:
            self.historical_RSI.append(rsi)
            return rsi


    def calculate_moving_averages(self, newPrice):

        if len(self.prices) >= self.short_window:
            
            if len(self.prices) == self.short_window:
                self.short_sum += newPrice - self.prices[0]
            else:
                self.short_sum += newPrice - self.prices[-self.short_window]

            if len(self.prices) >= self.long_window:
                if self.long_sum == self.maxLen:
                    self.long_sum += newPrice - self.prices[0] 
                else:
                    self.long_sum += newPrice - self.prices[-self.long_window] 

                # Calculate the moving averages
                short_ma = self.short_sum / self.short_window
                long_ma = self.long_sum / self.long_window
    
                # Append to historical data
                self.historical_short_ma.append(short_ma)
                self.historical_long_ma.append(long_ma)

                return short_ma, long_ma
            
            else:
                self.long_sum += newPrice
                self.historical_long_ma.append(None)
                self.historical_short_ma.append(None)
                return None, None
            
        else:
            self.short_sum += newPrice
            self.long_sum += newPrice
            self.historical_long_ma.append(None)
            self.historical_short_ma.append(None)
            return None, None
    
    def strategy(self, orderClass):

        for asset in OBData.assets:

            newPrice = OBData.mid(asset)
            self.prices.append(newPrice)

            rsi = self.compute_RSI()
            short_ma, long_ma = self.calculate_moving_averages(newPrice)
            
            if rsi is None or short_ma is None or long_ma is None:
                pass
            
            else:

                buyOrderOut = [id for id, trade in self.order_out.items() 
                            if trade[orders.orderIndex["quantity"]] > 0]

                sellOrderOut = [id for id, trade in self.order_out.items() 
                            if trade[orders.orderIndex["quantity"]] < 0]
            
                # Implement Dual Moving Average Crossover Strategy
                if self.inventory["quantity"]+len(buyOrderOut) <= MAX_INVENT:  # Ensure no long position above 6
                    if (short_ma > long_ma):  # Buy Signal
                        price, quantity = newPrice, 1  # Buy one unit at slightly higher price
                        orderClass.send_order(self, price, quantity)
                        self.orderID += 1
                else:
                    buyOrderToCancel = buyOrderOut[:MAX_INVENT-(self.inventory["quantity"]+len(buyOrderOut))]

                    if len(buyOrderToCancel) > 0:
                        for id in buyOrderToCancel:
                            orderClass.cancel_order(self, id)
                
                if self.inventory["quantity"]-len(sellOrderOut) >= -MAX_INVENT:  # Ensure no short position below 6
                    if (short_ma < long_ma):  # Sell Signal
                        price, quantity = newPrice, -1  # Sell one unit at slightly lower price
                        orderClass.send_order(self, price, quantity)
                        self.orderID += 1


                else:
                    sellOrderToCancel = sellOrderOut[:MAX_INVENT-(-self.inventory["quantity"]+len(sellOrderOut))]

                    if len(sellOrderToCancel) >0:
                        for id in sellOrderToCancel:
                            orderClass.cancel_order(self, id)


        # Update filled orders
        orderClass.filled_order(self)

