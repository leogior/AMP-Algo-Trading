import pandas as pd
import numpy as np
from collections import deque
from backtesterClass.orderClass import orders
from backtesterClass.orderBookClass import OBData
from backtesterClass.tradingStratClass import autoTrader

from river import compose
from river import preprocessing
from river import evaluate
from river import metrics
from river import linear_model
from river import feature_extraction
from river import optim
from river import feature_selection



from debug import logger


MAX_INVENT = 5

class momentumOnlineLearnStrat(autoTrader):
    
    def __init__(self, name,
                short_window: int, long_window: int,
                RSI_window=1000, sellThreshold=70,
                buyThreshold=30, alpha=2,
                trainLen = 20000, forecast=5):
        
        super().__init__(name)
        self.name = name

        self.RSIWindow = RSI_window 
        self.short_window = short_window
        self.long_window = long_window
        self.maxLen = max(self.RSIWindow,self.long_window)
        
        self.forecast = forecast
        self.prices = deque(maxlen=self.maxLen) # Store prices only up to the max window necessary
        self.X_list = deque(maxlen= self.forecast) # Store features to train the model in streaming
        self.y_list = deque(maxlen= self.forecast) # Store features to train the model in streaming
        self.y_pred_list = deque(maxlen= self.forecast) # Store features to train the model in streaming



        self.sellThreshold = sellThreshold
        self.buyThreshold = buyThreshold
        self.alpha = alpha/self.RSIWindow # Smoothing factor

        self.short_sum = 0
        self.long_sum = 0        
        
        self.historical_RSI = []
        self.historical_short_ma = []
        self.historical_long_ma = []

        self.trainLen = trainLen
        self.midBuffer = 0

        self.metric = metrics.ROCAUC()

        self.cumRets = 0

        self.prediction = []
        
        # self.model = compose.Pipeline(
        #     preprocessing.StandardScaler(),
        #     feature_extraction.PolynomialExtender(degree=2),
        #     linear_model.LogisticRegression(optimizer=optim.SGD(.1), l2=0.1), 
        #         )

        self.model = compose.Pipeline(
            preprocessing.StandardScaler(),
            feature_selection.VarianceThreshold(),
            linear_model.LogisticRegression(), 
                )
              
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

    def compute_lag_rets(self, lag:int):
        return self.prices[-1]/self.prices[-lag] - 1

    def compute_ema_cum_rets(self, alpha, cumsum_rets, rets):
        return cumsum_rets*(1-alpha) + rets*alpha
            
    def strategy(self, orderClass):

        mid = OBData.mid()
        self.prices.append(mid)
        rsi = self.compute_RSI()
        long_ma, short_ma = self.calculate_moving_averages(mid)

        if mid - self.midBuffer == 0:
            # No mid update -> move forward
            pass
        
        else:
            # y = (mid/self.midBuffer)//1
            self.midBuffer = mid
            if len(self.prices) >=2:
                self.cumRets = self.compute_ema_cum_rets(self.alpha, self.cumRets, self.compute_lag_rets(-2))

            if rsi is not None and long_ma is not None and short_ma is not None:
                        
                X = {"rsi" : rsi,
                    "long_ma": long_ma,
                    "short_ma": short_ma,
                    "lag_ret_5": self.compute_lag_rets(-5),
                    "lag_ret_10": self.compute_lag_rets(-10),
                    "lag_ret_50": self.compute_lag_rets(-50),
                    "lag_ret_100": self.compute_lag_rets(-100),
                    "cumulative returns" : self.cumRets}
                
                self.X_list.append(X)
                
                if len(self.X_list) == self.forecast:
                    y = (mid/self.prices[-self.forecast])//1
                    self.y_list.append(y)
                    y_pred = self.model.predict_proba_one(X)
                    self.y_pred_list.append(y_pred)
                
                if OBData.step > self.trainLen :
                    

                    buyOrderOut = [id for id, trade in self.order_out.items() 
                                if trade[orders.orderIndex["quantity"]] > 0]

                    sellOrderOut = [id for id, trade in self.order_out.items() 
                                if trade[orders.orderIndex["quantity"]] < 0]
                
                    if self.inventory["quantity"]+len(buyOrderOut) < MAX_INVENT:  # Ensure no long position above 5           
                        if y_pred[True] > 0.95:
                            y_hat = 1
                            price, quantity = orderClass.bids, 1  # Buy one unit - limit order
                            orderClass.send_order(self, price, quantity)
                            self.orderID += 1           

                    else:
                        buyOrderToCancel = buyOrderOut[:int(MAX_INVENT-(self.inventory["quantity"]+len(buyOrderOut)))]

                        if len(buyOrderToCancel) > 0:
                            for id in buyOrderToCancel:
                                orderClass.cancel_order(self, id)
                    
                    if self.inventory["quantity"]-len(sellOrderOut) > -MAX_INVENT:  # Ensure no short position below 5
                        if  y_pred[False] > 0.95:
                            y_hat = 0
                            price, quantity = orderClass.asks, -1  # Sell one unit - limit order
                            orderClass.send_order(self, price, quantity)
                            self.orderID += 1

                    else:
                        
                        sellOrderToCancel = sellOrderOut[:int(MAX_INVENT-(-self.inventory["quantity"]+len(sellOrderOut)))]

                        if len(sellOrderToCancel) > 0:
                            for id in sellOrderToCancel:
                                orderClass.cancel_order(self, id)
                    
                    # if (self.inventory["quantity"] < 0) and (long_ma < short_ma):
                    #     price, quantity = 10000000, -self.inventory["quantity"] # Stop barrier
                    #     orderClass.send_order(self, price, quantity)
                    #     self.orderID += 1
                        
                    #     buyOrderToCancel = buyOrderOut

                    #     if len(buyOrderToCancel) > 0:
                    #         for id in buyOrderToCancel:
                    #             orderClass.cancel_order(self, id)                       

                    # if (self.inventory["quantity"] > 0) and (long_ma  short_ma):
                    #     price, quantity = 0, -self.inventory["quantity"] # Stop barrier
                    #     orderClass.send_order(self, price, quantity)
                    #     self.orderID += 1
                        
                    #     sellOrderToCancel = sellOrderOut

                    #     if len(sellOrderToCancel) > 0:
                    #         for id in sellOrderToCancel:
                    #             orderClass.cancel_order(self, id)   

                if len(self.X_list) == self.forecast:
                
                    self.model.learn_one(self.X_list[0], self.y_list[-1])
                    self.metric.update(self.y_list[-1], self.y_pred_list[0])
                    self.prediction.append([self.y_pred_list[0],self.y_list[-1]])


                # if OBData.step % 200000 == 0:   
                # #     # logger.info(f"x:{self.X_list[0]}, y:{self.y_list[-1]}, y_pred:{self.y_pred_list[0]}")
                #     print(f"short_window: {self.short_window}, long_window: {self.long_window}, RSI_window: {self.RSIWindow}, sellThreshold: {self.sellThreshold}, buyThreshold:{self.buyThreshold}, forecast:{self.forecast}")
                #     print(self.model.debug_one(X))
                #     print(self.metric)
    
        orderClass.filled_order(self)