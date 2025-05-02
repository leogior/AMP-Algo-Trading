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


MAX_INVENTORY = 10000

class momentumOnlineLearnStrat(autoTrader):
    
    def __init__(self, name,
                short_window: int, long_window: int,
                RSI_window=50, sellThreshold=70,
                buyThreshold=30, alpha=0.02,
                trainLen = 20000, forecast=1):
        
        super().__init__(name)
        self.name = name
        self.asset_list = OBData.assets
        self.asset_count = len(self.asset_list)

        self.windowRSI = RSI_window 
        self.short_window = short_window
        self.long_window = long_window
        self.maxLen = max(self.windowRSI,self.long_window)


        self.forecast = forecast
        self.prices = [deque(maxlen=self.maxLen) for _ in range(self.asset_count)] # Store prices only up to the max window necessary
        self.X_list = [deque(maxlen= self.forecast) for _ in range(self.asset_count)]  # Store features to train the model in streaming
        self.y_list = [deque(maxlen= self.forecast) for _ in range(self.asset_count)] # Store features to train the model in streaming
        self.y_pred_list = [deque(maxlen= self.forecast) for _ in range(self.asset_count)] # Store features to train the model in streaming



        self.sellThreshold = sellThreshold
        self.buyThreshold = buyThreshold
        self.alpha = alpha # Smoothing factor

        self.short_sums = np.zeros(self.asset_count)
        self.long_sums = np.zeros(self.asset_count)     
        
        self.historical_RSI = [[] for _ in range(self.asset_count)]
        self.historical_short_ma = [[] for _ in range(self.asset_count)]
        self.historical_long_ma = [[] for _ in range(self.asset_count)]

        self.trainLen = trainLen
        self.midBuffer = np.zeros(self.asset_count)   

        self.metric = metrics.ROCAUC()

        self.cumRets = np.zeros(self.asset_count)

        self.prediction = [[] for _ in range(self.asset_count)]
        
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
            return None, None

        if len(price_queue) < self.long_window:
            self.short_sums[idx] += new_price - (price_queue[-self.short_window] if len(price_queue) >= self.short_window else 0)
            self.long_sums[idx] += new_price
            self.historical_short_ma[idx].append(None)
            self.historical_long_ma[idx].append(None)
            return None, None

        # Fast rolling sums for short and long window
        self.short_sums[idx] += new_price - price_queue[-self.short_window]
        self.long_sums[idx] += new_price - price_queue[0]

        short_ma = self.short_sums[idx] / self.short_window
        long_ma = self.long_sums[idx] / self.long_window

        self.historical_short_ma[idx].append(short_ma)
        self.historical_long_ma[idx].append(long_ma)

        return short_ma, long_ma

    def compute_lag_rets(self, asset, lag:int):
        idx = OBData.assetIdx[asset]-1
        return self.prices[idx][-1]/self.prices[idx][-lag] - 1

    def compute_ema_cum_rets(self, asset, alpha, cumsum_rets, rets):
        idx = OBData.assetIdx[asset]-1
        return cumsum_rets[idx]*(1-alpha) + rets*alpha
            
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