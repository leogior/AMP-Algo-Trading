import pandas as pd
import numpy as np
from backtesterClass.orderClass import orders
from backtesterClass.orderBookClass import OBData
from backtesterClass.tradingStratClass import autoTrader
from debug import logger



class basicStrat(autoTrader):

    def __init__(self, name):
        super().__init__(name)

    def usefull_function(self):
    # You can add function here if you want to use them in the strategy function #
      return

    def strategy(self, orderClass):

        MAX_INVENT = 100000 # Modify this


        for asset in OBData.assets:
            
            asset_price = OBData.currentPrice(asset)
            targetPrice = asset_price + np.random.randint(-1,2)

            if asset_price<targetPrice: # Modify this signal
              # Best ask below Buy Target -> I buy
              if  self.inventory[asset]["quantity"] <= MAX_INVENT and self.AUM_available > 0:
                  price, quantity = targetPrice, 1000 # Modify this
                  orderClass.send_order(self, asset, price, quantity)
                  self.AUM_available -= quantity
                  self.orderID +=1
              else:
                  buyOrderToCancel = [id for id, trade in self.order_out.items()
                                      if trade[orders.orderIndex["quantity"]] > 0]
                  if len(buyOrderToCancel) > 0:
                      for id in buyOrderToCancel:
                          orderClass.cancel_order(self, id)

            elif asset_price>=targetPrice: # Modify this signal
                # Mid above Sell Target -> I sell
                if self.inventory[asset]["quantity"] >= -MAX_INVENT:
                  price, quantity = targetPrice, 1000 # Modify this
                  orderClass.send_order(self, asset, price, -quantity)
                  self.AUM_available += quantity
                  self.orderID +=1
                else:
                  sellOrderToCancel = [id for id, trade in self.order_out.items()
                                      if trade[orders.orderIndex["quantity"]] < 0]
                  if len(sellOrderToCancel) >0:
                      for id in sellOrderToCancel:
                          orderClass.cancel_order(self, id)


        orderClass.filled_order(self)


