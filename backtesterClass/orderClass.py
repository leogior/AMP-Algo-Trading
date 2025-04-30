import pandas as pd
import numpy as np
from .orderBookClass import OBData
from debug import logger
import copy


class orders:
    def __init__(self):
        self.__class__.time = OBData.Date[OBData.step]
        try:
            self.__class__.tradeTime = OBData.Date[OBData.step + 1]
        except:
            # End of simulation
            pass

        # self.__class__.orderID = 0

        self.__class__.orderIndex = {"orderId":0, "asset": 1, "sendTime":2, "price":3, "quantity":4, "endTime":5, "status": 6}
        self.__class__.fees = {"market": 0.0002}

    @classmethod
    def send_order(self, trading_strat: object, asset: str, orderPrice: int, orderQuantity:int, fees: bool = True):
        
        currentPrice = OBData.currentPrice(asset)
        try:
            futurePrice = OBData.futurePrice(asset)
        except Exception as e:
            return

        # Send market order:
        orderPrice = futurePrice
        trading_strat.order_out[trading_strat.orderID]= [trading_strat.orderID, asset, self.time, orderPrice, orderQuantity, self.tradeTime, 0] # orderID, asset, sendTime, price, quantity, endTime, status={"out":0, "filled":1}
        trading_strat.historical_trade.append(trading_strat.order_out[trading_strat.orderID]) # Add to historical trades

        if fees:
          fee = abs(orderPrice)*abs(orderQuantity)*self.fees["market"] # Computes fees
          trading_strat.PnL -= fee

        # print(f"Market order sent : {quantity} @ {price} - fees = {fees}")


    @classmethod
    def cancel_order(self, trading_strat: object, orderID: int):
        trading_strat.order_out.pop(orderID)


    @classmethod
    def filled_order(self, trading_strat):
        orderFilledToCancel = []

        for orderID in trading_strat.order_out.keys():

            orderAsset = trading_strat.order_out[orderID][self.orderIndex["asset"]]
            orderQuantity = trading_strat.order_out[orderID][self.orderIndex["quantity"]]
            orderPrice = trading_strat.order_out[orderID][self.orderIndex["price"]]
            orderStatus = trading_strat.order_out[orderID][self.orderIndex["status"]]
            orderTradeTime =  trading_strat.order_out[orderID][self.orderIndex["endTime"]]

            if orderTradeTime == self.time:
               # Execute the order the next day
               orderStatus = 1 


            if orderStatus == 1:
                trading_strat.computePnL(orderID)
                trading_strat.updateInventory(orderPrice, orderQuantity, orderAsset)
                orderFilledToCancel.append(orderID)


        if len(orderFilledToCancel) > 0:
            for id in orderFilledToCancel:
                self.cancel_order(trading_strat, id)

        if len(trading_strat.historical_pnl)>0:
            trading_strat.PnL += trading_strat.historical_pnl[-1]


        trading_strat.historical_pnl.append(trading_strat.PnL) # add realized PnL to the historic
        trading_strat.computeUnrealPnL() # Compute unrealized pnl
        trading_strat.historical_unrealPnL.append(trading_strat.unrealPnL) # add unrealized PnL to the historic
        trading_strat.historical_pnl_per_asset.append(trading_strat.PnL_per_asset.copy()) # Add realized PnL per asset
        trading_strat.historical_unrealPnL_per_asset.append(trading_strat.unrealPnL_per_asset.copy()) # Add markout per asset

        for asset in trading_strat.assets:
          trading_strat.historical_inventory[OBData.step].append(trading_strat.inventory[asset]["quantity"]) # Add quantity

        # Reset :
        trading_strat.PnL = 0
        trading_strat.unrealPnL = 0



