import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from .orderClass import orders
from .orderBookClass import OBData
from utils.debug import logger


MAX_INVENTORY = 100000 # Modify this


class autoTrader(ABC):
    def __init__(self, strat_name):
        self.strat = strat_name
        self.AUM_available = 100000000

        self.__class__.assets = [col for col in OBData.assets]

        self.historical_trade = []
        self.historical_inventory = [[] for _ in range(len(OBData.OBData_))]

        self.historical_pnl = []
        self.historical_unrealPnL = []
        self.historical_AUM = []

        self.historical_pnl_per_asset = []
        self.historical_unrealPnL_per_asset = []

        self.PnL = 0
        self.unrealPnL = 0
        self.PnL_per_asset = np.zeros(len(OBData.assets))
        self.unrealPnL_per_asset = np.zeros(len(OBData.assets))

        self.inventory = {asset : {"price" : 0 , "quantity" : 0, "quantity ($)": 0} for asset in self.assets}
    
        self.order_out = {}
        self.orderID = 0

    def computePnL(self, orderID):

        asset = self.order_out[orderID][orders.orderIndex["asset"]]
        avgPrice = self.inventory[asset]["price"]
        PnL = 0

        if self.inventory[asset]["quantity"] > 0:
            # Compute of last filled order negative
            order = self.order_out[orderID]
            if order[orders.orderIndex["quantity"]] < 0 :
                if (
                    (np.sign(self.inventory[asset]["quantity"]+order[orders.orderIndex["quantity"]]) == np.sign(self.inventory[asset]["quantity"]))
                    or
                    (np.sign(self.inventory[asset]["quantity"]+order[orders.orderIndex["quantity"]]) == 0)

                    ):
                    # logger.info(f'order price : {order[orders.orderIndex["price"]]} - inventPrice : {avgPrice} - order qty: {order[orders.orderIndex["quantity"]]}')
                    PnL = (avgPrice-order[orders.orderIndex["price"]])*order[orders.orderIndex["quantity"]]

                    # logger.info(f'PnL Generated: {self.PnL}')
                else:
                    PnL = (avgPrice - order[orders.orderIndex["price"]]) * (self.inventory[asset]["quantity"])

                self.PnL += PnL

        elif self.inventory[asset]["quantity"] < 0:
            # Compute if last filled order positive
            order = self.order_out[orderID]
            if order[orders.orderIndex["quantity"]] > 0 :
                if (
                    (np.sign(self.inventory[asset]["quantity"]+order[orders.orderIndex["quantity"]]) == np.sign(self.inventory[asset]["quantity"]))
                    or
                    (np.sign(self.inventory[asset]["quantity"]+order[orders.orderIndex["quantity"]]) == 0)

                    ):
                    # logger.info(f'order price : {order[orders.orderIndex["price"]]} - inventPrice : {avgPrice} - order qty: {order[orders.orderIndex["quantity"]]}')
                    PnL = (avgPrice-order[orders.orderIndex["price"]])*order[orders.orderIndex["quantity"]]
                    # logger.info(f'PnL Generated: {self.PnL}')
                else:
                    PnL = (avgPrice-order[orders.orderIndex["price"]])*(self.inventory[asset]["quantity"])

                self.PnL += PnL

        self.PnL_per_asset[OBData.assetIdx[asset]-1] += PnL

        return

    def computeUnrealPnL(self):
        """
        Compute unrealized PnL - with only one asset available for now
        """
        for asset in self.assets:
            quantity = self.inventory[asset]["quantity"]
            avgPrice = self.inventory[asset]["price"]
            currentPrice = OBData.currentPrice(asset) # Compute cumulative rets for each asset

            unrealPnL = (avgPrice-currentPrice)*-quantity
            # self.unrealPnL += (avgPrice-currentPrice)*-quantity

            if len(self.historical_pnl_per_asset)>0:
                unrealPnL += self.historical_pnl_per_asset[-1][OBData.assetIdx[asset]-1]


            self.unrealPnL += unrealPnL
            self.unrealPnL_per_asset[OBData.assetIdx[asset]-1] = unrealPnL
        return

    def updateInventory(self, orderPrice: int, orderQuantity: int, asset: str):

        if self.inventory[asset]["quantity"] == 0:
            self.inventory[asset]["price"] = orderPrice
        elif np.sign(self.inventory[asset]["quantity"] + orderQuantity) != np.sign(self.inventory[asset]["quantity"]):
            self.inventory["price"] = orderPrice
        elif np.sign(self.inventory[asset]["quantity"]) != np.sign(orderQuantity):
            pass
        elif np.sign(self.inventory[asset]["quantity"]) == np.sign(orderQuantity):
            self.inventory[asset]["price"] = (self.inventory[asset]["price"]*self.inventory[asset]["quantity"]
                                                + orderPrice*orderQuantity) / (orderQuantity+self.inventory[asset]["quantity"])


        self.inventory[asset]["quantity"] += orderQuantity
        # self.inventory[asset]["quantity ($)"] += orderQuantity*orderPrice

    @abstractmethod
    def strategy():
        return