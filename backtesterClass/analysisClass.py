import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from datetime import datetime 
from debug import logger

from .orderBookClass import OBData

from strats.basicStrat import basicStrat
from strats.movingAverageStrat import movingAverageStrat
from strats.rsiStrat import rsiStrat
from strats.momentumStrat import momentumStrat
from strats.momentumOnlineLearn import momentumOnlineLearnStrat

from SQLite_Manager.sqlManager import SqlAlchemyDataBaseManager


class analysisClass:
    def __init__(self, autoTrader, path : str = None, dashboardName : str = None, dbName : str = None):
        self.autoTrader = autoTrader
        self.path = path
        self.dashboardName = dashboardName
        self.dbName = dbName

        self.data = pd.DataFrame(OBData.OBData_, columns = OBData.OBIndex.keys())

        self.pnl = pd.DataFrame({"Pnl":self.autoTrader.historical_pnl, "unrealPnl":self.autoTrader.historical_unrealPnL})
        self.pnl_per_asset = pd.DataFrame(self.autoTrader.historical_unrealPnL_per_asset, columns=OBData.assets)

        self.inventory = pd.DataFrame(np.array(self.autoTrader.historical_inventory), columns=OBData.assets)
        self.inventory["Date"] = self.data["Date"]
        self.pnl_per_asset["Date"] = self.data["Date"]

        self.historicalTrades = pd.DataFrame(self.autoTrader.historical_trade, columns=["tradeIndex", "asset", "sendTime", "price","volume", "tradeTime", "Status"])
        
    
    def create_dashboard(self, asset: str = False, save = True, show = False, streamlit=False):

        if asset :
            asset_idx = OBData.assetIdx[asset]-1
            self.dashboardName += f"_{asset}"

        if isinstance(self.autoTrader, rsiStrat) or isinstance(self.autoTrader, momentumStrat) or isinstance(self.autoTrader, momentumOnlineLearnStrat):
            # Intermediary dashboard to display rsi chart
            fig = make_subplots(specs=[[{"secondary_y": True}], [{}], [{}]], 
                                rows=3, cols=1,
                                shared_xaxes=True,
                                row_heights=[0.5, 0.2, 0.3],
                                vertical_spacing=0.02)
        else:
            
            fig = make_subplots(specs=[[{"secondary_y": True}], [{}]], 
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.02)
        
        if asset :
            fig.update_layout(title_text=f"{self.autoTrader.strat} Execution Dashboard {asset}")
        else:
            fig.update_layout(title_text=f"{self.autoTrader.strat} Execution Dashboard")
        
        fig.add_trace(go.Scatter(
                                    x=self.data.Date,
                                    y=self.data[asset],
                                    name=f"{asset} price",
                                    marker=dict()
                                        )
                                ,row=1, col=1, secondary_y=False)
    
        # fig.add_trace(go.Scatter(
        #                 x=self.data.Date,
        #                 y=self.pnl.Pnl,
        #                 name="pnl",
        #                 marker=dict(color='red')
        #                     )
        #             ,row=1, col=1, secondary_y=True)

        fig.add_trace(go.Scatter(
                        x=self.data.Date,
                        y=self.pnl_per_asset[asset],
                        name=f"Markout PnL {asset}",
                        marker=dict(color='deepskyblue')
                            )
                    ,row=1, col=1, secondary_y=True)
        
        # Buy orders
        fig.add_trace(go.Scatter(
                            x=self.historicalTrades[(self.historicalTrades["volume"]>0) & (self.historicalTrades["asset"] == asset)]["tradeTime"],
                            y=self.historicalTrades[(self.historicalTrades["volume"]>0) & (self.historicalTrades["asset"] == asset)]["price"],
                            name=f"buy orders",
                            mode='markers',
                            marker=dict(color="green", size=10, symbol='triangle-up')
                                )
                        ,row=1, col=1, secondary_y=False)

        # Sell orders
        fig.add_trace(go.Scatter(
                            x=self.historicalTrades[(self.historicalTrades["volume"]<0) & (self.historicalTrades["asset"] == asset)]["tradeTime"],
                            y=self.historicalTrades[(self.historicalTrades["volume"]<0) & (self.historicalTrades["asset"] == asset)]["price"],
                            name=f"sell orders",
                            mode='markers',
                            marker=dict(color="red", size=10, symbol='triangle-down')
                                )
                        ,row=1, col=1, secondary_y=False)

        if isinstance(self.autoTrader, rsiStrat) or isinstance(self.autoTrader, momentumStrat) or isinstance(self.autoTrader, momentumOnlineLearnStrat):

            fig.add_trace(go.Scatter(
                                x=self.inventory.Date,
                                y=self.inventory[asset],
                                name=f"{asset}_inventory",
                                    )
                            ,row=3, col=1)

        else:
            # Inventory
            fig.add_trace(go.Scatter(
                                x=self.inventory.Date,
                                y=self.inventory[asset],
                                name=f"{asset}_inventory",
                                    )
                            ,row=2, col=1)        

        if isinstance(self.autoTrader, movingAverageStrat) or isinstance(self.autoTrader, momentumStrat) or isinstance(self.autoTrader, momentumOnlineLearnStrat):

            fig.add_trace(go.Scatter(
                                x=self.data.Date,
                                y=self.autoTrader.historical_long_ma[asset_idx],
                                name="long_ma",
                                marker=dict(color='brown')
                                    )
                            ,row=1, col=1, secondary_y=False)

            fig.add_trace(go.Scatter(
                                x=self.data.Date,
                                y=self.autoTrader.historical_short_ma[asset_idx],
                                name="short_ma",
                                marker=dict(color='seagreen')
                                    )
                            ,row=1, col=1, secondary_y=False)
            
        if isinstance(self.autoTrader, rsiStrat) or isinstance(self.autoTrader, momentumStrat) or isinstance(self.autoTrader, momentumOnlineLearnStrat):

            fig.add_trace(go.Scatter(
                                x=self.data.Date,
                                y=self.autoTrader.historical_RSI[asset_idx],
                                name="rsi",
                                marker=dict(color='darkgray')
                                    )
                            ,row=2, col=1, secondary_y=False)

            fig.add_shape(
                type="line",
                x0=min(self.data.Date),  # Starting x-coordinate
                x1=max(self.data.Date),  # Ending x-coordinate
                y0=self.autoTrader.sellThreshold, # y-coordinate for the line
                y1=self.autoTrader.sellThreshold,
                line=dict(color="red", width=2, dash="dash"),
                xref="x2",  # Refers to the x-axis of the second row
                yref="y3"   # Refers to the y-axis of the second row
            )

            fig.add_shape(
                type="line",
                x0=min(self.data.Date),
                x1=max(self.data.Date),
                y0=self.autoTrader.buyThreshold,
                y1=self.autoTrader.buyThreshold,
                line=dict(color="green", width=2, dash="dash"),
                xref="x2",
                yref="y3"
            )            

            fig.update_layout(
                legend_orientation="h",
                xaxis3=dict(
                    rangeslider=dict(
                        visible=True,
                        bgcolor="darkgray",  # Set the background color of the slider
                        thickness=0.03  # Set the thickness of the range slider
                    ),
                    showgrid=True,
                ),

                legend=dict(
                    orientation="v",  # Vertical legend
                    xanchor="right",  # Anchor on right based on the x value
                    x=1.1,
                    yanchor="top"  # Anchor it to the top of the legend box
                ))
        
        if not isinstance(self.autoTrader, rsiStrat) and not isinstance(self.autoTrader, momentumStrat) and not isinstance(self.autoTrader, momentumOnlineLearnStrat) :

            fig.update_layout(
                legend_orientation="h",
                xaxis2=dict(
                    rangeslider=dict(
                        visible=True,
                        bgcolor="darkgray",  # Set the background color of the slider
                        thickness=0.03  # Set the thickness of the range slider
                    ),
                    showgrid=True,
                ),

                legend=dict(
                    orientation="v",  # Vertical legend
                    xanchor="right",  # Anchor on right based on the x value
                    x=1.1,
                    yanchor="top"  # Anchor it to the top of the legend box
                ))

        fig.update_layout(
            width=1500,  
            height=800, 
        )

        

        if show:

            fig.update_layout(
                width=700,  
                height=400, 
            )
            fig.show()



        if save:
            if self.path == None:
                fig.write_html(f"{self.dashboardName}.html")
            else:
                fig.write_html(f"{self.path}/{self.dashboardName}.html")
        
        if streamlit:
            return fig
        
        return
    
    @classmethod
    def streamlitDashboard(self, fig):
        st.plotly_chart(fig, use_container_width=True)
        return 


    def dataBase(self):

        if self.path == None:
            db = SqlAlchemyDataBaseManager(f"{self.dashboardName}.db")
        else:
            db = SqlAlchemyDataBaseManager(f"{self.path}/{self.dbName}.db")

        historicalInventory = self.inventory
        historicalPnL = self.pnl["Pnl"]
        historicalUnrealizedPnL = self.pnl["unrealPnl"]
        df_mid = self.data

        db.update("historicalPrices",df_mid)
        db.update("historicalTrades",self.historicalTrades)
        db.update("historicalInventory",historicalInventory)
        db.update("historicalPnL",historicalPnL)
        db.update("historicalUnrealizedPnL",historicalUnrealizedPnL)

        if isinstance(self.autoTrader, rsiStrat) or isinstance(self.autoTrader, momentumStrat):
            historicalRSI = pd.DataFrame({"Date":self.data.Date})
            for asset in OBData.assets:
                assetIdx = OBData.assetIdx[asset]-1
                historicalRSI[f"RSI_{asset}"] = self.autoTrader.historical_RSI[assetIdx]
            
            db.update("historicalRSI", historicalRSI)
        
        elif isinstance(self.autoTrader, movingAverageStrat) or isinstance(self.autoTrader, momentumStrat):
            historicalMA = pd.DataFrame({"Date":self.data.Date})

            for asset in OBData.assets:
                assetIdx = OBData.assetIdx[asset]-1
                historicalMA[f"long_ma_{asset}"] = self.autoTrader.historical_long_ma[assetIdx]
                historicalMA[f"short_ma_{asset}"] = self.autoTrader.historical_short_ma[assetIdx]

            db.update("historicalMA", historicalMA)
