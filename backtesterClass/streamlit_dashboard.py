import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
st.set_page_config(layout="wide")

from .orderBookClass import OBData


class st_dashboard:

    def __init__(self, autoTrader):
        self.autoTrader = autoTrader

        self.data = pd.DataFrame(OBData.OBData_, columns = OBData.OBIndex.keys())

        self.pnl = pd.DataFrame({"Pnl":self.autoTrader.historical_pnl, "unrealPnl":self.autoTrader.historical_unrealPnL})
        self.unrealpnl_per_asset = pd.DataFrame(self.autoTrader.historical_unrealPnL_per_asset, columns=OBData.assets)
        self.pnl_per_asset = pd.DataFrame(self.autoTrader.historical_pnl_per_asset, columns=OBData.assets)

        self.inventory = pd.DataFrame(np.array(self.autoTrader.historical_inventory), columns=OBData.assets)
        self.inventory["Date"] = self.data["Date"]
        self.pnl_per_asset["Date"] = self.data["Date"]
        self.unrealpnl_per_asset["Date"] = self.data["Date"]

        self.historicalTrades = pd.DataFrame(self.autoTrader.historical_trade, columns=["tradeIndex", "asset", "sendTime", "price","volume", "tradeTime", "Status"])
        

    def dashboard(self):

        st.subheader("Execution Dashboard")

        # Streamlit multi-select to select multiple assets
        selected_asset = st.selectbox('Select Assets', OBData.assets)
        selected_strats = st.multiselect('Select Strategies', ["Momentum", "RSI", "Value"])

        with st.container():
            fig = make_subplots(
                rows=2, cols=2,  # Two rows, two columns
                shared_xaxes=True,  # Share x-axis between columns 1 and 2
                vertical_spacing=0.05,  # Increased space between the two charts (higher than before)
                horizontal_spacing=0.15,  # Increased space between the columns (higher than before)
                subplot_titles=("Execution", "Markout PnL per Strategy", "Inventory", "Realized PnL per Strategy"),
                row_heights=[0.5, 0.5],  # Adjust row height proportions to allocate more space for price chart
                specs=[[{"secondary_y": True}, {"secondary_y": True}], 
                    [{"secondary_y": True}, {"secondary_y": True}]]  # Secondary axis for the PnL plot
                        )
            
            fig.add_trace(go.Scatter(
                                    x=self.data.Date,
                                    y=self.data[selected_asset],
                                    name=f"{selected_asset} price",
                                    marker=dict(color="orange")
                                        )
                                    ,row=1, col=1, secondary_y=False)
            
            # Buy orders
            fig.add_trace(go.Scatter(
                                x=self.historicalTrades[(self.historicalTrades["volume"]>0) & (self.historicalTrades["asset"] == selected_asset)]["tradeTime"],
                                y=self.historicalTrades[(self.historicalTrades["volume"]>0) & (self.historicalTrades["asset"] == selected_asset)]["price"],
                                name=f"buy orders",
                                mode='markers',
                                marker=dict(color="green", size=10, symbol='triangle-up')
                                    )
                            ,row=1, col=1, secondary_y=False)

            # Sell orders
            fig.add_trace(go.Scatter(
                                x=self.historicalTrades[(self.historicalTrades["volume"]<0) & (self.historicalTrades["asset"] == selected_asset)]["tradeTime"],
                                y=self.historicalTrades[(self.historicalTrades["volume"]<0) & (self.historicalTrades["asset"] == selected_asset)]["price"],
                                name=f"sell orders",
                                mode='markers',
                                marker=dict(color="red", size=10, symbol='triangle-down')
                                    )
                            ,row=1, col=1, secondary_y=False)
            

            # Inventory 
            fig.add_trace(go.Scatter(
                                x=self.inventory.Date,
                                y=self.inventory[selected_asset],
                                name=f"{selected_asset}_inventory",
                                    )
                            ,row=2, col=1)
            
            # Markout PnL
            fig.add_trace(go.Scatter(
                            x=self.data.Date,
                            y=self.unrealpnl_per_asset[selected_asset],
                            name=f"Markout PnL {selected_asset}",
                            marker=dict(color='deepskyblue')
                                )
                        ,row=1, col=2, secondary_y=False)

            # Realized PnL
            fig.add_trace(go.Scatter(
                            x=self.data.Date,
                            y=self.pnl_per_asset[selected_asset],
                            name=f"Realized PnL {selected_asset}",
                            marker=dict(color='red')
                                )
                        ,row=2, col=2, secondary_y=False)


            fig.update_xaxes(matches='x')


            # Update layout for the chart with multiple y-axes
            fig.update_layout(
                xaxis3_title='Timestamp',
                xaxis4_title='Timestamp',
                template='plotly_dark',
                showlegend=True,
                height=800,  # Adjust height for clarity
                margin=dict(t=80, b=0, l=40, r=40),  # Adjust margins for better spacing
                
                # Adjusting axis tick font size
                xaxis3=dict(
                    tickfont=dict(size=20),  # Larger font for x-axis values
                ),
                xaxis4=dict(
                    tickfont=dict(size=20), 
                ),
                yaxis=dict(
                    tickfont=dict(size=20), # Larger font for y-axis values
                ),
                yaxis2=dict(
                    tickfont=dict(size=20), 
                ),
                yaxis3=dict(
                    tickfont=dict(size=20), 
                ),
                yaxis4=dict(
                    tickfont=dict(size=20), 
                ),
                yaxis5=dict(
                    tickfont=dict(size=20), 
                ),
                yaxis6=dict(
                    tickfont=dict(size=20), 
                ),
                yaxis7=dict(
                    tickfont=dict(size=20), 
                ),
                yaxis8=dict(
                    tickfont=dict(size=20), 
                )

            )



            fig.update_yaxes(
                title_text="Price ($)", row=1, col=1,
                showgrid=True,  # Keep gridlines visible, or set to False to hide them
                showline=True,  # Adds a line at the edge of the y-axis
                linewidth=1,  # Line width for the y-axis
                ticks="outside",  # Ticks outside the plot for better separation
                ticklen=5,  # Length of the ticks
                tickwidth=1,  # Thickness of the ticks
                tickformat=".2f",  # Use 2 decimal places for PnL values
                tickangle=0  # Set tick angle to 0 to avoid angle issues with long tick values
            )

            fig.update_yaxes(
                title_text="Realized PnL ($)", 
                row=2, col=2,
                showgrid=True,   
                showline=True,   
                linewidth=1,
                ticks="outside", 
                ticklen=5,
                tickwidth=1, 
                tickformat=".2f", 
                tickangle=0 
            )

            fig.update_yaxes(
                title_text="Inventory ($)", 
                row=2, col=1,
                showgrid=True,  
                showline=True,  
                linewidth=1,
                ticks="outside",
                ticklen=5,
                tickwidth=1, 
                tickformat=".2f",
                tickangle=0  
            )

            fig.update_yaxes(
                title_text="Markout PnL ($)", 
                row=1, col=2,
                showgrid=True, 
                showline=True, 
                linewidth=1, 
                ticks="outside", 
                ticklen=5,
                tickwidth=1,
                tickformat=".2f",
                tickangle=0
            )

            st.plotly_chart(fig, use_container_width=True)


        with st.container():
            
            fig.update_layout(title_text=f"{self.autoTrader.strat} Overview")

            fig1 = make_subplots(
                rows=2, cols=1,  # Two rows, two columns
                shared_xaxes=True,  # Share x-axis between columns 1 and 2
                vertical_spacing=0.05,  # Increased space between the two charts (higher than before)
                horizontal_spacing=0.15,  # Increased space between the columns (higher than before)
                subplot_titles=("PnL", "AUM available"),
                # row_heights=[0.5, 0.5],  # Adjust row height proportions to allocate more space for price chart
                specs=[[{"secondary_y": True}], [{"secondary_y": True}]]
                        )
            
            fig1.add_trace(go.Scatter(
                            x=self.data.Date,
                            y=self.pnl.Pnl,
                            name="Total PnL",
                            marker=dict(color='red')
                                )
                        ,row=1, col=1, secondary_y=False)
            
            fig1.add_trace(go.Scatter(
                            x=self.data.Date,
                            y=self.pnl.unrealPnl,
                            name="Total Unrealized PnL",
                            marker=dict(color='skyblue')
                                )
                        ,row=2, col=1, secondary_y=False)

            fig1.update_yaxes(
                title_text="Realized PnL ($)", 
                row=1, col=1,
                showgrid=True,   
                showline=True,   
                linewidth=1,
                ticks="outside", 
                ticklen=5,
                tickwidth=1, 
                tickformat=".2f", 
                tickangle=0 
            )

            fig1.update_yaxes(
                title_text="Markout PnL ($)", 
                row=2, col=1,
                showgrid=False, 
                showline=False,
                secondary_y=False,
                linewidth=1, 
                ticks="outside", 
                ticklen=5,
                tickwidth=1,
                tickformat=".2f",
                tickangle=0
            )
        
            st.plotly_chart(fig1, use_container_width=True)        