import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import product

from backtesterClass.orderClass import orders
from backtesterClass.orderBookClass import OBData
from backtesterClass.analysisClass import analysisClass
from backtesterClass.streamlit_dashboard import st_dashboard

from strats.basicStrat import basicStrat
from strats.movingAverageStrat import movingAverageStrat
from strats.rsiStrat import rsiStrat
from strats.momentumStrat import momentumStrat
from strats.momentumOnlineLearn import momentumOnlineLearnStrat
from strats.LTSMOnlineLearn import LTSMOnlineLearnStrat

import streamlit as st

@st.cache_data(show_spinner="Running backtest...")
def run_backtest():

    ############ Load Data ############

    db_path = r"backtesterClass/stockDB.db"
    dataClass = OBData(db_path)


    ###### Instantiate Strategies ######

    autoTrader = basicStrat("autoTrader")
    movingAverageTrader = movingAverageStrat("movingAverageTrader", short_window = 20, long_window=200)
    rsiTrader = rsiStrat("rsiTrader", window=50, buyThreshold=30, sellThreshold=70, alpha=0.05)
    momentumTrader = momentumStrat("momentumTrader", short_window = 20, long_window=200, RSI_window=50, sellThreshold=70,buyThreshold=30, alpha=0.05)
    # momentumOnlineTrader = momentumOnlineLearnStrat("momentumOnlineTrader", short_window = 20, long_window=200, RSI_window=50, sellThreshold=70,buyThreshold=30, alpha=2,trainLen = 100000, forecast=5)
    # LTMSOnlineTrader = LTSMOnlineLearnStrat("LTSMOnlineTrader", short_window = 20, long_window=200, RSI_window=50, sellThreshold=70,buyThreshold=30, alpha=2,trainLen = 100000, forecast=5)


    ###### Run Simulation ######

    for _ in tqdm(range(len(dataClass.OBData_))):
        orderClass = orders()
        # autoTrader.strategy(orderClass)
        movingAverageTrader.strategy(orderClass)
        rsiTrader.strategy(orderClass)
        momentumTrader.strategy(orderClass)
        # momentumOnlineTrader.strategy(orderClass)
        OBData.step +=1
        
    return movingAverageTrader, rsiTrader, momentumTrader

####################################

movingAverageTrader, rsiTrader, momentumTrader = run_backtest()
dashboard = st_dashboard(movingAverageTrader)

dashboard.dashboard()