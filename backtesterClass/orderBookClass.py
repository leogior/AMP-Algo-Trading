import numpy as np
import sqlite3
import pandas as pd
import os

class OBData:
    def __init__(self, db_path):
        self.__class__.OBData_ = self.get_historicalData(db_path)
        self.__class__.step = 0
        self.__class__.assets =  [asset.split("_")[0] for asset in self.__class__.OBData_.columns[1:]]
        self.__class__.OBIndex = {"Date": 0, **{f"{firm.split('_')[0]}": i+1 for i, firm in enumerate(self.__class__.OBData_.columns[1:])}}
        self.__class__.Date = np.array(self.OBData_["Date"])
        self.__class__.OBData_ = np.array(self.__class__.OBData_)
        self.__class__.assetIdx = {asset: i+1 for i, asset in enumerate(self.__class__.assets)}

    
    def get_historicalData(self,db_path):

        df = pd.read_csv(db_path, header=[0,1])  # Use two header rows
        df = df.iloc[2:].copy()
        df[('Ticker', 'Price')] = pd.to_datetime(df[('Ticker', 'Price')])
        df.columns = ['_'.join([str(i) for i in col if str(i) != 'nan']) for col in df.columns]
        close_cols = [col for col in df.columns if col.endswith('_Close') or col == 'Ticker_Price']
        df_close = df[close_cols]
        df_close = df_close.reset_index(drop=True)
        df_close.rename(columns={df_close.columns[0]: 'Date'}, inplace=True)

        return df_close

    
    @classmethod
    def currentPrice(self, asset: str):
       price = self.OBData_[self.step][self.assetIdx[asset]]
       return price  
     
    @classmethod
    def futurePrice(self, asset: str):
       price = self.OBData_[self.step+1][self.assetIdx[asset]]
       return price   
