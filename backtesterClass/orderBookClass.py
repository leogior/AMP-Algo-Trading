import numpy as np
import sqlite3
import pandas as pd
import os

class OBData:
    def __init__(self, db_path):
        self.__class__.OBData_ = self.get_historicalData(db_path)
        self.__class__.OBData_ = self.__class__.OBData_[self.__class__.OBData_.Date >= "2000-01-01"].dropna(axis=1).reset_index(drop=True)
        # self.__class__.OBData_ = np.array(historicalData)
        self.__class__.step = 0
        self.__class__.assets =  [asset.split("_")[-1] for asset in self.__class__.OBData_.columns[1:]]
        self.__class__.OBIndex = {"Date": 0, **{f"{firm.split('_')[-1]}": i+1 for i, firm in enumerate(self.__class__.OBData_.columns[1:])}}
        self.__class__.Date = np.array(self.OBData_["Date"])
        self.__class__.OBData_ = np.array(self.__class__.OBData_)
        self.__class__.assetIdx = {asset: i+1 for i, asset in enumerate(self.__class__.assets)}

    
    def get_historicalData(self,db_path):

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # start_date, end_date = "01-01-2022", "12-12-2022"
        # Get all table names (tickers)
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        tickers = [t[0] for t in tables]  


        df_ini = pd.read_sql(f"SELECT Date, Close FROM {tickers[0]}", conn)
        df_ini['Date'] = pd.to_datetime(df_ini['Date'], format="%d-%m-%Y")

        for i,ticker in enumerate(tickers[1:]):
            try:
                df = pd.read_sql(f"SELECT Date, Close FROM {ticker}", conn)
                df['Date'] = pd.to_datetime(df['Date'], format="%d-%m-%Y")

                if i == 0:
                    merged_df = pd.merge_asof(df_ini.sort_values('Date'), 
                                        df.sort_values('Date'), 
                                        on='Date', 
                                        direction='backward',
                                        suffixes=(f'_{tickers[0]}', f'_{ticker}')) 
                else:
                    merged_df = pd.merge_asof(merged_df.sort_values('Date'), 
                                        df.sort_values('Date'), 
                                        on='Date', 
                                        direction='backward',
                                        suffixes=(f'_{tickers[i]}', f'_{ticker}'))
            
            except:
                pass
        return merged_df
    
    @classmethod
    def currentPrice(self, asset: str):
       price = self.OBData_[self.step][self.assetIdx[asset]]
       return price  
     
    @classmethod
    def futurePrice(self, asset: str):
       price = self.OBData_[self.step+1][self.assetIdx[asset]]
       return price   
