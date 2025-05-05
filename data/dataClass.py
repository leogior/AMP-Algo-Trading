import os
import pandas as pd

class DataClass:
    def __init__(self, directory, tickers_csv):
        self.directory = directory
        self.tickers_csv = tickers_csv
        self.ticker_list = self._load_ticker_list()
        self.data = None

    def _load_ticker_list(self):
        df = pd.read_csv(self.tickers_csv)
        # Use first column, uppercase, strip spaces
        return set(df.iloc[:, 0].astype(str).str.upper().str.strip())

    def load_adjusted_close(self):
        dfs = []
        for file_name in os.listdir(self.directory):
            if file_name.endswith('.csv'):
                ticker = os.path.splitext(file_name)[0].upper()
                if ticker in self.ticker_list:
                    file_path = os.path.join(self.directory, file_name)
                    try:
                        df = pd.read_csv(file_path)
                        # Find Adjusted Close column
                        adj_close_cols = [col for col in df.columns if col.strip().lower() == 'adjusted close']
                        if not adj_close_cols:
                            continue
                        adj_close_col = adj_close_cols[0]
                        df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
                        df = df.dropna(subset=['Date'])
                        df = df[['Date', adj_close_col]].rename(columns={adj_close_col: ticker})
                        dfs.append(df)
                    except Exception as e:
                        print(f"Error processing {file_name}: {e}")
        if not dfs:
            self.data = pd.DataFrame()
            return self.data
        # Merge all on 'Date'
        merged = dfs[0]
        for df in dfs[1:]:
            merged = pd.merge(merged, df, on='Date', how='outer')
        merged = merged.sort_values('Date').reset_index(drop=True)
        self.data = merged
        return self.data
