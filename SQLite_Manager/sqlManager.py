import logging 
from pandas.io import sql
from sqlalchemy import create_engine
import pandas as pd
from typing import Dict
from .localDataBaseManager import LocalDataManager


class SqlAlchemyDataBaseManager(LocalDataManager):
    """
    Create and connect to a local file db with SQLite. Allows for it to be update and read.
    """

    @staticmethod
    def get_logger_name():
        return "sqlalchemy_database_manager"
    
    def get_logger(self):
        return logging.getLogger(SqlAlchemyDataBaseManager.get_logger_name())
    
    def __init__(self, db_path, config = {"if_exits": "replace"}):
        self.config = config
        self.db_url = f"sqlite:///{db_path}"
        self.if_exits = self.config["if_exits"]
        self.engine = None
        self.initialise()
    
    def initialise(self):
        self.engine = create_engine(self.db_url)
        self.conn = self.engine.connect()
    
    def update(self, table_name:str, df:pd.DataFrame) -> None:
        """
        Update existing table with the df, of non existent creates. If exists replaces.
        """
        if self.engine is None:
            self.initialise()
        if not (df is None and df.empty):
            df.to_sql(table_name, con=self.conn.connection, if_exists="replace")

    def update_add(self, table_name:str, df:pd.DataFrame, index=True) -> None:
        if self.engine is None:
            self.initialise()
        
        self.get_logger().info(f"Add {df.shape[0]} entires in {table_name} ; {self.engine.url}")
        
        df.to_sql(table_name, con=self.engine, if_exists="append", index=index)

    def read(self, table_name:str) -> pd.DataFrame:
        df = pd.read_sql(sql=table_name, con=self.conn.connection)
        return df
    
    def __repr__(self):
        """
        Shows db local path
        """
        return self.db_url
    