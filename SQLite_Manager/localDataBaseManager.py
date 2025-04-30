from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict

class LocalDataManager(ABC):
    def __init__(self, config:Dict):
        self.config = config
    
    @abstractmethod
    def update(self, df:pd.DataFrame) -> None:
        pass

    @abstractmethod
    def update_add(self, df:pd.DataFrame) -> None:
        pass

    @abstractmethod
    def read(self) -> pd.DataFrame:
        pass

