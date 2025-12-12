#%%
from abc import ABC, abstractmethod
import pandas as pd
#%%
class BaseRecommender(ABC):
    def __init__(self, model_name:str, user_name:list, item_name:str, date_name:str|None, seed:int):
        self.model_name = model_name
        self.model = None
        self.user_name = user_name
        self.item_name = item_name
        self.date_name = date_name
        self.sparse_feature = None
        self.dense_feature = None
        self.seed = seed
        self.unique_item = None
        self.is_trained = False
    #%%
    @abstractmethod
    def fit(self, train_data:pd.DataFrame):
        pass
    #%%
    @abstractmethod
    def recommend(self, test_data:pd.DataFrame, k:int):
        pass
