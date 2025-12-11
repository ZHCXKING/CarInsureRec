#%%
from abc import ABC, abstractmethod
import pandas as pd
#%%
class BaseRecommender(ABC):
    def __init__(self, model_name, item_name, **params):
        self.model_name = model_name
        self.item_name = item_name
        self.params = params
        self.is_trained = False
    #%%
    @abstractmethod
    def fit(self, train_data: pd.DataFrame):
        pass
    #%%
    @abstractmethod
    def predict(self, test_data: pd.DataFrame):
        pass
    #%%
    def recommend(self, user_features: list):
        # 还未实现
        pass
