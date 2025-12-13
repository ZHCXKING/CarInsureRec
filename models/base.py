#%%
from abc import ABC, abstractmethod
import pandas as pd
from sklearn.preprocessing import StandardScaler
#%%
class BaseRecommender(ABC):
    def __init__(self, model_name:str, user_name:list, item_name:str, date_name:str|None,
                 sparse_features:list|None, dense_features:list|None, standard_bool:bool,
                 seed:int):
        self.model_name = model_name
        self.model = None
        self.kwargs = dict()
        self.user_name = user_name
        self.item_name = item_name
        self.date_name = date_name
        self.sparse_feature = sparse_features
        self.dense_feature = dense_features
        self.standard_bool = standard_bool
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
    #%%
    def _Standardize(self, data:pd.DataFrame, fit_bool:bool):
        if self.dense_feature is None:
            raise ValueError('dense_feature is None')
        if fit_bool:
            self.scaler = StandardScaler()
            data[self.dense_feature] = self.scaler.fit_transform(data[self.dense_feature])
        else:
            data[self.dense_feature] = self.scaler.transform(data[self.dense_feature])
        return data