# %%
import pandas as pd
import torch
from .base import BaseRecommender
from rec4torch.models import DeepFM
from rec4torch.snippets import seed_everything
# %%
class DeepFMRecommend(BaseRecommender):
    def __init__(self, user_name: list, item_name: str, date_name: str | None = None,
                 sparse_features: list | None = None, dense_features: list | None = None, standard_bool: bool = False,
                 seed: int = 42, **kwargs):
        if (user_name is None) or (item_name is None):
            raise ValueError('user_name and item_name are required')
        super().__init__('DeepFM', user_name, item_name, date_name, sparse_features, dense_features, standard_bool, seed)
        default_params = {
            'embedding_dim': 4,
            'dimension': 1,
            'batch_size': 16,
            'epochs': 10,
            'lr': 0.01
        }
        self.kwargs.update(default_params)
        self.kwargs.update(kwargs)
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.out_dim = None
        self.mapping = {}
        self.vocabulary_sizes = {}
        seed_everything(self.seed)
    #%%
    def _mapping(self, data: pd.DataFrame, fit_bool: bool):
        if fit_bool:
            for col in self.sparse_feature:
                unique = data[col].unique()
                mapping = {v: i+1 for i, v in enumerate(unique)}
                self.mapping[col] = mapping
                self.vocabulary_sizes[col] = len(mapping) + 1
                data[col] = data[col].map(lambda x: self.mapping[col].get(x, 0))
        else:
            for col in self.sparse_feature:
                data[col] = data[col].map(lambda x: self.mapping[col].get(x, 0))
        return data
    #%%
    def fit(self, train_data: pd.DataFrame):
        self.out_dim = train_data[self.item_name].nunique()
        self.unique_item = train_data[self.item_name].unique().tolist()
        print(self.unique_item)
        train_data = self._mapping(train_data, fit_bool=True)
        print(train_data['NCD'])
        for col in self.sparse_feature:
            print(train_data[col].min(), train_data[col].max(), train_data[col].nunique())
    #%%
    def recommend(self, test_data: pd.DataFrame, k: int = 5):
        pass