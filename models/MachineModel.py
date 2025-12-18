# %%
import pandas as pd
from .base import BaseRecommender
from pyagrum.skbn import BNClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
# %%
class KNNRecommend(BaseRecommender):
    def __init__(self, user_name: list, item_name: str, date_name: str | None = None,
                 sparse_features: list | None = None, dense_features: list | None = None, standard_bool: bool = False,
                 seed: int = 42, k: int = 3, **kwargs):
        if (user_name is None) or (item_name is None):
            raise ValueError('user_name and item_name are required')
        super().__init__('KNN', user_name, item_name, date_name, sparse_features, dense_features, standard_bool, seed,k)
        default_params = {
            'n_neighbors': 10,
            'weights': 'distance',
            'p': 2
        }
        model_params = ['n_neighbors', 'weights', 'p']
        self.kwargs.update(default_params)
        self.kwargs.update(kwargs)
        model_params = {key: self.kwargs[key] for key in model_params}
        self.model = KNeighborsClassifier(**model_params)
# %%
class BNRecommend(BaseRecommender):
    def __init__(self, user_name: list, item_name: str, date_name: str | None = None,
                 sparse_features: list | None = None, dense_features: list | None = None, standard_bool: bool = False,
                 seed: int = 42, k: int = 3, **kwargs):
        if (user_name is None) or (item_name is None):
            raise ValueError('user_name and item_name are required')
        super().__init__('BN', user_name, item_name, date_name, sparse_features, dense_features, standard_bool, seed, k)
        default_params = {
            'learningMethod': 'MIIC',
            'scoringType': 'BIC',
            'discretizationStrategy': 'quantile',
            'discretizationNbBins': 5
        }
        model_params = ['learningMethod', 'scoringType', 'discretizationStrategy', 'discretizationNbBins']
        self.kwargs.update(default_params)
        self.kwargs.update(kwargs)
        model_params = {key: self.kwargs[key] for key in model_params}
        self.model = BNClassifier(**model_params)
    # %%
    def fit(self, train_data: pd.DataFrame):
        X = train_data[self.user_name]
        if self.standard_bool:
            X = self._standardize(X, fit_bool=True)
        y = train_data[self.item_name]
        self.model.fit(X, y)
        self.unique_item = self.model.bn.variable(self.model.target).labels()
        self.is_trained = True
# %%
class LRRecommend(BaseRecommender):
    def __init__(self, user_name: list, item_name: str, date_name: str | None = None,
                 sparse_features: list | None = None, dense_features: list | None = None, standard_bool: bool = False,
                 seed: int = 42, k: int = 3, **kwargs):
        if (user_name is None) or (item_name is None):
            raise ValueError('user_name and item_name are required')
        super().__init__('LR', user_name, item_name, date_name, sparse_features, dense_features, standard_bool, seed,k)
        default_params = {
            'C': 10,
            'penalty': 'l2',
            'solver': 'lbfgs',
            'class_weight': 'balanced',
            'max_iter': 1000,
            'random_state': self.seed
        }
        model_params = ['C', 'penalty', 'solver', 'class_weight', 'max_iter', 'random_state']
        self.kwargs.update(default_params)
        self.kwargs.update(kwargs)
        model_params = {key: self.kwargs[key] for key in model_params}
        self.model = LogisticRegression(**model_params)