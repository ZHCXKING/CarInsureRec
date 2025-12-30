# %%
import pandas as pd
from .base import BaseRecommender
from pyagrum.skbn import BNClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
# %%
class KNNRecommend(BaseRecommender):
    def __init__(self, item_name: str, sparse_features: list, dense_features: list,
                 standard_bool: bool = True, seed: int = 42, k: int = 3, **kwargs):
        super().__init__('KNN', item_name, sparse_features, dense_features, standard_bool, seed, k)
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
    def __init__(self, item_name: str, sparse_features: list, dense_features: list,
                 standard_bool: bool = True, seed: int = 42, k: int = 3, **kwargs):
        super().__init__('BN', item_name, sparse_features, dense_features, standard_bool, seed, k)
        default_params = {
            'learningMethod': 'MIIC',
            'scoringType': 'BIC',
            'discretizationStrategy': 'uniform',
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
        labels = list(self.model.bn.variable(self.model.target).labels())
        self.unique_item = [int(x) for x in labels]
        self.is_trained = True
# %%
class LRRecommend(BaseRecommender):
    def __init__(self, item_name: str, sparse_features: list, dense_features: list,
                 standard_bool: bool = True, seed: int = 42, k: int = 3, **kwargs):
        super().__init__('LR', item_name, sparse_features, dense_features, standard_bool, seed, k)
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
