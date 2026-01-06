# %%
import pandas as pd
import numpy as np
from .base import BaseRecommender
from pyagrum.skbn import BNClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
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
        self.dense_uniques = {}
    # %%
    def fit(self, train_data: pd.DataFrame):
        X = train_data[self.user_name]
        if self.standard_bool:
            X = self._standardize(X, fit_bool=True)
        if self.dense_features:
            for col in self.dense_features:
                unique_vals = np.sort(X[col].unique())
                self.dense_uniques[col] = unique_vals
        y = train_data[self.item_name]
        self.model.fit(X, y)
        labels = list(self.model.bn.variable(self.model.target).labels())
        self.unique_item = [int(x) for x in labels]
        self.is_trained = True
    # %%
    def get_proba(self, test_data: pd.DataFrame):
        if not self.is_trained:
            raise ValueError('model is not trained')
        X = test_data[self.user_name].copy()
        if self.standard_bool:
            X = self._standardize(X, fit_bool=False)
        if self.dense_features:
            for col in self.dense_features:
                if col in self.dense_uniques:
                    train_vals = self.dense_uniques[col]
                    test_vals = X[col].values
                    test_vals = np.clip(test_vals, train_vals[0], train_vals[-1])
                    idx = np.searchsorted(train_vals, test_vals, side="left")
                    idx = np.clip(idx, 1, len(train_vals) - 1)
                    left_vals = train_vals[idx - 1]
                    right_vals = train_vals[idx]
                    use_left = (test_vals - left_vals) < (right_vals - test_vals)
                    X[col] = np.where(use_left, left_vals, right_vals)
        y = self.model.predict_proba(X)
        result = pd.DataFrame(y, index=test_data.index, columns=self.unique_item)
        return result
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
# %%
class NBRecommend(BaseRecommender):
    def __init__(self, item_name: str, sparse_features: list, dense_features: list,
                 standard_bool: bool = True, seed: int = 42, k: int = 3, **kwargs):
        super().__init__('NB', item_name, sparse_features, dense_features, standard_bool, seed, k)
        default_params = {
            'var_smoothing': 1e-9
        }
        model_params = ['var_smoothing']
        self.kwargs.update(default_params)
        self.kwargs.update(kwargs)
        model_params = {key: self.kwargs[key] for key in model_params}
        self.model = GaussianNB(**model_params)