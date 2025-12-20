# %%
import pandas as pd
from .base import BaseRecommender
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
# %%
class XGBRecommend(BaseRecommender):
    def __init__(self, user_name: list, item_name: str, date_name: str | None = None,
                 sparse_features: list | None = None, dense_features: list | None = None, standard_bool: bool = False,
                 seed: int = 42, k: int = 3, **kwargs):
        if (user_name is None) or (item_name is None):
            raise ValueError('user_name and item_name are required')
        super().__init__('XGBoost', user_name, item_name, date_name, sparse_features, dense_features, standard_bool, seed, k)
        default_params = {
            'n_estimators': None,
            'learning_rate': None,
            'booster': 'gbtree',
            'max_depth': None,
            'early_stopping_rounds': None,
            'random_state': self.seed
        }
        model_params = ['n_estimators', 'learning_rate', 'booster', 'max_depth', 'early_stopping_rounds', 'random_state']
        self.kwargs.update(default_params)
        self.kwargs.update(kwargs)
        model_params = {key: self.kwargs[key] for key in model_params}
        self.model = XGBClassifier(**model_params)
# %%
class LGBMRecommend(BaseRecommender):
    def __init__(self, user_name: list, item_name: str, date_name: str | None = None,
                 sparse_features: list | None = None, dense_features: list | None = None, standard_bool: bool = False,
                 seed: int = 42, k: int = 3, **kwargs):
        if (user_name is None) or (item_name is None):
            raise ValueError('user_name and item_name are required')
        super().__init__('LightGBM', user_name, item_name, date_name, sparse_features, dense_features, standard_bool, seed, k)
        default_params = {
            'learning_rate': 0.1,
            'n_estimators': 100,
            'num_leaves': 31,
            'max_depth': -1,
            'boosting_type': 'gbdt',
            'random_state': self.seed
        }
        model_params = ['learning_rate', 'n_estimators', 'num_leaves', 'max_depth', 'boosting_type', 'random_state']
        self.kwargs.update(default_params)
        self.kwargs.update(kwargs)
        model_params = {key: self.kwargs[key] for key in model_params}
        self.model = LGBMClassifier(**model_params)
# %%
class CatBRecommend(BaseRecommender):
    def __init__(self, user_name: list, item_name: str, date_name: str | None = None,
                 sparse_features: list | None = None, dense_features: list | None = None, standard_bool: bool = False,
                 seed: int = 42, k: int = 3, **kwargs):
        if (user_name is None) or (item_name is None):
            raise ValueError('user_name and item_name are required')
        super().__init__('XGBoost', user_name, item_name, date_name, sparse_features, dense_features, standard_bool, seed, k)
        default_params = {
            'iterations': None,
            'learning_rate': None,
            'depth': None,
            'allow_writing_files': False,
            'verbose': True,
            'cat_features': self.sparse_features,
            'random_state': self.seed
        }
        model_params = ['iterations', 'learning_rate', 'depth', 'allow_writing_files', 'verbose', 'cat_features', 'random_state']
        self.kwargs.update(default_params)
        self.kwargs.update(kwargs)
        model_params = {key: self.kwargs[key] for key in model_params}
        self.model = CatBoostClassifier(**model_params)
    # %%
    def fit(self, train_data: pd.DataFrame):
        train_data = self._mapping(train_data, fit_bool=True)
        X = train_data[self.user_name]
        if self.standard_bool:
            X = self._standardize(X, fit_bool=True)
        y = train_data[self.item_name]
        self.model.fit(X, y)
        self.unique_item = self.model.classes_
        self.is_trained = True
    # %%
    def get_proba(self, test_data: pd.DataFrame):
        if not self.is_trained:
            raise ValueError('model is not trained')
        test_data = self._mapping(test_data, fit_bool=False)
        X = test_data[self.user_name]
        if self.standard_bool:
            X = self._standardize(X, fit_bool=False)
        y = self.model.predict_proba(X)
        result = pd.DataFrame(y, index=test_data.index, columns=self.unique_item)
        return result
# %%
class RFRecommend(BaseRecommender):
    def __init__(self, user_name: list, item_name: str, date_name: str | None = None,
                 sparse_features: list | None = None, dense_features: list | None = None, standard_bool: bool = False,
                 seed: int = 42, k: int = 3, **kwargs):
        if (user_name is None) or (item_name is None):
            raise ValueError('user_name and item_name are required')
        super().__init__('RF', user_name, item_name, date_name, sparse_features, dense_features, standard_bool, seed, k)
        default_params = {
            'n_estimators': 100,
            'criterion': 'gini',
            'max_depth': None,
            'random_state': self.seed
        }
        model_params = ['n_estimators', 'criterion', 'max_depth', 'random_state']
        self.kwargs.update(default_params)
        self.kwargs.update(kwargs)
        model_params = {key: self.kwargs[key] for key in model_params}
        self.model = RandomForestClassifier(**model_params)
