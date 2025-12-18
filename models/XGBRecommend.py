# %%
from .base import BaseRecommender
from xgboost import XGBClassifier


# https://docs.xgboost.com.cn/en/stable/python/python_api.html#xgboost.XGBClassifier
# %%
class XGBRecommend(BaseRecommender):
    def __init__(self, user_name: list, item_name: str, date_name: str | None = None,
                 sparse_features: list | None = None, dense_features: list | None = None, standard_bool: bool = False,
                 seed: int = 42, k: int = 3, **kwargs):
        if (user_name is None) or (item_name is None):
            raise ValueError('user_name and item_name are required')
        super().__init__('XGBoost', user_name, item_name, date_name, sparse_features, dense_features, standard_bool, seed,k)
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
