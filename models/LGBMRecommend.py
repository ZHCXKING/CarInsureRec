# %%
from .base import BaseRecommender
from lightgbm import LGBMClassifier


# https://lightgbm.cn/en/stable/pythonapi/lightgbm.LGBMClassifier.html#lightgbm-lgbmclassifier
# %%
class LGBMRecommend(BaseRecommender):
    def __init__(self, user_name: list, item_name: str, date_name: str | None = None,
                 sparse_features: list | None = None, dense_features: list | None = None, standard_bool: bool = False,
                 seed: int = 42, **kwargs):
        if (user_name is None) or (item_name is None):
            raise ValueError('user_name and item_name are required')
        super().__init__('LightGBM', user_name, item_name, date_name, sparse_features, dense_features, standard_bool, seed)
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
