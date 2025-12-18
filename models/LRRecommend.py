# %%
from .base import BaseRecommender
from sklearn.linear_model import LogisticRegression


# https://scikit-learn.org.cn/view/378.html
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
