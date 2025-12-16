# %%
from .base import BaseRecommender
from sklearn.neural_network import MLPClassifier


# https://scikit-learn.org.cn/view/713.html
# %%
class MLPRecommend(BaseRecommender):
    def __init__(self, user_name: list, item_name: str, date_name: str | None = None,
                 sparse_features: list | None = None, dense_features: list | None = None, standard_bool: bool = False,
                 seed: int = 42, **kwargs):
        if (user_name is None) or (item_name is None):
            raise ValueError('user_name and item_name are required')
        super().__init__('MLP', user_name, item_name, date_name, sparse_features, dense_features, standard_bool, seed)
        default_params = {
            'hidden_layer_sizes': (64, 32),
            'activation': 'relu',
            'solver': 'adam',
            'learning_rate_init': 0.001,
            'max_iter': 1000,
            'batch_size': 8,
            'random_state': self.seed,
            'verbose': False
        }
        model_params = ['hidden_layer_sizes', 'activation', 'solver', 'learning_rate_init', 'max_iter', 'batch_size', 'random_state', 'verbose']
        self.kwargs.update(default_params)
        self.kwargs.update(kwargs)
        model_params = {key: self.kwargs[key] for key in model_params}
        self.model = MLPClassifier(**model_params)
