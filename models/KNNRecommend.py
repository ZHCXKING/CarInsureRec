# %%
from .base import BaseRecommender
from sklearn.neighbors import KNeighborsClassifier


# https://scikit-learn.org.cn/view/695.html
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
