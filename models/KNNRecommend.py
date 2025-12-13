# %%
import pandas as pd
from .base import BaseRecommender
from sklearn.neighbors import KNeighborsClassifier


# https://scikit-learn.org.cn/view/695.html
# %%
class KNNRecommend(BaseRecommender):
    def __init__(self, user_name: list, item_name: str, date_name: str | None = None,
                 sparse_features: list | None = None, dense_features: list | None = None, standard_bool: bool = False,
                 seed: int = 42, **kwargs):
        if (user_name is None) or (item_name is None):
            raise ValueError('user_name and item_name are required')
        super().__init__('KNN', user_name, item_name, date_name, sparse_features, dense_features, standard_bool, seed)
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
    def fit(self, train_data: pd.DataFrame):
        X = train_data[self.user_name]
        if self.standard_bool:
            X = self._Standardize(X, fit_bool=True)
        y = train_data[self.item_name]
        self.model.fit(X, y)
        self.unique_item = self.model.classes_
        self.is_trained = True

    # %%
    def recommend(self, test_data: pd.DataFrame, k: int = 5):
        if not self.is_trained:
            raise ValueError('model is not trained')
        X = test_data[self.user_name]
        if self.standard_bool:
            X = self._Standardize(X, fit_bool=False)
        y = self.model.predict_proba(X)
        result = pd.DataFrame(y, index=test_data.index, columns=self.unique_item)
        topk_item = result.apply(lambda row: row.nlargest(k).index.tolist(), axis=1)
        topk_item = pd.DataFrame(topk_item.tolist(), columns=[f'top{i + 1}' for i in range(k)])
        return topk_item
