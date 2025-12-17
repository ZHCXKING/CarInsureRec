# %%
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.evaluation import *


# %%
class BaseRecommender:
    def __init__(self, model_name: str, user_name: list, item_name: str, date_name: str | None,
                 sparse_features: list | None, dense_features: list | None, standard_bool: bool,
                 seed: int):
        self.model_name = model_name
        self.model = None
        self.kwargs = dict()
        self.user_name = user_name
        self.item_name = item_name
        self.date_name = date_name
        self.sparse_feature = sparse_features
        self.dense_feature = dense_features
        self.standard_bool = standard_bool
        self.mapping = {}
        self.vocabulary_sizes = {}
        self.seed = seed
        self.unique_item = None
        self.is_trained = False

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
    def get_proba(self, test_data: pd.DataFrame):
        if not self.is_trained:
            raise ValueError('model is not trained')
        X = test_data[self.user_name]
        if self.standard_bool:
            X = self._Standardize(X, fit_bool=False)
        y = self.model.predict_proba(X)
        result = pd.DataFrame(y, index=test_data.index, columns=self.unique_item)
        return result
    # %%
    def get_topk_proba(self, test_data: pd.DataFrame, k: int=5):
        result = self.get_proba(test_data)
        topk_item = result.apply(lambda row: row.nlargest(k).index.tolist(), axis=1)
        topk_item = pd.DataFrame(topk_item.tolist(), columns=[f'top{i + 1}' for i in range(k)])
        topk_proba = result.apply(lambda row: row.nlargest(k).tolist(), axis=1)
        topk_proba = pd.DataFrame(topk_proba.tolist(), columns=[f'top{i + 1}' for i in range(k)])
        return topk_item, topk_proba

    # %%
    def recommend(self, test_data: pd.DataFrame, k: int=5):
        result = self.get_proba(test_data)
        topk_item = result.apply(lambda row: row.nlargest(k).index.tolist(), axis=1)
        topk_item = pd.DataFrame(topk_item.tolist(), columns=[f'top{i + 1}' for i in range(k)])
        return topk_item

    # %%
    def score_test(self, test_data: pd.DataFrame, method: str='mrr', k: int=5):
        item = test_data[self.item_name]
        topk_item = self.recommend(test_data, k=k)
        if method == 'mrr_k':
            score = mrr_k(item, topk_item, k)
        elif method == 'recall_k':
            score = recall_k(item, topk_item, k)
        elif method == 'ndcg_k':
            score = ndcg_k(item, topk_item, k)
        else:
            raise ValueError('method is not supported')
        return score
    # %%
    def _Standardize(self, data: pd.DataFrame, fit_bool: bool):
        if self.dense_feature is None:
            raise ValueError('dense_feature is None')
        if fit_bool:
            self.scaler = StandardScaler()
            data[self.dense_feature] = self.scaler.fit_transform(data[self.dense_feature])
        else:
            data[self.dense_feature] = self.scaler.transform(data[self.dense_feature])
        return data

    # %%
    def _mapping(self, data: pd.DataFrame, fit_bool: bool):
        if fit_bool:
            for col in self.sparse_feature:
                unique = sorted(data[col].unique())
                mapping = {v: i + 1 for i, v in enumerate(unique)}
                self.mapping[col] = mapping
                self.vocabulary_sizes[col] = len(mapping) + 1
                data[col] = data[col].map(lambda x: self.mapping[col].get(x, 0))
        else:
            for col in self.sparse_feature:
                data[col] = data[col].map(lambda x: self.mapping[col].get(x, 0))
        return data
