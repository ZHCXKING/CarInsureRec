# %%
from sklearn.preprocessing import StandardScaler
from src.evaluation import *
# %%
class BaseRecommender:
    def __init__(self, model_name: str, user_name: list, item_name: str, date_name: str | None,
                 sparse_features: list | None, dense_features: list | None, standard_bool: bool,
                 seed: int, k: int):
        self.model_name = model_name
        self.model = None
        self.kwargs = dict()
        self.user_name = user_name
        self.item_name = item_name
        self.date_name = date_name
        self.sparse_features = sparse_features
        self.dense_features = dense_features
        self.standard_bool = standard_bool
        self.mapping = {}
        self.vocabulary_sizes = {}
        self.seed = seed
        self.k = k
        self.unique_item = None
        self.is_trained = False
    # %%
    def fit(self, train_data: pd.DataFrame):
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
        X = test_data[self.user_name]
        if self.standard_bool:
            X = self._standardize(X, fit_bool=False)
        y = self.model.predict_proba(X)
        result = pd.DataFrame(y, index=test_data.index, columns=self.unique_item)
        return result
    # %%
    def get_topk_proba(self, test_data: pd.DataFrame):
        result = self.get_proba(test_data)
        topk_item = result.apply(lambda row: row.nlargest(self.k).index.tolist(), axis=1)
        topk_item = pd.DataFrame(topk_item.tolist(), columns=[f'top{i + 1}' for i in range(self.k)])
        topk_proba = result.apply(lambda row: row.nlargest(self.k).tolist(), axis=1)
        topk_proba = pd.DataFrame(topk_proba.tolist(), columns=[f'top{i + 1}' for i in range(self.k)])
        return topk_item, topk_proba
    # %%
    def recommend(self, test_data: pd.DataFrame):
        result = self.get_proba(test_data)
        topk_item = result.apply(lambda row: row.nlargest(self.k).index.tolist(), axis=1)
        topk_item = pd.DataFrame(topk_item.tolist(), columns=[f'top{i + 1}' for i in range(self.k)])
        return topk_item
    # %%
    def score_test(self, test_data: pd.DataFrame, method: str = 'mrr'):
        score = None
        item = test_data[self.item_name]
        if method in ['auc', 'logloss']:
            all_proba = self.get_proba(test_data)
            if method == 'auc':
                score = auc(item, all_proba, self.unique_item)
            elif method == 'logloss':
                score = logloss(item, all_proba, self.unique_item)
        else:
            topk_item = self.recommend(test_data)
            if method == 'mrr_k':
                score = mrr_k(item, topk_item, self.k)
            elif method == 'recall_k':
                score = recall_k(item, topk_item, self.k)
            elif method == 'ndcg_k':
                score = ndcg_k(item, topk_item, self.k)
            else:
                raise ValueError('method is not supported')
        return score
    # %%
    def _standardize(self, data: pd.DataFrame, fit_bool: bool):
        if self.dense_features is None:
            raise ValueError('dense_feature is None')
        if fit_bool:
            self.scaler = StandardScaler()
            data[self.dense_features] = self.scaler.fit_transform(data[self.dense_features])
        else:
            data[self.dense_features] = self.scaler.transform(data[self.dense_features])
        return data
    # %%
    def _mapping(self, data: pd.DataFrame, fit_bool: bool):
        if fit_bool:
            for col in self.sparse_features:
                unique = sorted(data[col].unique())
                mapping = {v: i + 1 for i, v in enumerate(unique)}
                self.mapping[col] = mapping
                self.vocabulary_sizes[col] = len(mapping) + 1
                data[col] = data[col].map(lambda x: self.mapping[col].get(x, 0))
        else:
            for col in self.sparse_features:
                data[col] = data[col].map(lambda x: self.mapping[col].get(x, 0))
        return data
