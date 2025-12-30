# %%
from sklearn.preprocessing import StandardScaler
from src.evaluation import *
# %%
class BaseRecommender:
    def __init__(self, model_name: str, item_name: str, sparse_features: list, dense_features: list,
                 standard_bool: bool, seed: int, k: int):
        self.model_name = model_name
        self.model = None
        self.kwargs = dict()
        self.user_name = sparse_features + dense_features
        self.item_name = item_name
        self.sparse_features = sparse_features
        self.dense_features = dense_features
        self.standard_bool = standard_bool
        self.mapping = {}
        self.vocabulary_sizes = {}
        self.scaler = None
        self.seed = seed
        self.k = k
        self.unique_item = None
        self.is_trained = False
    # %%
    def fit(self, train_data: pd.DataFrame):
        X = train_data[self.user_name].copy()
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
        X = test_data[self.user_name].copy()
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
    def score_test(self, test_data: pd.DataFrame, methods: list):
        scores_dict = {}
        item = test_data[self.item_name]
        all_proba = self.get_proba(test_data)
        topk_item = self.recommend(test_data)
        for m in methods:
            if m == 'auc':
                scores_dict['auc'] = auc(item, all_proba, self.unique_item)
            elif m == 'logloss':
                scores_dict['logloss'] = logloss(item, all_proba, self.unique_item)
            elif m == 'mrr_k':
                scores_dict['mrr_k'] = mrr_k(item, topk_item, self.k)
            elif m == 'recall_k':
                scores_dict['recall_k'] = recall_k(item, topk_item, self.k)
            elif m == 'ndcg_k':
                scores_dict['ndcg_k'] = ndcg_k(item, topk_item, self.k)
        # 3. 按照输入 methods 的顺序构造返回列表
        final_scores = []
        for m in methods:
            final_scores.append(scores_dict[m])
        return final_scores
    # %%
    def _standardize(self, data: pd.DataFrame, fit_bool: bool):
        data = data.copy()
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
                data[col] = data[col].map(mapping).fillna(0).astype('int64')
        else:
            for col in self.sparse_features:
                data[col] = data[col].map(self.mapping[col]).fillna(0).astype('int64')
        return data
