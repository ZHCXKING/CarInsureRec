# %%
import pandas as pd
import numpy as np
from src.utils import load
from models import *
# %%
class experiment():
    def __init__(self, data_type: str, amount: int, split_num: int, times: int, metric: str, k: int, default_params: dict, seed: int):
        self.data_type = data_type
        self.amount = amount
        self.split_num = split_num
        self.times = times
        self.metric = metric
        self.k = k
        self.default_params = default_params
        self.seed = seed
        self.models = ['DCNv2', 'DeepFM', 'WideDeep', 'AutoInt', 'CoMICE']
        self._data()
    def _data(self):
        self.train, self.test, info = load(self.data_type, self.amount, self.split_num)
        self.item_name = info['item_name']
        self.sparse_features = info['sparse_features']
        self.dense_features = info['dense_features']
    def add_missing_values(self, df: pd.DataFrame, missing_rate: float):
        df_copy = df.copy()
        data = df_copy[self.sparse_features + self.dense_features].values.astype(float)
        n_rows, n_cols = data.shape
        rng = np.random.default_rng(self.seed)
        random_matrix = rng.random((n_rows, n_cols))
        mask = random_matrix < missing_rate
        data[mask] = np.nan
        df_copy[self.sparse_features + self.dense_features] = data
        return df_copy
    def test_seeds(self, model_name: str):
        score = []
        for i in range(self.times):
            if model_name == 'DCNv2':
                model = DCNv2Recommend(self.item_name, self.sparse_features, self.dense_features, seed=i, k=self.k, **default_params)
            elif model_name == 'DeepFM':
                model = DeepFMRecommend(self.item_name, self.sparse_features, self.dense_features, seed=i, k=self.k, **default_params)
            elif model_name == 'WideDeep':
                model = WideDeepRecommend(self.item_name, self.sparse_features, self.dense_features, seed=i, k=self.k, **default_params)
            elif model_name == 'AutoInt':
                model = AutoIntRecommend(self.item_name, self.sparse_features, self.dense_features, seed=i, k=self.k, **default_params)
            elif model_name == 'CoMICE':
                model = CoMICERecommend(self.item_name, self.sparse_features, self.dense_features, seed=i, k=self.k, **default_params)
            else:
                raise ValueError('model_name is not valid')
            train_data, test_data = self.train.copy(), self.test.copy()
            model.fit(train_data)
            score.append(model.score_test(test_data, method=self.metric))
        mean = np.mean(score)
        std = np.std(score)
        return mean, std
    def test_models(self):
        result = {}
        for model_name in self.models:
            score = self.test_seeds(model_name)
            result[model_name] = score
        return result
# %%
default_params = {
    'lr': 1e-4,
    'batch_size': 512,
    'feature_dim': 32,
    'proj_dim': 32,
    'epochs': 100,
    'lambda_nce': 1.0,
    'temperature': 0.1,
    'mice_method': 'MICE_RF',
    'cross_layers': 3,
    'hidden_units': [256, 128],
    'dropout': 0.1,
    'attention_layers': 3,
    'num_heads': 2
}
data_type = 'AWM'
amount = 2700
split_num = 2000
times = 2
metric = 'auc'
k = 3
seed = 42
# train = add_missing_values(train, 0.1, feature_cols=sparse_features + dense_features, seed=42)
# test = add_missing_values(test, 0.1, feature_cols=sparse_features + dense_features, seed=42)
exper = experiment(data_type, amount, split_num, times, metric, k, default_params, seed)
score = exper.test_models()
print(score)