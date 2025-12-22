# %%
import pandas as pd
import numpy as np
from src.utils import load
from models import *
# %%
def add_missing_values(df: pd.DataFrame, missing_rate: float, feature_cols: list = None, seed: int = 42):
    df_copy = df.copy()
    if feature_cols is None:
        feature_cols = df_copy.columns.tolist()
    data = df_copy[feature_cols].values.astype(float)
    n_rows, n_cols = data.shape
    rng = np.random.default_rng(seed)
    random_matrix = rng.random((n_rows, n_cols))
    mask = random_matrix < missing_rate
    data[mask] = np.nan
    df_copy[feature_cols] = data
    return df_copy
# train = add_missing_values(train, 0.1, feature_cols=sparse_features + dense_features, seed=42)
# test = add_missing_values(test, 0.1, feature_cols=sparse_features + dense_features, seed=42)
# %%
def test_seeds(data_type: str, amount: int, split_num: int, times: int, model_name: str, k: int):
    train, test, info = load(data_type, amount, split_num)
    item_name = info['item_name']
    sparse_features = info['sparse_features']
    dense_features = info['dense_features']
    score = []
    CoMICE_score = []
    for i in range(times):
        if model_name == 'DCNv2':
            model = DCNv2Recommend(item_name, sparse_features, dense_features, seed=i, k=k)
        elif model_name == 'DeepFM':
            model = DeepFMRecommend(item_name, sparse_features, dense_features, seed=i, k=k)
        elif model_name == 'WideDeep':
            model = WideDeepRecommend(item_name, sparse_features, dense_features, seed=i, k=k)
        elif model_name == 'AutoInt':
            model = AutoIntRecommend(item_name, sparse_features, dense_features, seed=i, k=k)
        else:
            raise ValueError('model_name is not valid')
        CoMICE_model = CoMICERecommend(item_name, sparse_features, dense_features, seed=i, k=k)
        train_data, test_data = train.copy(), test.copy()
        model.fit(train_data)
        score.append(model.score_test(test_data, method='auc'))
        train_data, test_data = train.copy(), test.copy()
        CoMICE_model.fit(train_data)
        CoMICE_score.append(CoMICE_model.score_test(test_data, method='auc'))
    return score, CoMICE_score
# %%
score, CoMICE_score = test_seeds('AWM', 2000, 1000, 3, 'AutoInt', 5)
print(score)
print(CoMICE_score)