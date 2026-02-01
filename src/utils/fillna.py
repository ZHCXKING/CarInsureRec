# %%
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.svm import LinearSVR
from sklearn.impute import KNNImputer
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from .GAIN import GAINImputer
from .MIWAE import MIWAEImputer
# %%
def select_interpolation(Max, Min, method: str = 'MICE_NB', seed: int = 42):
    if method == 'MICE_RF':
        imputer = IterativeImputer(
            estimator=RandomForestRegressor(n_estimators=20, max_depth=5, random_state=seed, n_jobs=-1),
            max_value=Max, min_value=Min, random_state=seed)
    elif method == 'MICE_NB':
        imputer = IterativeImputer(
            estimator=BayesianRidge(),
            max_value=Max, min_value=Min, random_state=seed, sample_posterior=True)
    elif method == 'MICE_Ga':
        imputer = IterativeImputer(
            estimator=GaussianProcessRegressor(random_state=seed),
            max_value=Max, min_value=Min, random_state=seed, sample_posterior=True)
    elif method == 'MICE_SVM':
        imputer = IterativeImputer(
            estimator=LinearSVR(random_state=seed),
            max_value=Max, min_value=Min, random_state=seed)
    elif method == 'KNN':
        imputer = KNNImputer(n_neighbors=5)
    elif method == 'MICE_XGB':
        xgb_estimator = XGBRegressor(n_estimators=20, max_depth=5, random_state=seed)
        imputer = IterativeImputer(estimator=xgb_estimator, max_value=Max, min_value=Min, random_state=seed)
    elif method == 'MICE_LGBM':
        lgbm_estimator = LGBMRegressor(n_estimators=20, max_depth=5, random_state=seed)
        imputer = IterativeImputer(estimator=lgbm_estimator, max_value=Max, min_value=Min, random_state=seed)
    elif method == 'GAIN':
        imputer = GAINImputer(batch_size=1024, epoch=50, seed=seed)
    elif method == 'MIWAE':
        imputer = MIWAEImputer(batch_size=1024, epoch=50, K=20, L=1000, seed=seed)
    else:
        raise ValueError('method must is not supported')
    imputer.set_output(transform='pandas')
    return imputer
# %%
def round(df: pd.DataFrame, sparse_features: list, item_name: str = 'product_item'):
    for col in sparse_features:
        df[col] = np.round(df[col])
        df[col] = df[col].astype('int64')
    df[item_name] = np.round(df[item_name]).astype('int64')
    return df
# %%
def filling(df: pd.DataFrame, method: str = 'iterative_NB', seed: int = 42):
    Max, Min = df.max(), df.min()
    imputer = select_interpolation(Max, Min, method, seed)
    df = imputer.fit_transform(df)
    return df, imputer
# %%
def get_filled_data(train, valid, test, sparse_features, method='MICE_NB', seed=42):
    _, imputer = filling(train.copy(), method=method, seed=seed)
    train_filled = imputer.transform(train.copy())
    valid_filled = imputer.transform(valid.copy())
    test_filled = imputer.transform(test.copy())
    train_filled = round(train_filled, sparse_features)
    valid_filled = round(valid_filled, sparse_features)
    test_filled = round(test_filled, sparse_features)
    return train_filled, valid_filled, test_filled
# %%
def inject_missingness(df, sparse_feats, dense_feats, ratio, seed=42, mode='random'):
    if ratio <= 0.0:
        return df.copy()
    data = df.copy()
    features = sparse_feats + dense_feats
    np.random.seed(seed)
    if mode == 'random':
        for col in features:
            non_null_idx = data[data[col].notnull()].index.tolist()
            n_mask = int(len(non_null_idx) * ratio)
            if n_mask > 0:
                mask_idx = np.random.choice(non_null_idx, n_mask, replace=False)
                data.loc[mask_idx, col] = np.nan
    elif mode == 'row':
        n_rows = len(data)
        n_mask = int(n_rows * ratio)
        if n_mask > 0:
            mask_rows = np.random.choice(data.index, n_mask, replace=False)
            data.loc[mask_rows, features] = np.nan
    elif mode == 'col':
        n_cols = len(features)
        n_mask = int(n_cols * ratio)
        if n_mask > 0:
            mask_cols = np.random.choice(features, n_mask, replace=False)
            data.loc[:, mask_cols] = np.nan
    elif mode == 'row_partial':
        n_rows = len(data)
        n_mask_rows = int(n_rows * ratio)
        mask_rows = np.random.choice(data.index, n_mask_rows, replace=False)
        cols_missing_ratio = 0.8
        for idx in mask_rows:
            available_feats = [c for c in features if pd.notnull(data.at[idx, c])]
            n_drop = int(len(available_feats) * cols_missing_ratio)
            if n_drop > 0:
                drop_feats = np.random.choice(available_feats, n_drop, replace=False)
                data.loc[idx, drop_feats] = np.nan
    return data
