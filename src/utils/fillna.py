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
# %%
def select_interpolation(Max, Min, method: str = 'iterative_NB', seed: int = 42):
    if method == 'MICE_RF':
        imputer = IterativeImputer(estimator=RandomForestRegressor(random_state=seed), max_value=Max, min_value=Min, random_state=seed)
    elif method == 'MICE_NB':
        imputer = IterativeImputer(estimator=BayesianRidge(), max_value=Max, min_value=Min, random_state=seed, sample_posterior=True)
    elif method == 'MICE_Ga':
        imputer = IterativeImputer(estimator=GaussianProcessRegressor(random_state=seed), max_value=Max, min_value=Min, random_state=seed, sample_posterior=True)
    elif method == 'MICE_SVM':
        imputer = IterativeImputer(estimator=LinearSVR(random_state=seed), max_value=Max, min_value=Min, random_state=seed)
    elif method == 'KNN':
        imputer = KNNImputer(n_neighbors=5)
    elif method == 'MICE_XGB':
        xgb_estimator = XGBRegressor(random_state=seed)
        imputer = IterativeImputer(estimator=xgb_estimator, max_value=Max, min_value=Min, random_state=seed)
    elif method == 'MICE_LGBM':
        lgbm_estimator = LGBMRegressor(random_state=seed)
        imputer = IterativeImputer(estimator=lgbm_estimator, max_value=Max, min_value=Min, random_state=seed)
    else:
        raise ValueError('method must is not supported')
    imputer.set_output(transform='pandas')
    return imputer
# %%
def round(df: pd.DataFrame, sparse_features: list, item_name: str='product_item'):
    for col in sparse_features:
        df[col] = np.round(df[col])
        df[col] = df[col].astype('int64')
    df[item_name] = df[item_name].astype('int64')
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