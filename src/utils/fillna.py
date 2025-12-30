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
    else:
        raise ValueError('method must be MICE_RF, MICE_NB, MICE_Ga, MICE_SVM, KNN')
    imputer.set_output(transform='pandas')
    return imputer
# %%
def round(df: pd.DataFrame, sparse_features: list):
    for col in sparse_features:
        df[col] = np.round(df[col])
        df[col] = df[col].astype('int64')
    return df
# %%
def filling(df: pd.DataFrame, method: str = 'iterative_NB', seed: int = 42):
    Max, Min = df.max(), df.min()
    imputer = select_interpolation(Max, Min, method, seed)
    df = imputer.fit_transform(df)
    return df, imputer