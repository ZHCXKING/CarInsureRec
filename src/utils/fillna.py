# %%
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.svm import LinearSVR


# %%
def _select_interpolation(Max, Min, method: str = 'iterative_NB', seed: int = 42):
    if method == 'iterative_RF':
        imputer = IterativeImputer(estimator=RandomForestRegressor(random_state=seed), max_value=Max, min_value=Min, random_state=seed)
    elif method == 'iterative_NB':
        imputer = IterativeImputer(estimator=BayesianRidge(), max_value=Max, min_value=Min, random_state=seed, sample_posterior=True)
    elif method == 'iterative_Ga':
        imputer = IterativeImputer(estimator=GaussianProcessRegressor(random_state=seed), max_value=Max, min_value=Min, random_state=seed, sample_posterior=True)
    elif method == 'iterative_SVM':
        imputer = IterativeImputer(estimator=LinearSVR(random_state=seed), max_value=Max, min_value=Min, random_state=seed)
    else:
        raise ValueError('method must be iterative_RF, iterative_NB, iterative_Ga, iterative_SVM')
    imputer.set_output(transform='pandas')
    return imputer


# %%
def _round(df: pd.DataFrame):
    discrete_cols = df.columns
    for col in discrete_cols:
        if col == 'NCD':
            df[col] = np.round(df[col], decimals=1)
        elif col == 'Car.price':
            continue
        else:
            df[col] = np.round(df[col])
            df[col] = df[col].astype('int64')
    return df


# %%
def filling(df: pd.DataFrame, method: str = 'iterative_NB', seed: int = 42):
    Max, Min = df.max(), df.min()
    imputer = _select_interpolation(Max, Min, method, seed)
    df = imputer.fit_transform(df)
    df = _round(df)
    return df


# %%
def split_filling(train_data: pd.DataFrame, test_data: pd.DataFrame, method: str = 'iterative_NB', seed: int = 42):
    Max, Min = train_data.max(), train_data.min()
    imputer = _select_interpolation(Max, Min, method, seed)
    train_data = imputer.fit_transform(train_data)
    test_data = imputer.transform(test_data)
    train_data = _round(train_data)
    test_data = _round(test_data)
    return train_data, test_data


# %%
def mice_samples(train_data: pd.DataFrame, test_data: pd.DataFrame, method: str = 'iterative_SVM', m: int = 5, seed: int = 42):
    Max, Min = train_data.max(), train_data.min()
    train_data_sets = []
    test_data_sets = []
    imputer_sets = []
    rng = np.random.default_rng(seed=seed)
    rints = rng.choice(a=m*10, size=m, replace=False)
    for i in range(m):
        imputer = _select_interpolation(Max, Min, method, seed=rints[i])
        train = imputer.fit_transform(train_data)
        test = imputer.transform(test_data)
        train = _round(train)
        test = _round(test)
        train_data_sets.append(train)
        test_data_sets.append(test)
        imputer_sets.append(imputer)
    return train_data_sets, test_data_sets, imputer_sets