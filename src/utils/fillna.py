# %%
import pandas as pd
import numpy as np
import torch
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.svm import LinearSVR
from sklearn.impute import KNNImputer
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
    else:
        raise ValueError('method must be MICE_RF, MICE_NB, MICE_Ga, MICE_SVM, KNN')
    imputer.set_output(transform='pandas')
    return imputer
# %%
def round(df: pd.DataFrame):
    discrete_cols = df.columns
    for col in discrete_cols:
        df[col] = np.round(df[col])
        df[col] = df[col].astype('int64')
    return df
# %%
def filling(df: pd.DataFrame, method: str = 'iterative_NB', seed: int = 42):
    Max, Min = df.max(), df.min()
    imputer = select_interpolation(Max, Min, method, seed)
    df = imputer.fit_transform(df)
    df = round(df)
    return df, imputer
# %%
def split_filling(train_data: pd.DataFrame, test_data: pd.DataFrame, method: str = 'iterative_NB', seed: int = 42):
    Max, Min = train_data.max(), train_data.min()
    imputer = select_interpolation(Max, Min, method, seed)
    train_data = imputer.fit_transform(train_data)
    test_data = imputer.transform(test_data)
    train_data = round(train_data)
    test_data = round(test_data)
    return train_data, test_data, imputer
# %%
def mice_samples(train_data: pd.DataFrame, test_data: pd.DataFrame, method: str = 'iterative_SVM', m: int = 5, seed: int = 42):
    Max, Min = train_data.max(), train_data.min()
    train_data_sets = []
    test_data_sets = []
    imputer_sets = []
    rng = np.random.default_rng(seed=seed)
    rints = rng.choice(a=m * 10, size=m, replace=False)
    for i in range(m):
        imputer = select_interpolation(Max, Min, method, seed=rints[i])
        train = imputer.fit_transform(train_data)
        test = imputer.transform(test_data)
        train = round(train)
        test = round(test)
        train_data_sets.append(train)
        test_data_sets.append(test)
        imputer_sets.append(imputer)
    return train_data_sets, test_data_sets, imputer_sets
# %%
def process_mice_list(df_list, user_name, item_name):
    # 堆叠成 [N, m, d]
    all_versions = [torch.tensor(df[user_name].values, dtype=torch.float) for df in df_list]
    X_imputed = torch.stack(all_versions, dim=1)
    y = torch.tensor(df_list[0][item_name].values, dtype=torch.long)
    return X_imputed, y
