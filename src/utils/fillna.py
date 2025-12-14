# %%
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.gaussian_process import GaussianProcessRegressor


# %%
def _select_interpolation(Max, Min, method: str = 'iterative_NB', seed: int = 42):
    if method == 'iterative_RF':
        imputer = IterativeImputer(estimator=RandomForestRegressor(), max_iter=50, max_value=Max, min_value=Min, random_state=seed)
    elif method == 'iterative_NB':
        imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=50, max_value=Max, min_value=Min, random_state=seed)
    elif method == 'iterative_Ga':
        imputer = IterativeImputer(estimator=GaussianProcessRegressor(), max_iter=50, max_value=Max, min_value=Min, random_state=seed)
    else:
        raise ValueError('method must be iterative_RF or iterative_NB')
    return imputer


# %%
def _round(df: pd.DataFrame):
    discrete_cols = ['Age', 'DrivingExp', 'Occupation', 'NCD', 'InsCov', 'Make', 'Car.year', 'year', 'month', 'day']
    for col in discrete_cols:
        if col == 'NCD':
            df[col] = np.round(df[col], decimals=1)
        else:
            df[col] = np.round(df[col])
            df[col] = df[col].astype('int64')
    return df


# %%
def _date_to_numeric(df: pd.DataFrame):
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    df.drop('Date', axis=1, inplace=True)
    return df


# %%
def _numeric_to_date(df: pd.DataFrame):
    df['Date'] = pd.to_datetime(df[['year', 'month', 'day']])
    df.drop(['year', 'month', 'day'], axis=1, inplace=True)
    df.sort_values(by='Date', inplace=True, ignore_index=True)
    return df


# %%
def filling(df: pd.DataFrame, method: str = 'iterative_NB', seed: int = 42):
    df = _date_to_numeric(df)
    cols = df.columns
    Max, Min = df.max(), df.min()
    imputer = _select_interpolation(Max, Min, method, seed)
    df = imputer.fit_transform(df)
    df = pd.DataFrame(df, columns=cols)
    df = _round(df)
    df = _numeric_to_date(df)
    return df


# %%
def split_filling(train_data: pd.DataFrame, test_data: pd.DataFrame, method: str = 'iterative_NB', seed: int = 42):
    train_data = _date_to_numeric(train_data)
    test_data = _date_to_numeric(test_data)
    cols = train_data.columns
    Max, Min = train_data.max(), train_data.min()
    imputer = _select_interpolation(Max, Min, method, seed)
    train_data = pd.DataFrame(imputer.fit_transform(train_data), columns=cols)
    test_data = pd.DataFrame(imputer.transform(test_data), columns=cols)
    train_data = _round(train_data)
    test_data = _round(test_data)
    train_data = _numeric_to_date(train_data)
    test_data = _numeric_to_date(test_data)
    return train_data, test_data
