#%%
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
#%%
def _select_interpolation(method: str='iterative_NB'):
    if method == 'iterative_RF':
        imputer = IterativeImputer(estimator=RandomForestRegressor(), max_iter=50, verbose=1)
    elif method == 'iterative_NB':
        imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=50, verbose=1)
    else:
        raise ValueError('method must be iterative_RF or iterative_NB')
    return imputer
#%%
def _round(df:pd.DataFrame):
    discrete_cols = ['Age', 'DrivingExp', 'Occupation', 'NCD', 'InsCov', 'Make', 'Car.year', 'year', 'month', 'day']
    for col in discrete_cols:
        if col == 'NCD':
            df[col] = np.round(df[col], decimals=1)
        else:
            df[col] = np.round(df[col])
            df[col] = df[col].astype('int64')
    return df
#%%
def _date_to_numeric(df:pd.DataFrame):
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    df.drop('Date', axis=1, inplace=True)
    return df
#%%
def _numeric_to_date(df: pd.DataFrame):
    df['Date'] = pd.to_datetime(df[['year', 'month', 'day']])
    df.drop(['year', 'month', 'day'], axis=1, inplace=True)
    df.sort_values(by='Date', inplace=True, ignore_index=True)
    return df
#%%
def filling(df: pd.DataFrame, method: str= 'iterative_NB', split_num: int|None=None):
    df = _date_to_numeric(df)
    cols = df.columns
    imputer = _select_interpolation(method)
    if split_num is not None:
        data_train = df[:split_num].reset_index(drop=True)
        data_train = pd.DataFrame(imputer.fit_transform(data_train), columns=cols)
        data_train = _round(data_train)
        data_train = _numeric_to_date(data_train)
        data_test = df[split_num:].reset_index(drop=True)
        data_test = pd.DataFrame(imputer.transform(data_test), columns=cols)
        data_test = _round(data_test)
        data_test = _numeric_to_date(data_test)
        return data_train, data_test
    else:
        df = imputer.fit_transform(df)
        df = pd.DataFrame(df, columns=cols)
        df = _round(df)
        df = _numeric_to_date(df)
        return df