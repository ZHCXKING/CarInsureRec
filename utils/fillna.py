#%%
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
#%%
def iterative(df: pd.DataFrame, method: str='NB'):
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    df.drop('Date', axis=1, inplace=True)
    cols = df.columns
    idx = df.index
    if method == 'RF':
        imputer = IterativeImputer(estimator=RandomForestRegressor(), max_iter=50, verbose=1)
    elif method == 'NB':
        imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=50, verbose=1)
    df = imputer.fit_transform(df)
    df = pd.DataFrame(df, index=idx, columns=cols)
    discrete_cols = ['Age', 'DrivingExp', 'Occupation', 'NCD', 'InsCov', 'Make', 'Car.year', 'year', 'month', 'day']
    for col in discrete_cols:
        if col == 'NCD':
            df[col] = np.round(df[col], decimals=1)
        else:
            df[col] = np.round(df[col])
            df[col] = df[col].astype('int64')
    df['Date'] = pd.to_datetime(df[['year', 'month', 'day']])
    df.drop(['year', 'month', 'day'], axis=1, inplace=True)
    df.sort_values(by='Date', inplace=True, ignore_index=True)
    return df