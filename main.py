# %%
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from src.utils import load
from src.utils import filling, split_filling, mice_samples
from models import XGBRecommend, LGBMRecommend
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
#%%
m = 3
train, test = load('dropna', amount=4000, split_num=2000) #original, dropna
user_name = ['Age', 'DrivingExp', 'Occupation', 'NCD', 'Make', 'Car.year', 'Car.price']
item_name = 'InsCov'
date_name = 'Date'
sparse_features = ['Occupation', 'NCD', 'Make']
dense_features = ['Age', 'Car.year', 'Car.price', 'DrivingExp']
train_data_sets, test_data_sets, _ = mice_samples(train, test, method='iterative_NB', m=m, seed=42)
for i in range(m):
    model = XGBRecommend(user_name, item_name, date_name, sparse_features, dense_features, seed=42)
    model.fit(train_data_sets[i])
    score = model.score_test(test_data_sets[i])
    print(score)
#%%
# model = XGBRecommend(user_name, item_name, date_name, sparse_features, dense_features)
# model.fit(train)
# score = model.score_test(test)
# print(score)
#%%
# Max, Min = train.max(), train.min()
# imputer = IterativeImputer(estimator=LinearSVR(random_state=41), max_iter=50, max_value=Max, min_value=Min, random_state=41, add_indicator=True, sample_posterior=False, verbose=1)
# imputer.set_output(transform='pandas')
# train = imputer.fit_transform(train)
# test = imputer.transform(test)
# all_features = imputer.get_feature_names_out()
# input_features = imputer.feature_names_in_
# indicator = [f for f in all_features if f not in input_features]
# user_name = user_name + indicator
# model = XGBRecommend(user_name, item_name, date_name, sparse_features, dense_features)
# model.fit(train)
# score = model.score_test(test)
# print(score)
#%%
# m = 20
# train_datas = []
# test_datas = []
# imputers = []
# for i in range(m):
#     imputer = IterativeImputer(
#         max_value = Max,
#         min_value = Min,
#         sample_posterior = True,
#         random_state = i,
#         add_indicator = True
#     ).set_output(transform='pandas')
#     train_data = imputer.fit_transform(train)
#     train_datas.append(train_data)
#     imputers.append(imputer)
# for i in range(m):
#     imputer = imputers[i]
#     test_data = imputer.transform(test)
#     test_datas.append(test_data)
# def create(train_datas):
#     n_samples = train_datas[0].shape[0]
#     n_features = train_datas[0].shape[1]
#     m = len(train_datas)
#     rng = np.random.RandomState(42)
#     source_indices = rng.randint(0, m, size=n_samples)
#     stacked_data = np.stack([df.values for df in train_datas])
#     selected_values = stacked_data[source_indices, np.arange(n_samples), :]
#     df_random_select = pd.DataFrame(selected_values, columns=train_datas[0].columns, index=train_datas[0].index)
#     return df_random_select
# train = create(train_datas)
# test = test_datas[0]
# model = XGBRecommend(user_name, item_name, date_name, sparse_features, dense_features)
# model.fit(train)
# score = model.score_test(test)
# print(score)