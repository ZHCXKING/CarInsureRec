# %%
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from src.utils import load
from src.utils import filling, split_filling, mice_samples
from models import XGBRecommend, LGBMRecommend, DeepFMRecommend, MLPRecommend, KNNRecommend, CoMICERecommend
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from src.evaluation import *
#%%
m = 3
k = 5
train, test = load('original', amount=4000, split_num=250) #original, dropna
#train, test, _ = split_filling(train, test, method='iterative_SVM', seed=42)
user_name = ['Age', 'DrivingExp', 'Occupation', 'NCD', 'Make', 'Car.year', 'Car.price']
item_name = 'InsCov'
date_name = 'Date'
sparse_features = ['Occupation', 'NCD', 'Make']
dense_features = ['Age', 'Car.year', 'Car.price', 'DrivingExp']
score = []
model = CoMICERecommend(user_name, item_name, date_name, sparse_features, dense_features, seed=42, k=k, standard_bool=True)
model.fit(train)
score.append(model.score_test(test, method='recall_k'))
print(score)
# train_data_sets, test_data_sets, _ = mice_samples(train, test, method='iterative_NB', m=m, seed=42)
# all_test_probs = []
# for i in range(m):
#     model = CoMICERecommend(user_name, item_name, date_name, sparse_features, dense_features, seed=42, k=k)
#     model.fit(train_data_sets[i])
#     print(model.score_test(test_data_sets[i], method='recall_k'))
#     proba = model.get_proba(test_data_sets[i])
#     all_test_probs.append(proba)
# results = np.mean(all_test_probs, axis=0)
# results = pd.DataFrame(results, index=test.index)
# topk_item = results.apply(lambda row: row.nlargest(k).index.tolist(), axis=1)
# topk_item = pd.DataFrame(topk_item.tolist(), columns=[f'top{i + 1}' for i in range(k)])
# score = recall_k(test[item_name], topk_item, k=k)
# print(score)