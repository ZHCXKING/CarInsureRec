# %%
from src.utils import load, filling, round, inject_missingness
from src.models import *
from src.evaluation import *
import json
from pathlib import Path
#%%
k = 3
seed = 0
ratio = 0.1
train, valid, test, info = load('AWM', amount=None, train_ratio=0.7, val_ratio=0.1, is_dropna=False) #original, dropna
item_name = info['item_name']
sparse_features = info['sparse_features']
dense_features = info['dense_features']
train = inject_missingness(train, sparse_features, dense_features, ratio=ratio, seed=seed)
valid = inject_missingness(valid, sparse_features, dense_features, ratio=ratio, seed=seed)
test = inject_missingness(test, sparse_features, dense_features, ratio=ratio, seed=seed)
_, imputer = filling(train, method='MICE_NB', seed=seed)
train_filled = imputer.transform(train.copy())
valid_filled = imputer.transform(valid.copy())
test_filled = imputer.transform(test.copy())
train_filled = round(train_filled, sparse_features)
valid_filled = round(valid_filled, sparse_features)
test_filled = round(test_filled, sparse_features)
score = []
# model = DeepFMRecommend.load('experiment/AWM/Original_model/DeepFM.pth')
# score.append(model.score_test(test_filled.copy(), methods=['auc', 'ndcg_k']))
# model = CoMICERecommend.load('experiment/AWM/Original_model/CoMICE_DeepFM.pth')
# score.append(model.score_test(test_filled.copy(), methods=['auc', 'ndcg_k']))
root = Path(__file__).parents[0]
param_file = root / 'experiment' / 'AWM' / ('DCN' + "_param.json")
with open(param_file, 'r') as f:
    params = json.load(f)
#params['lambda_nce'] = 1.0
# params['temperature'] = 0.1
#params['batch_size'] = 1024
# params['proj_dim'] = 64
all = []
test_param = params.copy()
#test_param['lambda_nce'] = 0.0
# test_param['temperature'] = 0.1
#test_param['batch_size'] = 1024
# test_param['proj_dim'] = 64
for seed in range(5):
    score = []
    model = CoMICERecommend(item_name, sparse_features, dense_features, seed=seed, k=k, **params)
    model.fit(train.copy(), valid.copy())
    score.append(model.score_test(test.copy(), methods=['auc', 'logloss', 'hr_k', 'ndcg_k']))
    model = MaskCoMICE(item_name, sparse_features, dense_features, seed=seed, k=k, **params)
    model.fit(train.copy(), valid.copy())
    score.append(model.score_test(test.copy(), methods=['auc', 'logloss', 'hr_k', 'ndcg_k']))
    model = StandardCoMICE(item_name, sparse_features, dense_features, seed=seed, k=k, **params)
    model.fit(train.copy(), valid.copy())
    score.append(model.score_test(test.copy(), methods=['auc', 'logloss', 'hr_k', 'ndcg_k']))
    all.append(score)
print(all)
# for missing_rate in [0.1]:
#     train_miss = add_missing_values(train, missing_rate, feature_cols=sparse_features+dense_features, seed=42)
#     valid_miss = add_missing_values(valid, missing_rate, feature_cols=sparse_features+dense_features, seed=42)
#     test_miss = add_missing_values(test, missing_rate, feature_cols=sparse_features+dense_features, seed=42)
#     _, imputer = filling(train_miss, method='MICE_NB', seed=42)
#     train_filled = imputer.transform(train_miss.copy())
#     valid_filled = imputer.transform(valid_miss.copy())
#     test_filled = imputer.transform(test_miss.copy())
#     train_filled = round(train_filled, sparse_features)
#     valid_filled = round(valid_filled, sparse_features)
#     test_filled = round(test_filled, sparse_features)
#     model = WideDeepRecommend(item_name, sparse_features, dense_features, seed=42, k=k)
#     model.fit(train_filled)
#     score.append(model.score_test(test_filled, methods=['auc']))
#     model = CoMICERecommend(item_name, sparse_features, dense_features, seed=42, k=k, num_views=3, mice_method='MICE_NB', backbone='WideDeep', lambda_nce=1.0)
#     model.fit(train_miss)
#     score.append(model.score_test(test_miss, methods=['auc']))
    # model = XGBRecommend(item_name, sparse_features, dense_features, seed=seed, k=k)
    # model.fit(train_miss)
    # score.append(model.score_test(test_miss, methods=['auc']))
# print(score)