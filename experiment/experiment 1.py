import pandas as pd
import numpy as np
import json
from src.models import *
from src.utils import load, get_filled_data
from pathlib import Path
# %%
datasets = ['AWM', 'HIP', 'VID']
NN_MODELS = ['DCN', 'DCNv2', 'DeepFM', 'WideDeep', 'FiBiNET', 'AutoInt']
TREE_MODELS = ['RF', 'XGB', 'LGBM', 'CatB']
STATISTIC_MODELS = ['LR', 'KNN', 'NB']
CoMICE_Backbone = 'AutoInt'
metrics = ['auc', 'logloss', 'hr_k', 'ndcg_k']
seeds = list(range(20, 40))
amount = None
train_ratio = 0.7
val_ratio = 0.1
root = Path(__file__).parents[0]
# %%
def test():
    all_raw_data = []
    for data_type in datasets:
        train, valid, test, info = load(data_type, amount, train_ratio, val_ratio, is_dropna=False)
        for model_name in NN_MODELS + TREE_MODELS + STATISTIC_MODELS + ['CoMICE']:
            if model_name == 'CoMICE':
                param_file = root / data_type / (CoMICE_Backbone + "_param.json")
            else:
                param_file = root / data_type / (model_name + "_param.json")
            with open(param_file, 'r') as f:
                params = json.load(f)
            for seed in seeds:
                train_filled, valid_filled, test_filled = get_filled_data(train, valid, test, info['sparse_features'], seed=seed)
                ModelClass = globals()[f"{model_name}Recommend"]
                if model_name == 'CoMICE':
                    base_model = CoMICERecommend(info['item_name'], info['sparse_features'], info['dense_features'], seed=seed, k=3, backbone=CoMICE_Backbone, **params)
                else:
                    base_model = ModelClass(info['item_name'], info['sparse_features'], info['dense_features'], seed=seed, k=3, **params)
                # %%
                if model_name in NN_MODELS:
                    base_model.fit(train_filled.copy(), valid_filled.copy())
                    b_score = base_model.score_test(test_filled.copy(), methods=metrics)
                elif model_name in TREE_MODELS:
                    base_model.fit(train.copy())
                    b_score = base_model.score_test(test.copy(), methods=metrics)
                elif model_name in STATISTIC_MODELS:
                    base_model.fit(train_filled.copy())
                    b_score = base_model.score_test(test_filled.copy(), methods=metrics)
                else:
                    base_model.fit(train.copy(), valid.copy())
                    b_score = base_model.score_test(test.copy(), methods=metrics)
                # %%
                for i, metric in enumerate(metrics):
                    all_raw_data.append({
                        'Dataset': data_type,
                        'Model': model_name,
                        'Seed': seed,
                        'Metric': metric,
                        'Score': b_score[i]
                    })
    df_raw = pd.DataFrame(all_raw_data)
    with pd.ExcelWriter('experiment 1.xlsx', engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        df_raw.to_excel(writer, sheet_name='Raw_Data')
# %%
if __name__ == "__main__":
    test()