import pandas as pd
import json
from src.models import *
from src.utils import load, get_filled_data
from pathlib import Path
# %%
datasets = ['AWM', 'HIP', 'VID']
NN_MODELS = ['DCN', 'DCNv2', 'DeepFM', 'WideDeep', 'FiBiNET', 'AutoInt']
TREE_MODELS = ['RF', 'XGB', 'LGBM', 'CatB']
STATISTIC_MODELS = ['LR', 'KNN', 'NB']
CoMICE_Backbone = 'DCN'
metrics = ['auc', 'logloss', 'hr_k', 'ndcg_k']
imputers = ['MICE_NB', 'MICE_RF', 'MICE_LGBM']
seeds = list(range(15, 20))
amount = None
train_ratio = 0.7
val_ratio = 0.1
root = Path(__file__).parents[0]
# %%
def test_Perf():
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
                elif model_name in STATISTIC_MODELS + TREE_MODELS:
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
    with pd.ExcelWriter('experiment.xlsx', engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        df_raw.to_excel(writer, sheet_name='Perf_Data')
# %%
def test_model():
    all_raw_data = []
    for data_type in datasets:
        train, valid, test, info = load(data_type, amount, train_ratio, val_ratio, is_dropna=False)
        for model_name in NN_MODELS:
            param_file = root / data_type / (model_name + "_param.json")
            with open(param_file, 'r') as f:
                params = json.load(f)
            for seed in seeds:
                train_filled, valid_filled, test_filled = get_filled_data(train, valid, test, info['sparse_features'], seed=seed)
                ModelClass = globals()[f"{model_name}Recommend"]
                base_model = ModelClass(info['item_name'], info['sparse_features'], info['dense_features'], seed=seed, k=3, **params)
                base_model.fit(train_filled.copy(), valid_filled.copy())
                b_score = base_model.score_test(test_filled.copy(), methods=metrics)
                augment_model = AugmentRecommend(info['item_name'], info['sparse_features'], info['dense_features'], seed=seed, k=3, backbone=model_name, **params)
                augment_model.fit(train.copy(), valid.copy())
                a_score = augment_model.score_test(test.copy(), methods=metrics)
                comice_model = CoMICERecommend(info['item_name'], info['sparse_features'], info['dense_features'], seed=seed, k=3, backbone=model_name, **params)
                comice_model.fit(train.copy(), valid.copy())
                c_score = comice_model.score_test(test.copy(), methods=metrics)
                for i, metric in enumerate(metrics):
                    all_raw_data.append({
                        'Dataset': data_type,
                        'Backbone': model_name,
                        'Seed': seed,
                        'Metric': metric,
                        'BaseScore': b_score[i],
                        'AugmentScore': a_score[i],
                        'CoMICEScore': c_score[i],
                        'Diff': c_score[i] - a_score[i]
                    })
    df_raw = pd.DataFrame(all_raw_data)
    with pd.ExcelWriter('experiment.xlsx', engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        df_raw.to_excel(writer, sheet_name='Model_Comparison')
# %%
def test_imputer():
    all_raw_data = []
    for data_type in datasets:
        train, valid, test, info = load(data_type, amount, train_ratio, val_ratio, is_dropna=False)
        param_file = root / data_type / (CoMICE_Backbone + "_param.json")
        with open(param_file, 'r') as f:
            params = json.load(f)
        for imputer in imputers:
            for seed in seeds:
                train_filled, valid_filled, test_filled = get_filled_data(train, valid, test, info['sparse_features'], method=imputer, seed=seed)
                ModelClass = globals()[f"{CoMICE_Backbone}Recommend"]
                base_model = ModelClass(info['item_name'], info['sparse_features'], info['dense_features'], seed=seed, k=3, **params)
                base_model.fit(train_filled.copy(), valid_filled.copy())
                b_score = base_model.score_test(test_filled.copy(), methods=metrics)
                augment_model = AugmentRecommend(info['item_name'], info['sparse_features'], info['dense_features'], seed=seed, k=3, backbone=CoMICE_Backbone, mice_method=imputer, **params)
                augment_model.fit(train.copy(), valid.copy())
                a_score = augment_model.score_test(test.copy(), methods=metrics)
                comice_model = CoMICERecommend(info['item_name'], info['sparse_features'], info['dense_features'], seed=seed, k=3, backbone=CoMICE_Backbone, mice_method=imputer, **params)
                comice_model.fit(train.copy(), valid.copy())
                c_score = comice_model.score_test(test.copy(), methods=metrics)
                for i, metric in enumerate(metrics):
                    all_raw_data.append({
                        'Dataset': data_type,
                        'imputer': imputer,
                        'Seed': seed,
                        'Metric': metric,
                        'BaseScore': b_score[i],
                        'AugmentScore': a_score[i],
                        'CoMICEScore': c_score[i],
                        'Diff': c_score[i] - a_score[i]
                    })
    df_raw = pd.DataFrame(all_raw_data)
    with pd.ExcelWriter('experiment.xlsx', engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        df_raw.to_excel(writer, sheet_name='Imputer_Comparison')
# %%
def test_SSL():
    all_raw_data = []
    for data_type in datasets:
        train, valid, test, info = load(data_type, amount, train_ratio, val_ratio, is_dropna=False)
        for model_name in NN_MODELS:
            param_file = root / data_type / (model_name + "_param.json")
            with open(param_file, 'r') as f:
                params = json.load(f)
            CE_Lose_param = {**params, 'lambda_ce': 1.0, 'lambda_nce': 0.0}
            CE_NCE_Lose_param = {**params, 'lambda_ce': 1.0, 'lambda_nce': 1.0}
            for seed in seeds:
                CE_model = CoMICERecommend(info['item_name'], info['sparse_features'], info['dense_features'], seed=seed, k=3, backbone=model_name, **CE_Lose_param)
                CE_model.fit(train.copy(), valid.copy())
                CE_score = CE_model.score_test(test.copy(), methods=metrics)
                CE_NCE_model = CoMICERecommend(info['item_name'], info['sparse_features'], info['dense_features'], seed=seed, k=3, backbone=model_name, **CE_NCE_Lose_param)
                CE_NCE_model.fit(train.copy(), valid.copy())
                CE_NCE_score = CE_NCE_model.score_test(test.copy(), methods=metrics)
                for i, metric in enumerate(metrics):
                    all_raw_data.append({
                        'Dataset': data_type,
                        'Backbone': model_name,
                        'Seed': seed,
                        'Metric': metric,
                        'CE_score': CE_score[i],
                        'CE_NCE_score': CE_NCE_score[i],
                        'Diff': CE_NCE_score[i] - CE_score[i]
                    })
    df_raw = pd.DataFrame(all_raw_data)
    with pd.ExcelWriter('experiment.xlsx', engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        df_raw.to_excel(writer, sheet_name='SSL_Comparison')
# %%
if __name__ == "__main__":
    test_SSL()