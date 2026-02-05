import pandas as pd
import json
import itertools
import time
from src.models import *
from src.utils import load, get_filled_data, inject_missingness
from pathlib import Path
# %%
datasets = ['AWM', 'HIP', 'VID']
NN_MODELS = ['DCN', 'DCNv2', 'DeepFM', 'WideDeep', 'FiBiNET', 'AutoInt']
TREE_MODELS = ['RF', 'XGB', 'LGBM', 'CatB']
STATISTIC_MODELS = ['LR', 'NB']
CoMICE_Backbone = 'DCN'
metrics = ['auc', 'logloss', 'hr_k', 'ndcg_k']
mice_imputers = ['MICE_NB', 'MICE_RF', 'MICE_LGBM']
other_imputers = ['GAIN', 'MIWAE', 'KNN']
mask_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
lambda_list = [0.01, 0.1, 0.5, 1.0, 2.0]
temp_list = [0.01, 0.05, 0.1, 0.15, 0.2]
views_list = [1, 2, 3, 4, 5]
batchsizes_list = [256, 512, 1024, 2048, 4096]
mask_strategies = ['random', 'feature', 'noise']
seeds = list(range(0, 35))
amount = None
train_ratio = 0.7
val_ratio = 0.1
root = Path(__file__).parents[0]
# %%
def test_Perf():
    all_raw_data = []
    for data_type in datasets:
        train, valid, test, info = load(data_type, amount, train_ratio, val_ratio, is_dropna=False)
        with open(root / data_type / 'Seeds.json', 'r') as f:
            Seeds = json.load(f)
        for model_name in NN_MODELS + TREE_MODELS + STATISTIC_MODELS + ['CoMICE']:
            if model_name == 'CoMICE':
                param_file = root / data_type / (CoMICE_Backbone + "_param.json")
                Seeds['CoMICE'] = Seeds[CoMICE_Backbone]
            else:
                param_file = root / data_type / (model_name + "_param.json")
            with open(param_file, 'r') as f:
                params = json.load(f)
            for seed in Seeds[model_name]:
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
def test_NaRatio():
    all_raw_data = []
    for data_type in datasets:
        train, valid, test, info = load(data_type, amount, train_ratio, val_ratio, is_dropna=False)
        with open(root / data_type / 'Seeds.json', 'r') as f:
            Seeds = json.load(f)
        param_file = root / data_type / (CoMICE_Backbone + "_param.json")
        with open(param_file, 'r') as f:
            params = json.load(f)
        all_imputers = mice_imputers + other_imputers + ['CoMICE']
        for ratio, imputer, seed in itertools.product(mask_ratios, all_imputers, Seeds[CoMICE_Backbone]):
            train_mask = inject_missingness(train, info['sparse_features'], info['dense_features'], ratio, seed=seed)
            ModelClass = globals()[f"{CoMICE_Backbone}Recommend"]
            if imputer == 'CoMICE':
                base_model = CoMICERecommend(info['item_name'], info['sparse_features'], info['dense_features'], seed=seed, k=3, backbone=CoMICE_Backbone, **params)
                base_model.fit(train_mask.copy(), valid.copy())
                score = base_model.score_test(test.copy(), methods=metrics)
            else:
                train_filled, valid_filled, test_filled = get_filled_data(train_mask, valid, test, info['sparse_features'], method=imputer, seed=seed)
                base_model = ModelClass(info['item_name'], info['sparse_features'], info['dense_features'], seed=seed, k=3, **params)
                base_model.fit(train_filled.copy(), valid_filled.copy())
                score = base_model.score_test(test_filled.copy(), methods=metrics)
            for i, metric in enumerate(metrics):
                all_raw_data.append({
                    'Dataset': data_type,
                    'Ratio': ratio,
                    'Model': imputer,
                    'Seed': seed,
                    'Metric': metric,
                    'Score': score[i]
                })
    df_raw = pd.DataFrame(all_raw_data)
    with pd.ExcelWriter('experiment.xlsx', engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        df_raw.to_excel(writer, sheet_name='NaRatio_Data')
# %%
def test_model():
    all_raw_data = []
    for data_type in datasets:
        train, valid, test, info = load(data_type, amount, train_ratio, val_ratio, is_dropna=False)
        with open(root / data_type / 'Seeds.json', 'r') as f:
            Seeds = json.load(f)
        for model_name in NN_MODELS:
            param_file = root / data_type / (model_name + "_param.json")
            with open(param_file, 'r') as f:
                params = json.load(f)
            for seed in Seeds[model_name]:
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
        with open(root / data_type / 'Seeds.json', 'r') as f:
            Seeds = json.load(f)
        param_file = root / data_type / (CoMICE_Backbone + "_param.json")
        with open(param_file, 'r') as f:
            params = json.load(f)
        for imputer in mice_imputers:
            for seed in Seeds[CoMICE_Backbone]:
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
        with open(root / data_type / 'Seeds.json', 'r') as f:
            Seeds = json.load(f)
        for model_name in NN_MODELS:
            param_file = root / data_type / (model_name + "_param.json")
            with open(param_file, 'r') as f:
                params = json.load(f)
            CE_Lose_param = {**params, 'lambda_ce': 1.0, 'lambda_nce': 0.0}
            CE_NCE_Lose_param = {**params, 'lambda_ce': 1.0, 'lambda_nce': 1.0}
            for seed in Seeds[model_name]:
                CE_model = CoMICERecommend(info['item_name'], info['sparse_features'], info['dense_features'], seed=seed, k=3, backbone=model_name, **CE_Lose_param)
                CE_model.fit(train.copy(), valid.copy())
                CE_score = CE_model.score_test(test.copy(), methods=metrics)
                CE_NCE_model = StandardCoMICE(info['item_name'], info['sparse_features'], info['dense_features'], seed=seed, k=3, backbone=model_name, **CE_NCE_Lose_param)
                CE_NCE_model.fit(train.copy(), valid.copy())
                CE_NCE_score = CE_NCE_model.score_test(test.copy(), methods=metrics)
                CE_weight_NCE_model = CoMICERecommend(info['item_name'], info['sparse_features'], info['dense_features'], seed=seed, k=3, backbone=model_name, **CE_NCE_Lose_param)
                CE_weight_NCE_model.fit(train.copy(), valid.copy())
                CE_weight_NCE_score = CE_weight_NCE_model.score_test(test.copy(), methods=metrics)
                for i, metric in enumerate(metrics):
                    all_raw_data.append({
                        'Dataset': data_type,
                        'Backbone': model_name,
                        'Seed': seed,
                        'Metric': metric,
                        'CE_score': CE_score[i],
                        'CE_NCE_score' : CE_NCE_score[i],
                        'CE_weight_NCE_score': CE_weight_NCE_score[i],
                    })
    df_raw = pd.DataFrame(all_raw_data)
    with pd.ExcelWriter('experiment.xlsx', engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        df_raw.to_excel(writer, sheet_name='SSL_Comparison')
# %%
def test_mask_ablation():
    all_raw_data = []
    for data_type in datasets:
        train, valid, test, info = load(data_type, amount, train_ratio, val_ratio, is_dropna=False)
        with open(root / data_type / 'Seeds.json', 'r') as f:
            Seeds = json.load(f)
        param_file = root / data_type / (CoMICE_Backbone + "_param.json")
        with open(param_file, 'r') as f:
            params = json.load(f)
        for seed in Seeds[CoMICE_Backbone]:
            model = CoMICERecommend(info['item_name'], info['sparse_features'], info['dense_features'], seed=seed, k=3, backbone=CoMICE_Backbone, **params)
            model.fit(train.copy(), valid.copy())
            score = model.score_test(test.copy(), methods=metrics)
            for i, metric in enumerate(metrics):
                all_raw_data.append({
                    'Dataset': data_type,
                    'View_Type': 'MICE_Multiple_Views',
                    'Seed': seed,
                    'Metric': metric,
                    'Score': score[i]
                })
        for strategy in mask_strategies:
            current_params = params.copy()
            current_params['mask_type'] = strategy
            current_params['num_views'] = 3
            for seed in Seeds[CoMICE_Backbone]:
                model = MaskCoMICE(info['item_name'], info['sparse_features'], info['dense_features'], seed=seed, k=3, backbone=CoMICE_Backbone, **current_params)
                model.fit(train.copy(), valid.copy())
                score = model.score_test(test.copy(), methods=metrics)
                for i, metric in enumerate(metrics):
                    all_raw_data.append({
                        'Dataset': data_type,
                        'View_Type': f'Mask_{strategy}',
                        'Seed': seed,
                        'Metric': metric,
                        'Score': score[i]
                    })
    df_raw = pd.DataFrame(all_raw_data)
    with pd.ExcelWriter('experiment.xlsx', engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        df_raw.to_excel(writer, sheet_name='Mask_Ablation_Study')
# %%
def test_sensitivity_heatmap():
    all_raw_data = []
    for data_type in datasets:
        train, valid, test, info = load(data_type, amount, train_ratio, val_ratio, is_dropna=False)
        with open(root / data_type / 'Seeds.json', 'r') as f:
            Seeds = json.load(f)
        param_file = root / data_type / (CoMICE_Backbone + "_param.json")
        with open(param_file, 'r') as f:
            params = json.load(f)
        combinations = list(itertools.product(lambda_list, temp_list))
        for lam, temp in combinations:
            current_params = params.copy()
            current_params['lambda_nce'] = lam
            current_params['temperature'] = temp
            for seed in Seeds[CoMICE_Backbone]:
                model = CoMICERecommend(info['item_name'], info['sparse_features'], info['dense_features'], seed=seed, k=3, backbone=CoMICE_Backbone, **current_params)
                model.fit(train.copy(), valid.copy())
                score = model.score_test(test.copy(), methods=metrics)
                for i, metric in enumerate(metrics):
                    all_raw_data.append({
                        'Dataset': data_type,
                        'lambda_nce': lam,
                        'temperature': temp,
                        'Seed': seed,
                        'Metric': metric,
                        'Score': score[i]
                    })
    df_raw = pd.DataFrame(all_raw_data)
    with pd.ExcelWriter('experiment.xlsx', engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        df_raw.to_excel(writer, sheet_name='sensitivity_heatmap')
# %%
def test_views_tradeoff():
    all_raw_data = []
    for data_type in datasets:
        train, valid, test, info = load(data_type, amount, train_ratio, val_ratio, is_dropna=False)
        with open(root / data_type / 'Seeds.json', 'r') as f:
            Seeds = json.load(f)
        param_file = root / data_type / (CoMICE_Backbone + "_param.json")
        with open(param_file, 'r') as f:
            params = json.load(f)
        for n_views in views_list:
            current_params = params.copy()
            current_params['num_views'] = n_views
            for seed in Seeds[CoMICE_Backbone]:
                model = CoMICERecommend(info['item_name'], info['sparse_features'], info['dense_features'], seed=seed, k=3, backbone=CoMICE_Backbone, **current_params)
                start_time = time.time()
                model.fit(train.copy(), valid.copy())
                end_time = time.time()
                training_time = end_time - start_time
                score = model.score_test(test.copy(), methods=metrics)
                for i, metric in enumerate(metrics):
                    all_raw_data.append({
                        'Dataset': data_type,
                        'num_views': n_views,
                        'Seed': seed,
                        'Time_Sec': training_time,
                        'Metric': metric,
                        'Score': score[i]
                    })
    df_raw = pd.DataFrame(all_raw_data)
    with pd.ExcelWriter('experiment.xlsx', engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        df_raw.to_excel(writer, sheet_name='views_tradeoff')
# %%
def test_batchsizes_tradeoff():
    all_raw_data = []
    for data_type in datasets:
        train, valid, test, info = load(data_type, amount, train_ratio, val_ratio, is_dropna=False)
        with open(root / data_type / 'Seeds.json', 'r') as f:
            Seeds = json.load(f)
        param_file = root / data_type / (CoMICE_Backbone + "_param.json")
        with open(param_file, 'r') as f:
            params = json.load(f)
        for batchsize in batchsizes_list:
            current_params = params.copy()
            current_params['batch_size'] = batchsize
            for seed in Seeds[CoMICE_Backbone]:
                model = CoMICERecommend(info['item_name'], info['sparse_features'], info['dense_features'], seed=seed, k=3, backbone=CoMICE_Backbone, **current_params)
                start_time = time.time()
                model.fit(train.copy(), valid.copy())
                end_time = time.time()
                training_time = end_time - start_time
                score = model.score_test(test.copy(), methods=metrics)
                for i, metric in enumerate(metrics):
                    all_raw_data.append({
                        'Dataset': data_type,
                        'batchsize': batchsize,
                        'Seed': seed,
                        'Time_Sec': training_time,
                        'Metric': metric,
                        'Score': score[i]
                    })
    df_raw = pd.DataFrame(all_raw_data)
    with pd.ExcelWriter('experiment.xlsx', engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        df_raw.to_excel(writer, sheet_name='batchsizes_tradeoff')
# %%
if __name__ == "__main__":
    test_Perf()
    test_sensitivity_heatmap()
    test_batchsizes_tradeoff()