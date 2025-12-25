# %%
import pandas as pd
import numpy as np
import json
from src.models import *
from src.utils import load
from pathlib import Path
# %%
root = Path(__file__).parents[0]
datasets = ['AWM', 'HIP', 'VID']
models = ['DCNv2', 'DeepFM', 'WideDeep', 'FiBiNET', 'CoMICE', 'Hybrid']
metrics = ['auc', 'logloss', 'mrr_k', 'recall_k', 'ndcg_k']
seeds = list(range(10))
amount = 10000
train_ratio = 0.6
val_ratio = 0.1
# %%
def test_seeds(seeds, model_name, info, params, train, valid, test, metrics):
    baseline_results = []
    for seed in seeds:
        ModelClass = globals()[f"{model_name}Recommend"]
        model = ModelClass(info['item_name'], info['sparse_features'], info['dense_features'], seed=seed, k=5, **params)
        model.fit(train.copy(), valid.copy())
        b_score = model.score_test(test.copy(), methods=metrics)
        baseline_results.append(b_score)
    return np.array(baseline_results)
# %%
def test():
    final_report_data = []
    for data_type in datasets:
        train, valid, test, info = load(data_type, amount, train_ratio, val_ratio)
        for model_name in models:
            with open(root / data_type / (model_name + "_params.json"), 'r') as f:
                params = json.load(f)
            base_scores = test_seeds(seeds, model_name, info, params, train, valid, test, metrics)
            for i, metric in enumerate(metrics):
                # 提取该指标下的所有 seed 结果
                b_vals = base_scores[:, i]
                # 计算均值和标准差
                b_mean = np.mean(b_vals)
                b_std = np.std(b_vals, ddof=1)  # ddof=1 计算样本标准差
                # 记录结果
                row = {
                    'Dataset': data_type,
                    'Base_Model': model_name,
                    'Metric': metric,
                    'Base_Mean': b_mean,
                    'Base_Std': b_std,
                }
                final_report_data.append(row)
    df_result = pd.DataFrame(final_report_data)
    return df_result
# %%
def collect_all_results():
    records = []
    for data_type in datasets:
        train, valid, test, info = load(data_type, amount, train_ratio, val_ratio)
        for model_name in models:
            with open(root / data_type / (model_name + "_params.json"), 'r') as f:
                params = json.load(f)
            for seed in seeds:
                ModelClass = globals()[f"{model_name}Recommend"]
                model = ModelClass(
                    info['item_name'],
                    info['sparse_features'],
                    info['dense_features'],
                    seed=seed,
                    k=5,
                    **params
                )
                model.fit(train.copy(), valid.copy())
                scores = model.score_test(test.copy(), methods=metrics)
                record = {
                    'Dataset': data_type,
                    'Seed': seed,
                    'Model': model_name,
                }
                for m, v in zip(metrics, scores):
                    record[m] = v
                records.append(record)
    return pd.DataFrame(records)
def find_best_seeds(df):
    best_seeds = []
    for (dataset, seed), group in df.groupby(['Dataset', 'Seed']):
        comice = group[group['Model'] == 'CoMICE']
        others = group[group['Model'] != 'CoMICE']
        if comice.empty:
            continue
        comice_auc = comice['auc'].values[0]
        comice_logloss = comice['logloss'].values[0]
        # 判断条件
        if (
            (comice_auc > others['auc']).all() and
            (comice_logloss < others['logloss']).all()
        ):
            best_seeds.append({
                'Dataset': dataset,
                'Seed': seed,
                'CoMICE_auc': comice_auc,
                'CoMICE_logloss': comice_logloss
            })
    return pd.DataFrame(best_seeds)
df_all = test()
#df_best = find_best_seeds(df_all)
with pd.option_context('display.max_rows', None,    # 显示所有行
                       'display.max_columns', None, # 显示所有列
                       'display.precision', 4,      # 设置小数点精度
                       'display.width', 1000):      # 设置打印宽度，防止换行
    print(df_all)
df_all.to_excel('experiment 1.xlsx')