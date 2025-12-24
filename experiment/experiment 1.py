# %%
import pandas as pd
import numpy as np
import json
from scipy import stats
from src.models import *
from src.utils import load
from pathlib import Path
# %%
def test_seeds(seeds, model_name, info, params, train, test, metrics):
    baseline_results = []
    comice_results = []
    for seed in seeds:
        ModelClass = globals()[f"{model_name}Recommend"]
        model = ModelClass(info['item_name'], info['sparse_features'], info['dense_features'], seed=42, k=5, **params)
        model.fit(train.copy(), valid.copy())
        b_score = model.score_test(test.copy(), methods=metrics)
        baseline_results.append(b_score)
        Comodel = CoMICERecommend(info['item_name'], info['sparse_features'], info['dense_features'], seed=seed, k=5, backbone=model_name, **params)
        Comodel.fit(train.copy())
        c_score = Comodel.score_test(test.copy(), methods=metrics)
        comice_results.append(c_score)
    return np.array(baseline_results), np.array(comice_results)
# %%
root = Path(__file__).parents[0]
datasets = ['AWM', 'HIP', 'VID']
models = ['DCNv2', 'DeepFM', 'WideDeep', 'AutoInt', 'FiBiNET', 'Hybrid']
metrics = ['auc', 'logloss', 'mrr_k', 'recall_k', 'ndcg_k']
seeds = list(range(2))
amount = 10000
train_ratio = 0.6
val_ratio = 0.1
final_report_data = []
for data_type in datasets:
    train, valid, test, info = load(data_type, amount, train_ratio, val_ratio)
    for model_name in models:
        with open(root / data_type / (model_name + "_params.json"), 'r') as f:
            params = json.load(f)
        params['epochs'] = 500
        base_scores, comice_scores = test_seeds(seeds, model_name, info, params, train, test, metrics)
        for i, metric in enumerate(metrics):
            # 提取该指标下的所有 seed 结果
            b_vals = base_scores[:, i]
            c_vals = comice_scores[:, i]
            # 计算均值和标准差
            b_mean = np.mean(b_vals)
            b_std = np.std(b_vals, ddof=1)  # ddof=1 计算样本标准差
            c_mean = np.mean(c_vals)
            c_std = np.std(c_vals, ddof=1)
            # 计算 p-value (配对 t 检验)
            t_stat, p_value = stats.ttest_rel(b_vals, c_vals)
            # 记录结果
            row = {
                'Dataset': data_type,
                'Base_Model': model_name,
                'Metric': metric,
                'Base_Mean': b_mean,
                'Base_Std': b_std,
                'CoMICE_Mean': c_mean,
                'CoMICE_Std': c_std,
                'Change value': c_mean - b_mean,  # 提升值（注意：Logloss 负值代表提升）
                'P_Value': p_value
            }
            final_report_data.append(row)
df_result = pd.DataFrame(final_report_data)
df_result.to_excel('experiment 1.xlsx')