# %%
import pandas as pd
import numpy as np
import json
from pathlib import Path
from scipy import stats  # 引入统计库
from src.utils import load
from src.models import *
root = Path(__file__).parents[0]
datasets = ['AWM', 'HIP', 'VID']
models = ['DCNv2', 'DeepFM', 'WideDeep', 'FiBiNET', 'Hybrid', 'AutoInt']
metrics = ['auc', 'logloss', 'mrr_k', 'recall_k', 'ndcg_k']
seeds = list(range(5, 10))
amount = 10000
train_ratio = 0.6
val_ratio = 0.1
# %%
def test_seeds(seeds, model_name, info, params, train, valid, test, metrics):
    baseline_results = []
    comice_results = []

    # 为了防止 CoMICE 缺少特有参数，我们可以加载 CoMICE 的默认参数进行 update
    # 如果你的 params 字典里已经包含了 CoMICE 所需的参数 (如 lambda_nce)，则不需要这一步
    # 这里假设 params 仅包含 model_name 的参数，因此我们可能需要补充 CoMICE 的默认值
    comice_params = params.copy()
    comice_defaults = {
        'lambda_nce': 1.0, 'lambda_fcl': 1.0, 'temperature': 0.1,
        'mask_prob': 0.2, 'noise_std': 0.05
    }
    for k, v in comice_defaults.items():
        if k not in comice_params:
            comice_params[k] = v

    for seed in seeds:
        # 1. 训练 Baseline Model
        ModelClass = globals()[f"{model_name}Recommend"]
        model = ModelClass(info['item_name'], info['sparse_features'], info['dense_features'], seed=seed, k=5, **params)
        model.fit(train.copy())
        b_score = model.score_test(test.copy(), methods=metrics)
        baseline_results.append(b_score)

        # 2. 训练 CoMICE Model (传入 backbone 参数)
        # 注意：这里假设 CoMICERecommend 的 __init__ 或 kwargs 能处理 backbone
        comice_model = CoMICERecommend(
            info['item_name'], info['sparse_features'], info['dense_features'],
            seed=seed, k=5, backbone=model_name, **comice_params
        )
        comice_model.fit(train.copy())
        c_score = comice_model.score_test(test.copy(), methods=metrics)
        comice_results.append(c_score)

    return np.array(baseline_results), np.array(comice_results)
# %%
def test():
    final_report_data = []

    for data_type in datasets:
        print(f"Processing Dataset: {data_type}")
        train, valid, test, info = load(data_type, amount, train_ratio, val_ratio)

        for model_name in models:
            print(f"  Evaluating Model: {model_name} vs CoMICE({model_name})...")

            # 加载该模型的最佳参数
            param_path = root / data_type / (model_name + "_params.json")
            if param_path.exists():
                with open(param_path, 'r') as f:
                    params = json.load(f)
            else:
                print(f"    Warning: Params file not found for {model_name}, using defaults.")
                params = {'batch_size': 1024, 'epochs': 10}  # 极其简单的默认值防止崩溃

            # 获取所有种子的结果
            base_scores, comice_scores = test_seeds(seeds, model_name, info, params, train, valid, test, metrics)

            # 对每个指标进行统计分析
            for i, metric in enumerate(metrics):
                b_vals = base_scores[:, i]
                c_vals = comice_scores[:, i]

                # 1. 计算均值和标准差
                b_mean = np.mean(b_vals)
                b_std = np.std(b_vals, ddof=1)
                c_mean = np.mean(c_vals)
                c_std = np.std(c_vals, ddof=1)

                # 2. 计算配对样本 T 检验 (Paired T-Test)
                # 使用配对检验是因为使用了相同的 seed，排除了初始化的随机性干扰
                if np.allclose(b_vals, c_vals):
                    p_value = 1.0
                    t_stat = 0.0
                else:
                    t_stat, p_value = stats.ttest_rel(c_vals, b_vals)

                # 3. 判断是否变好 (Is Better?)
                # logloss 越小越好，其他指标越大越好
                if metric == 'logloss':
                    is_better = c_mean < b_mean
                else:
                    is_better = c_mean > b_mean

                # 4. 判断显著性 (Significant?)
                # p < 0.05 且 CoMICE 均值更优
                is_significant = (p_value < 0.05) and is_better

                row = {
                    'Dataset': data_type,
                    'Backbone': model_name,
                    'Metric': metric,
                    # Baseline Info
                    'Base_Mean': b_mean,
                    'Base_Std': b_std,
                    # CoMICE Info
                    'CoMICE_Mean': c_mean,
                    'CoMICE_Std': c_std,
                    # Comparison
                    'Improvement': (c_mean - b_mean) if metric != 'logloss' else (b_mean - c_mean),
                    'Is_Better': is_better,
                    'P_Value': p_value,
                    'Significant': is_significant
                }
                final_report_data.append(row)

    df_result = pd.DataFrame(final_report_data)

    # 格式化输出，保留4位小数
    numeric_cols = ['Base_Mean', 'Base_Std', 'CoMICE_Mean', 'CoMICE_Std', 'Improvement', 'P_Value']
    df_result[numeric_cols] = df_result[numeric_cols].round(4)

    return df_result
if __name__ == '__main__':
    df = test()
    print(df)
    # df.to_csv("comice_comparison_results.csv", index=False)