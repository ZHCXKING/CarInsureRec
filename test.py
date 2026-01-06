import pandas as pd
import numpy as np
import json
from scipy import stats
from src.models import *
from src.utils import load, get_filled_data
from pathlib import Path
datasets = ['AWM', 'HIP', 'VID']
NN_MODELS = ['DCN', 'DCNv2', 'DeepFM', 'WideDeep', 'FiBiNET', 'AutoInt']
TREE_MODELS = ['RF', 'XGB', 'LGBM', 'CatB']
STATISTIC_MODELS = ['LR', 'KNN', 'NB']  # Modified here
metrics = ['auc', 'logloss', 'mrr_k', 'recall_k', 'ndcg_k']
seeds = list(range(0, 5))
amount = None
train_ratio = 0.7
val_ratio = 0.1

root = Path(__file__).parents[0]
def test():
    final_report_data = []

    for data_type in datasets:

        # 加载原始数据
        train, valid, test, info = load(data_type, amount, train_ratio, val_ratio, is_dropna=False)

        for model_name in NN_MODELS + TREE_MODELS + STATISTIC_MODELS:
            baseline_results = []
            comice_results = []

            # --- 2. 性能优化：在 Seed 循环外加载模型 ---
            model_file = root / data_type / 'Original_model' / (model_name + ".pth")
            if not model_file.exists():
                print(f"  [Warning] {model_file} not found. Skipping.")
                continue

            try:
                # 动态获取类并加载
                ModelClass = globals()[f"{model_name}Recommend"]
                base_model = ModelClass.load(str(model_file))
            except Exception as e:
                print(f"  [Error] Failed to load {model_name}: {e}")
                continue

            # 如果是 NN 模型，预加载 CoMICE 模型
            comice_model = None
            if model_name in NN_MODELS:
                CoMICE_file = root / data_type / 'Original_model' / ('CoMICE_' + model_name + ".pth")
                if CoMICE_file.exists():
                    comice_model = CoMICERecommend.load(str(CoMICE_file))

            # --- 3. Seed 循环只负责生成数据和预测 ---
            for seed in seeds:
                # 生成当前 Seed 下的填充数据 (用于测试 Base 模型对插补的敏感度)
                # 注意：这里我们只关心 test_filled
                _, _, test_filled = get_filled_data(train, valid, test, info['sparse_features'], seed=seed)

                # A. 测试 Base 模型
                # 统一使用 test_filled。因为之前的 original_train 中，
                # 无论是 Tree 还是 NN 都是用 train_filled 训练的。
                b_score = base_model.score_test(test_filled.copy(), methods=metrics)
                baseline_results.append(b_score)

                # B. 测试 CoMICE 模型
                if comice_model:
                    # CoMICE 直接处理含 NaN 的原始数据
                    # 它的结果通常对外部 Seed 不敏感（除非开启了 Inference Dropout）
                    # 但为了对齐数据结构，我们还是每次记录
                    c_score = comice_model.score_test(test.copy(), methods=metrics)
                    comice_results.append(c_score)
                else:
                    comice_results.append([0] * len(metrics))

            # --- 4. 统计结果 ---
            baseline_results = np.array(baseline_results)
            comice_results = np.array(comice_results)

            for i, metric in enumerate(metrics):
                b_vals = baseline_results[:, i]
                c_vals = comice_results[:, i]

                b_mean = np.mean(b_vals)
                b_std = np.std(b_vals, ddof=1)  # 加上标准差更有说服力
                c_mean = np.mean(c_vals)

                row = {
                    'Dataset': data_type,
                    'Base_Model': model_name,
                    'Metric': metric,
                    'Base_Mean': b_mean,
                    'Base_Std': b_std,  # 推荐加上
                    'CoMICE_Mean': c_mean,
                    'ValueChange': c_mean - b_mean,
                }
                final_report_data.append(row)

    df_result = pd.DataFrame(final_report_data)
    return df_result