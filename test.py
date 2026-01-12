import numpy as np
import pandas as pd
from copy import deepcopy
# %% [新增工具函数] 注入人工缺失
# %% [新增实验函数] 鲁棒性测试
def test_robustness():
    # 实验设置
    mask_ratios = [0.0, 0.1, 0.3, 0.5]  # 分别增加 0%, 10%, 30%, 50% 的缺失
    target_backbone = CoMICE_Backbone  # 使用默认骨干，如 'DCN'

    all_raw_data = []

    print(f"Starting Robustness Test on Backbone: {target_backbone}")

    for data_type in datasets:
        # 加载原始数据
        train_org, valid_org, test_org, info = load(data_type, amount, train_ratio, val_ratio, is_dropna=False)

        # 加载参数
        param_file = root / data_type / (target_backbone + "_param.json")
        with open(param_file, 'r') as f:
            params = json.load(f)

        for ratio in mask_ratios:
            print(f"Dataset: {data_type} | Mask Ratio: {ratio}")

            for seed in seeds[:5]:  # 为了节省时间，种子数量可以适当减少，比如取前5个
                # 1. 注入缺失值 (对 Train/Valid/Test 都进行 Mask，模拟稀疏环境)
                # 注意：这里seed要变化，保证不同seed下的mask位置不同
                train_masked = inject_missingness(train_org, info['sparse_features'], info['dense_features'], ratio,
                                                  seed)
                valid_masked = inject_missingness(valid_org, info['sparse_features'], info['dense_features'], ratio,
                                                  seed)
                test_masked = inject_missingness(test_org, info['sparse_features'], info['dense_features'], ratio, seed)

                # --- A. 测试 Baseline (普通模型) ---
                # 获取填充数据 (Base模型通常只用单次插补)
                train_filled, valid_filled, test_filled = get_filled_data(train_masked, valid_masked, test_masked,
                                                                          info['sparse_features'], seed=seed)

                BaseModelClass = globals()[f"{target_backbone}Recommend"]
                base_model = BaseModelClass(info['item_name'], info['sparse_features'], info['dense_features'],
                                            seed=seed, k=3, **params)
                base_model.fit(train_filled.copy(), valid_filled.copy())
                b_score = base_model.score_test(test_filled.copy(), methods=metrics)

                # --- B. 测试 CoMICE ---
                # CoMICE 内部会处理多视图插补，所以传入 Masked 的原始数据（包含NaN）
                comice_model = CoMICERecommend(info['item_name'], info['sparse_features'], info['dense_features'],
                                               seed=seed, k=3, backbone=target_backbone, **params)
                # 注意：CoMICE.fit 内部逻辑是接收含NaN数据 -> 生成多视图 -> 训练
                comice_model.fit(train_masked.copy(), valid_masked.copy())
                # 测试时 CoMICE 也会进行插补
                c_score = comice_model.score_test(test_masked.copy(), methods=metrics)

                # 记录结果
                for i, metric in enumerate(metrics):
                    all_raw_data.append({
                        'Dataset': data_type,
                        'Mask_Ratio': ratio,
                        'Seed': seed,
                        'Metric': metric,
                        'Base_Score': b_score[i],
                        'CoMICE_Score': c_score[i],
                        'Improvement': c_score[i] - b_score[i]
                    })

    # 保存结果
    df_raw = pd.DataFrame(all_raw_data)
    with pd.ExcelWriter('experiment.xlsx', engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        df_raw.to_excel(writer, sheet_name='Robustness_Test')
    print("Robustness experiment finished.")
# %% 在 main 中调用
if __name__ == "__main__":
    # test_Perf()
    test_robustness()