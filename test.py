import time
import numpy as np
# 定义要测试的视图数量列表
# 1: 普通填补, >1: MICE 多视图
views_list = [1, 2, 3, 4, 5]
def test_num_views_tradeoff():
    all_raw_data = []

    # 为了节省时间，只跑 3 个种子取平均
    tradeoff_seeds = seeds[:3]

    total_runs = len(datasets) * len(views_list) * len(tradeoff_seeds)
    print(f"Total runs planned: {total_runs}")

    current_run = 0

    for data_type in datasets:
        print(f"Processing {data_type}...")
        # CoMICE 内部会处理填补，所以这里加载原始数据
        train, valid, test, info = load(data_type, amount, train_ratio, val_ratio, is_dropna=False)

        # 加载默认参数
        param_file = root / data_type / (CoMICE_Backbone + "_param.json")
        with open(param_file, 'r') as f:
            base_params = json.load(f)

        for n_views in views_list:
            # 更新视图数量参数
            current_params = base_params.copy()
            current_params['num_views'] = n_views

            for seed in tradeoff_seeds:
                current_run += 1
                print(f"[{current_run}/{total_runs}] Dataset:{data_type}, Views:{n_views}, Seed:{seed}")

                # 初始化模型
                model = CoMICERecommend(
                    info['item_name'], info['sparse_features'], info['dense_features'],
                    seed=seed, k=3, backbone=CoMICE_Backbone, **current_params
                )

                # --- 计时开始 ---
                start_time = time.time()

                # 训练 (fit 内部包含 MICE 生成视图的过程)
                model.fit(train.copy(), valid.copy())

                # --- 计时结束 ---
                end_time = time.time()
                training_duration = end_time - start_time

                # 评估
                score = model.score_test(test.copy(), methods=metrics)

                # 记录结果
                for i, metric in enumerate(metrics):
                    all_raw_data.append({
                        'Dataset': data_type,
                        'Num_Views': n_views,
                        'Seed': seed,
                        'Metric': metric,
                        'Score': score[i],
                        'Time_Sec': training_duration  # 记录时间
                    })

    # 保存结果
    df_raw = pd.DataFrame(all_raw_data)
    with pd.ExcelWriter('experiment.xlsx', engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        df_raw.to_excel(writer, sheet_name='views_tradeoff')
    print("Num_Views tradeoff experiment finished and saved.")
if __name__ == "__main__":
    test_num_views_tradeoff()