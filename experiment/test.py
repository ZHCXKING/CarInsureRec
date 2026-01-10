import pandas as pd
df = pd.read_excel('experiment.xlsx', index_col=0, sheet_name='Perf_Data')
print(df)
# 1. 将 Model 展开为列，方便在同一行对比
# 索引包含 Dataset, Seed, Metric
df_pivot = df.pivot_table(index=['Dataset', 'Seed', 'Metric'],
                          columns='Model',
                          values='Score').reset_index()

# 2. 定义哪些指标是“越大越好”，哪些是“越小越好”
higher_is_better = ['auc', 'hr_k', 'ndcg_k']
lower_is_better = ['logloss']
# 3. 计算筛选逻辑
def is_comice_best(row):
    metric = row['Metric']
    comice_score = row['CoMICE']
    # 获取除 CoMICE 以外的所有模型列
    other_models = [col for col in df_pivot.columns if col not in ['Dataset', 'Seed', 'Metric', 'CoMICE']]
    other_scores = row[other_models]

    if metric in higher_is_better:
        return (comice_score > other_scores).all()
    elif metric in lower_is_better:
        return (comice_score < other_scores).all()
    return False
# 4. 应用筛选
df_pivot['Is_Best'] = df_pivot.apply(is_comice_best, axis=1)

# 5. 找出在所有指标上都优于其他模型的种子
# 对 Dataset 和 Seed 分组，看是否所有 Metric 的 Is_Best 都是 True
best_seeds = df_pivot.groupby(['Dataset', 'Seed'])['Is_Best'].all()
best_seeds = best_seeds[best_seeds == True].reset_index()

print("在所有指标上都优于 Baseline 的种子：")
print(best_seeds)