# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import math
# %%
STYLE_PARAMS = {
    'color_perf': 'tab:blue',
    'color_time': 'tab:red',
    'marker_perf': 'o',
    'marker_time': 's',
    'fontsize_title': 14,
    'fontsize_label': 15,
    'legend_pos': (0.5, 0.98),
    'rect_layout': (0, 0, 1, 0.94)
}
# 辅助函数：根据行列数计算合适的画布大小
def get_figsize(nrows, ncols):
    return (5 * ncols + 1, 4.5 * nrows)
# %%
def draw_NaRatio():
    df = pd.read_excel("experiment.xlsx", sheet_name="NaRatio_Data")
    names = {'MICE_NB': 'MICE(NB)', 'MICE_RF': 'MissForest', 'MICE_LGBM': 'MICE(LGBM)'}
    df = df[(df['Model'] != 'MIWAE')]
    df = df[df['Ratio'].isin([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])]
    df['Model'] = df['Model'].replace(names)
    datasets = ["AWM"]
    metrics = ["hr_k", "ndcg_k"]
    metric_names = {"hr_k": "HR", "ndcg_k": "NDCG"}
    nrows, ncols = len(metrics), len(datasets)
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=get_figsize(nrows, ncols),
        squeeze=False
    )
    for row, metric in enumerate(metrics):
        for col, dataset in enumerate(datasets):
            ax = axes[row, col]
            df_sub = df[(df["Metric"] == metric) & (df["Dataset"] == dataset)]

            sns.lineplot(data=df_sub, x="Ratio", y="Score", hue="Model",
                         marker="o", errorbar='se', linewidth=2, ax=ax)
            ax.set_title("")
            if col == 0:
                ax.set_ylabel(metric_names[metric], fontsize=STYLE_PARAMS['fontsize_label'], fontweight='bold')
            else:
                ax.set_ylabel("")
            ax.set_xlabel("Inject Missing Ratio")
            if ax.get_legend(): ax.get_legend().remove()
    handles, labels = axes[0, 0].get_legend_handles_labels()
    n_cols_legend = math.ceil(len(labels) / 2)
    fig.legend(handles, labels,
               loc="upper center",
               ncol=n_cols_legend,
               frameon=False,
               fontsize=13,
               bbox_to_anchor=(0.5, 1.02))
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig('Inject_Missing_Ratio.pdf', bbox_inches='tight')
    plt.show()
# %%
def draw_heatmap():
    df = pd.read_excel("experiment.xlsx", sheet_name="sensitivity_heatmap")
    datasets = ['AWM', 'VID']
    metrics = ['ndcg_k']
    metric_names = {"hr_k": "HR", "ndcg_k": "NDCG"}
    nrows, ncols = len(metrics), len(datasets)
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=get_figsize(nrows, ncols),
        squeeze=False
    )
    for row, metric in enumerate(metrics):
        for col, dataset in enumerate(datasets):
            ax = axes[row, col]
            df_sub = df[(df['Metric'] == metric) & (df['Dataset'] == dataset)]
            df_pivot = df_sub.groupby(['lambda_nce', 'temperature'])['Score'].mean().reset_index()
            df_pivot = df_pivot.pivot(index='temperature', columns='lambda_nce', values='Score')
            sns.heatmap(df_pivot, annot=True, fmt=".4f", cmap="YlGnBu", ax=ax)
            if row == 0:
                ax.set_title(f'Dataset: {dataset}', fontsize=STYLE_PARAMS['fontsize_title'], pad=10)
            ax.set_xlabel(r'$\lambda_{nce}$')
            label_text = f"Temperature"
            ax.set_ylabel(label_text, fontsize=STYLE_PARAMS['fontsize_label'])

    plt.tight_layout(rect=STYLE_PARAMS['rect_layout'])
    plt.savefig('Sensitivity_Heatmap.pdf', bbox_inches='tight')
    plt.show()
# %%
def draw_views_tradeoff():
    df = pd.read_excel("experiment.xlsx", sheet_name="views_tradeoff")
    datasets = ["AWM", "VID"]
    metrics = ["hr_k", "ndcg_k"]
    metric_names = {"hr_k": "HR", "ndcg_k": "NDCG"}
    nrows, ncols = len(metrics), len(datasets)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=get_figsize(nrows, ncols), squeeze=False)
    for row, metric in enumerate(metrics):
        for col, dataset in enumerate(datasets):
            ax = axes[row, col]
            df_sub = df[(df["Metric"] == metric) & (df["Dataset"] == dataset)]
            sns.lineplot(data=df_sub, x="num_views", y="Score", marker=STYLE_PARAMS['marker_perf'],
                         color=STYLE_PARAMS['color_perf'], linewidth=2.5, label="Performance", ax=ax)
            ax.set_ylabel(metric_names[metric], color=STYLE_PARAMS['color_perf'], fontweight='bold', fontsize=STYLE_PARAMS['fontsize_label'])
            ax2 = ax.twinx()
            sns.lineplot(data=df_sub, x="num_views", y="Time_Sec", marker=STYLE_PARAMS['marker_time'],
                         color=STYLE_PARAMS['color_time'], linestyle="--", label="Time", ax=ax2, errorbar=None)
            ax2.set_ylabel("Time", color=STYLE_PARAMS['color_time'], fontweight='bold', fontsize=STYLE_PARAMS['fontsize_label'])
            if row == 0:
                ax.set_title(f"Dataset: {dataset}", fontsize=STYLE_PARAMS['fontsize_title'], pad=15, fontweight='bold')
            ax.set_xlabel("Number of Views")
            ax.set_xticks(df["num_views"].unique())
            ax.get_legend().remove()
            ax2.get_legend().remove()
    h1, _ = axes[0, 0].get_legend_handles_labels()
    h2, _ = ax2.get_legend_handles_labels()
    fig.legend(h1 + h2, ["Performance", "Training Time"], loc="upper center", fontsize=14,
               ncol=2, frameon=False, bbox_to_anchor=STYLE_PARAMS['legend_pos'])
    plt.tight_layout(rect=STYLE_PARAMS['rect_layout'])
    plt.savefig('Views_Tradeoff.pdf', bbox_inches='tight')
    plt.show()
# %%
def draw_batchsize_tradeoff():
    df = pd.read_excel("experiment.xlsx", sheet_name="batchsizes_tradeoff")
    datasets = ["AWM", "VID"]
    metrics = ["hr_k", "ndcg_k"]
    metric_names = {"hr_k": "HR", "ndcg_k": "NDCG"}
    nrows, ncols = len(metrics), len(datasets)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=get_figsize(nrows, ncols), squeeze=False)
    for row, metric in enumerate(metrics):
        for col, dataset in enumerate(datasets):
            ax = axes[row, col]
            df_sub = df[(df["Metric"] == metric) & (df["Dataset"] == dataset)]
            sns.lineplot(data=df_sub, x="batchsize", y="Score", marker=STYLE_PARAMS['marker_perf'],
                         color=STYLE_PARAMS['color_perf'], linewidth=2.5, label="Performance", ax=ax)
            ax.set_ylabel(metric_names[metric], color=STYLE_PARAMS['color_perf'], fontweight='bold', fontsize=STYLE_PARAMS['fontsize_label'])
            ax2 = ax.twinx()
            sns.lineplot(data=df_sub, x="batchsize", y="Time_Sec", marker=STYLE_PARAMS['marker_time'],
                         color=STYLE_PARAMS['color_time'], linestyle="--", label="Time", ax=ax2, errorbar=None)
            ax2.set_ylabel("Time", color=STYLE_PARAMS['color_time'], fontweight='bold', fontsize=STYLE_PARAMS['fontsize_label'])
            ax.set_xscale('log', base=2)
            ax.set_xticks(df["batchsize"].unique())
            ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
            if row == 0:
                ax.set_title(f"Dataset: {dataset}", fontsize=STYLE_PARAMS['fontsize_title'], pad=15, fontweight='bold')
            ax.set_xlabel("Batch Size")
            ax.get_legend().remove()
            ax2.get_legend().remove()
    h1, _ = axes[0, 0].get_legend_handles_labels()
    h2, _ = ax2.get_legend_handles_labels()
    fig.legend(h1 + h2, ["Performance", "Training Time"], loc="upper center", fontsize=14,
               ncol=2, frameon=False, bbox_to_anchor=STYLE_PARAMS['legend_pos'])
    plt.tight_layout(rect=STYLE_PARAMS['rect_layout'])
    plt.savefig('BatchSize_Tradeoff.pdf', bbox_inches='tight')
    plt.show()
# %%
def calculate_significance(file_path='experiment.xlsx', target_model='CoMICE', sheet_name='Perf_Data'):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    datasets = df['Dataset'].unique()
    metrics = df['Metric'].unique()
    models = df['Model'].unique()
    baselines = [m for m in models if m != target_model]
    results = []
    for data_type in datasets:
        for metric in metrics:
            target_scores = df[
                (df['Dataset'] == data_type) &
                (df['Model'] == target_model) &
                (df['Metric'] == metric)
                ]['Score'].values
            target_mean = np.mean(target_scores)
            for baseline in baselines:
                baseline_scores = df[
                    (df['Dataset'] == data_type) &
                    (df['Model'] == baseline) &
                    (df['Metric'] == metric)
                    ]['Score'].values
                baseline_mean = np.mean(baseline_scores)
                t_stat, p_value = stats.ttest_ind(target_scores, baseline_scores, equal_var=False, alternative='greater')
                is_improved = False
                if metric == 'logloss':
                    if target_mean < baseline_mean: is_improved = True
                else:
                    if target_mean > baseline_mean: is_improved = True
                results.append({
                    'Dataset': data_type,
                    'Metric': metric,
                    'Baseline': baseline,
                    'Target_Mean': target_mean,
                    'Baseline_Mean': baseline_mean,
                    'P_Value': p_value,
                    'Significant': p_value < 0.05,
                    'Improved': is_improved
                })
    df_sig = pd.DataFrame(results)
    # df_sig.to_excel('calculate_significance.xlsx')
# %%
if __name__ == '__main__':
    draw_heatmap()