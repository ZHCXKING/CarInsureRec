# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# %%
STYLE_PARAMS = {
    'color_perf': 'tab:blue',
    'color_time': 'tab:red',
    'marker_perf': 'o',
    'marker_time': 's',
    'fontsize_title': 14,
    'fontsize_label': 12,
    'legend_pos': (0.5, 0.98),
    'rect_layout': (0, 0, 1, 0.94)
}
# 辅助函数：根据行列数计算合适的画布大小
def get_figsize(nrows, ncols):
    return (5 * ncols + 1, 4.5 * nrows)
# %%
def draw_NaRatio():
    df = pd.read_excel("experiment.xlsx", sheet_name="NaRatio_Data")
    df = df[df['Seed'].between(10, 19) & (df['Model'] != 'MIWAE')]
    datasets = ["AWM", "HIP", "VID"]
    metrics = ["hr_k", "ndcg_k"]
    metric_names = {"hr_k": "HR", "ndcg_k": "NDCG"}
    nrows, ncols = len(metrics), len(datasets)
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=get_figsize(nrows, ncols),
        sharex=True,
        squeeze=False  # 确保 axes 始终是二维的
    )
    for row, metric in enumerate(metrics):
        for col, dataset in enumerate(datasets):
            ax = axes[row, col]
            df_sub = df[(df["Metric"] == metric) & (df["Dataset"] == dataset)]

            sns.lineplot(data=df_sub, x="Ratio", y="Score", hue="Model",
                         marker="o", errorbar=None, linewidth=2, ax=ax)
            if row == 0:
                ax.set_title(f'Dataset: {dataset}', fontsize=STYLE_PARAMS['fontsize_title'], pad=10)
            if col == 0:
                ax.set_ylabel(metric_names[metric], fontsize=STYLE_PARAMS['fontsize_label'], fontweight='bold')
            else:
                ax.set_ylabel("")
            ax.set_xlabel("Inject Missing Ratio")
            if ax.get_legend(): ax.get_legend().remove()
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               frameon=False, fontsize=11, bbox_to_anchor=STYLE_PARAMS['legend_pos'])
    plt.tight_layout(rect=STYLE_PARAMS['rect_layout'])
    plt.savefig('Inject_Missing_Ratio.pdf', bbox_inches='tight')
    plt.show()
# %%
def draw_heatmap():
    df = pd.read_excel("experiment.xlsx", sheet_name="sensitivity_heatmap")
    df_filtered = df[df['Seed'].between(10, 19)]
    datasets = ['AWM', 'HIP', 'VID']
    metrics = ['hr_k', 'ndcg_k']
    for metric in metrics:
        ncols = len(datasets)
        # 热力图通常按指标分图，横向排列数据集
        fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols, 5), squeeze=False)
        for i, dataset in enumerate(datasets):
            ax = axes[0, i]
            df_sub = df_filtered[(df_filtered['Metric'] == metric) & (df_filtered['Dataset'] == dataset)]
            df_avg = df_sub.groupby(['lambda_nce', 'temperature'])['Score'].mean().reset_index()
            df_pivot = df_avg.pivot(index='temperature', columns='lambda_nce', values='Score')
            sns.heatmap(df_pivot, annot=True, fmt=".4f", cmap="YlGnBu", ax=ax)
            ax.set_title(f'Dataset: {dataset}')
            ax.set_xlabel('$\lambda_{nce}$')
            ax.set_ylabel('Temperature')
        plt.tight_layout()
        plt.savefig(f'Heatmap_{metric}.pdf', bbox_inches='tight')
        plt.show()
# %%
def draw_views_tradeoff():
    df = pd.read_excel("experiment.xlsx", sheet_name="views_tradeoff")
    df = df[df['Seed'].between(10, 19)]
    datasets = ["AWM", "HIP", "VID"]
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
            ax.set_ylabel(metric_names[metric], color=STYLE_PARAMS['color_perf'], fontweight='bold')
            ax2 = ax.twinx()
            sns.lineplot(data=df_sub, x="num_views", y="Time_Sec", marker=STYLE_PARAMS['marker_time'],
                         color=STYLE_PARAMS['color_time'], linestyle="--", label="Time", ax=ax2, errorbar=None)
            ax2.set_ylabel("Time", color=STYLE_PARAMS['color_time'], fontweight='bold')
            if row == 0:
                ax.set_title(f"Dataset: {dataset}", fontsize=STYLE_PARAMS['fontsize_title'], pad=15, fontweight='bold')
            ax.set_xlabel("Number of Views")
            ax.set_xticks(df["num_views"].unique())
            ax.get_legend().remove()
            ax2.get_legend().remove()
    h1, l1 = axes[0, 0].get_legend_handles_labels()
    h2, _ = ax2.get_legend_handles_labels()
    fig.legend(h1 + h2, ["Performance", "Training Time"], loc="upper center",
               ncol=2, frameon=False, bbox_to_anchor=STYLE_PARAMS['legend_pos'])
    plt.tight_layout(rect=STYLE_PARAMS['rect_layout'])
    plt.savefig('Views_Tradeoff.pdf', bbox_inches='tight')
    plt.show()
# %%
def draw_batchsize_tradeoff():
    df = pd.read_excel("experiment.xlsx", sheet_name="batchsizes_tradeoff")
    df = df[df['Seed'].between(10, 19)]
    datasets = ["AWM", "HIP", "VID"]
    metrics = ["hr_k", "ndcg_k"]
    metric_names = {"hr_k": "HR", "ndcg_k": "NDCG"}
    nrows, ncols = len(metrics), len(datasets)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=get_figsize(nrows, ncols), squeeze=False)
    for row, metric in enumerate(metrics):
        for col, dataset in enumerate(datasets):
            ax = axes[row, col]
            df_sub = df[(df["Metric"] == metric) & (df["Dataset"] == dataset)]
            sns.lineplot(data=df_sub, x="batchsize", y="Score", marker=STYLE_PARAMS['marker_perf'],
                         color=STYLE_PARAMS['color_perf'], linewidth=2.5, ax=ax)
            ax.set_ylabel(metric_names[metric], color=STYLE_PARAMS['color_perf'], fontweight='bold')
            ax2 = ax.twinx()
            sns.lineplot(data=df_sub, x="batchsize", y="Time_Sec", marker=STYLE_PARAMS['marker_time'],
                         color=STYLE_PARAMS['color_time'], linestyle="--", ax=ax2, errorbar=None)
            ax2.set_ylabel("Time", color=STYLE_PARAMS['color_time'], fontweight='bold')
            ax.set_xscale('log', base=2)
            ax.set_xticks(df["batchsize"].unique())
            ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
            if row == 0:
                ax.set_title(f"Dataset: {dataset}", fontsize=STYLE_PARAMS['fontsize_title'], pad=15, fontweight='bold')
            ax.set_xlabel("Batch Size")
    h1, _ = axes[0, 0].get_legend_handles_labels()
    h2, _ = ax2.get_legend_handles_labels()
    fig.legend(h1 + h2, ["Performance", "Training Time"], loc="upper center",
               ncol=2, frameon=False, bbox_to_anchor=STYLE_PARAMS['legend_pos'])
    plt.tight_layout(rect=STYLE_PARAMS['rect_layout'])
    plt.savefig('BatchSize_Tradeoff.pdf', bbox_inches='tight')
    plt.show()
# %%
if __name__ == '__main__':
    draw_NaRatio()