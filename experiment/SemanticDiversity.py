# %%
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from src.models import MICLRecommend
from src.utils import load, filling, round, inject_missingness
# %%
datasets = ['AWM', 'HIP', 'VID']
MICL_Backbone = 'DCN'
amount = None
train_ratio = 0.7
val_ratio = 0.1
root = Path(__file__).parents[0]
# %%
def _select_samples(test_df, sparse_features, dense_features, item_name, n, seed):
    feature_cols = sparse_features + dense_features
    na_mask = test_df[feature_cols].isna().any(axis=1)
    candidates = test_df[na_mask]
    if len(candidates) < n:
        print(f"  [warn] only {len(candidates)} samples contain missing values, less than n={n}.")
        n = len(candidates)
    rng = np.random.RandomState(seed)
    chosen_idx = rng.choice(candidates.index, size=n, replace=False)
    selected = candidates.loc[chosen_idx].reset_index(drop=True)
    selected['_sample_id'] = np.arange(len(selected))
    return selected
# %%
def _generate_mice_views(train_df, selected_df, sparse_features, item_name, mice_method, V, base_seed):
    feature_and_item_cols = [c for c in selected_df.columns if c != '_sample_id']
    sample_ids = selected_df['_sample_id'].values
    n_selected = len(selected_df)
    train_part = train_df[feature_and_item_cols].copy()
    selected_part = selected_df[feature_and_item_cols].copy()
    combined = pd.concat([train_part, selected_part], axis=0).reset_index(drop=True)
    views = []
    for v in range(V):
        view_seed = base_seed + 1000 + v
        filled, _ = filling(combined.copy(), method=mice_method, seed=view_seed)
        filled = round(filled, sparse_features, item_name=item_name)
        view_part = filled.iloc[-n_selected:].reset_index(drop=True)
        view_part['_sample_id'] = sample_ids
        view_part['_view_id'] = v
        views.append(view_part)
    return views
# %%
def _extract_projections(model, view_df, item_name):
    view_df = view_df.copy()
    df_for_input = view_df.drop(columns=['_sample_id', '_view_id'])
    df_mapped = model._mapping(df_for_input, fit_bool=False)
    if model.standard_bool:
        df_mapped = model._standardize(df_mapped, fit_bool=False)
    X = torch.tensor(df_mapped[model.user_name].values, dtype=torch.float32).to(model.device)
    model.model.eval()
    with torch.no_grad():
        logits, _ = model.model(X)
        logits = F.normalize(logits, dim=1).cpu().numpy()
    labels = df_mapped[item_name].values
    sample_ids = view_df['_sample_id'].values
    view_ids = view_df['_view_id'].values
    return logits, labels, sample_ids, view_ids
# %%
def _plot_tsne(axes_row, dataset_name, Z_2d, sample_ids, view_ids, labels, V):
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    if n_classes <= 10:
        cmap = plt.get_cmap('tab10')
    elif n_classes <= 20:
        cmap = plt.get_cmap('tab20')
    else:
        cmap = plt.get_cmap('hsv')
    color_map = {lab: cmap(i % cmap.N) for i, lab in enumerate(unique_labels)}
    ax = axes_row
    for lab in unique_labels:
        idx = np.where(labels == lab)[0]
        ax.scatter(
            Z_2d[idx, 0], Z_2d[idx, 1],
            c=[color_map[lab]],
            s=22,
            alpha=0.75,
            edgecolors='none',
            label=f'item {int(lab)}'
        )
    ax.set_title(f'{dataset_name} (V={V})', fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE dim 1', fontsize=11)
    ax.set_ylabel('t-SNE dim 2', fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
# %%
def run_semantic_diversity(n=50, V=3, seed=0, inject_ratio=0.0, output_path=None):
    fig, axes = plt.subplots(nrows=1, ncols=len(datasets), figsize=(6 * len(datasets), 5.5), squeeze=False)
    axes = axes[0]
    for col, data_type in enumerate(datasets):
        print(f"\n[{data_type}] loading data ...")
        train, valid, test, info = load(data_type, amount, train_ratio, val_ratio, is_dropna=False, seed=seed)
        item_name = info['item_name']
        sparse_features = info['sparse_features']
        dense_features = info['dense_features']
        if inject_ratio and inject_ratio > 0.0:
            print(f"[{data_type}] injecting extra missingness with ratio={inject_ratio} ...")
            train = inject_missingness(train, sparse_features, dense_features, ratio=inject_ratio, seed=seed)
            valid = inject_missingness(valid, sparse_features, dense_features, ratio=inject_ratio, seed=seed)
            test = inject_missingness(test, sparse_features, dense_features, ratio=inject_ratio, seed=seed)
        param_file = root / data_type / (MICL_Backbone + "_param.json")
        with open(param_file, 'r') as f:
            params = json.load(f)
        params['num_views'] = V
        print(f"[{data_type}] training MICLRec (backbone={MICL_Backbone}, seed={seed}, V={V}) ...")
        model = MICLRecommend(
            item_name, sparse_features, dense_features,
            seed=seed, k=3, backbone=MICL_Backbone, **params
        )
        model.fit(train.copy(), valid.copy())
        print(f"[{data_type}] selecting n={n} test samples with missing values ...")
        selected = _select_samples(test, sparse_features, dense_features, item_name, n=n, seed=seed)
        if len(selected) == 0:
            print(f"[{data_type}] no samples with missing values, skip.")
            continue
        print(f"[{data_type}] generating V={V} MICE views for selected samples ...")
        views = _generate_mice_views(
            train, selected, sparse_features, item_name,
            mice_method=model.kwargs['mice_method'], V=V, base_seed=seed
        )
        all_proj, all_labels, all_sids, all_vids = [], [], [], []
        for view_df in views:
            proj, labels, sids, vids = _extract_projections(model, view_df, item_name)
            all_proj.append(proj)
            all_labels.append(labels)
            all_sids.append(sids)
            all_vids.append(vids)
        Z = np.concatenate(all_proj, axis=0)
        labels_arr = np.concatenate(all_labels, axis=0)
        sample_ids = np.concatenate(all_sids, axis=0)
        view_ids = np.concatenate(all_vids, axis=0)
        print(f"[{data_type}] running t-SNE on Z of shape {Z.shape} ...")
        perplexity = max(5, min(30, (len(Z) - 1) // 3))
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            init='pca',
            metric='cosine',
            random_state=seed,
            learning_rate='auto'
        )
        Z_2d = tsne.fit_transform(Z)
        _plot_tsne(axes[col], data_type, Z_2d, sample_ids, view_ids, labels_arr, V)
    plt.tight_layout()
    if output_path is None:
        output_path = root / 'SemanticDiversity.pdf'
    plt.savefig(output_path, bbox_inches='tight')
    print(f"\nSaved figure to: {output_path}")
    plt.show()
# %%
if __name__ == "__main__":
    run_semantic_diversity(n=5, V=20, seed=0, inject_ratio=0.0)
