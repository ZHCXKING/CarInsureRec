import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from src.network import RecDataset
from src.utils import get_filled_data, load
import json
from pathlib import Path
# 假设这些类已经在你的环境中加载
# from src.models import CoMICERecommend, DCNRecommend ...

def extract_embeddings(recommender, test_data, sample_size=2000, device='cuda'):
    """
    从模型中提取 Backbone 输出的 Embedding 向量。
    """
    recommender.model.eval()

    # --- 1. 数据预处理 (与 predict 逻辑一致) ---
    data = test_data.copy()

    # 如果是 CoMICE，模型内部有 imputer；如果是 Base，假设数据已经填补好
    if hasattr(recommender, 'imputer'):
        data = recommender.imputer.transform(data)
        # 这里略去了 round 逻辑，实际使用需保持一致

    # Mapping & Standardize
    data = recommender._mapping(data, fit_bool=False)
    if recommender.standard_bool:
        data = recommender._standardize(data, fit_bool=False)

    # 转换为 Tensor
    X_tensor = torch.tensor(data[recommender.user_name].values, dtype=torch.float32)
    y_tensor = torch.tensor(data[recommender.item_name].values, dtype=torch.long)

    # --- 2. 随机采样 (t-SNE 跑太慢，只取部分点) ---
    total_len = len(data)
    if total_len > sample_size:
        indices = np.random.choice(total_len, sample_size, replace=False)
        X_tensor = X_tensor[indices]
        y_tensor = y_tensor[indices]

    # --- 3. 提取 Backbone 特征 ---
    # 构造 DataLoader
    loader = DataLoader(RecDataset(X_tensor, y_tensor), batch_size=1024, shuffle=False)

    embeddings_list = []
    labels_list = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            # 关键点：只调用 backbone，不调用 classifier_head
            # 你的 StandardModel 和 ContrastiveModel 都有 .backbone 属性
            feats = recommender.model.backbone(x)

            embeddings_list.append(feats.cpu().numpy())
            labels_list.append(y.numpy())

    return np.concatenate(embeddings_list, axis=0), np.concatenate(labels_list, axis=0)
def plot_tsne_distribution(base_model, comice_model, test_data, dataset_name):
    """
    绘制对比图：左边是 Base 模型分布，右边是 CoMICE 模型分布
    """
    print("Extracting embeddings for Base Model...")
    # Base Model 需要外部填补好的数据，这里假设传入的 test_data 已经是处理过的或者是原始的
    # 为了严谨，这里我们假设 test_data 是原始的，对 Base 需要先填补
    # 注意：这里简化处理，实际调用时请确保数据状态正确
    base_emb, base_labels = extract_embeddings(base_model, test_data)

    print("Extracting embeddings for CoMICE Model...")
    comice_emb, comice_labels = extract_embeddings(comice_model, test_data)

    print("Running t-SNE (this may take a moment)...")
    # 初始化 t-SNE
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42, init='pca', learning_rate='auto')

    # 降维
    base_2d = tsne.fit_transform(base_emb)
    comice_2d = tsne.fit_transform(comice_emb)

    # --- 绘图配置 ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # 颜色映射：0 (负样本) -> 蓝色, 1 (正样本) -> 红色
    palette = {0: '#4E79A7', 1: '#E15759'}
    style_params = {'s': 20, 'alpha': 0.6, 'edgecolor': 'w', 'linewidth': 0.1}

    # 图1: Base Model
    sns.scatterplot(x=base_2d[:, 0], y=base_2d[:, 1], hue=base_labels, palette=palette, ax=axes[0], **style_params)
    axes[0].set_title(f"Base Model ({base_model.model_name}) Representation", fontsize=14, fontweight='bold')
    axes[0].legend(title='Label', loc='upper right')
    axes[0].grid(True, linestyle='--', alpha=0.3)

    # 图2: CoMICE Model
    sns.scatterplot(x=comice_2d[:, 0], y=comice_2d[:, 1], hue=comice_labels, palette=palette, ax=axes[1], **style_params)
    axes[1].set_title(f"CoMICE ({comice_model.model_name}) Representation", fontsize=14, fontweight='bold')
    axes[1].legend(title='Label', loc='upper right')
    axes[1].grid(True, linestyle='--', alpha=0.3)

    plt.suptitle(f"t-SNE Visualization of Embedding Space on {dataset_name}", fontsize=18, y=1.02)
    plt.tight_layout()

    # 保存图片
    save_path = f"{dataset_name}_tsne_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {save_path}")
    plt.show()
# --- 如何调用这个功能的示例 ---
def run_visualization_experiment():
    # 1. 设置
    data_type = 'HIP'  # 选择一个数据集
    backbone_name = 'DCN'
    seed = 42
    root = Path(__file__).parents[0]  # 根据你的目录结构调整

    print(f"Loading data for {data_type}...")
    train, valid, test, info = load(data_type, None, 0.7, 0.1, is_dropna=False)

    # 加载参数
    param_file = root / data_type / (backbone_name + "_param.json")
    with open(param_file, 'r') as f:
        params = json.load(f)

    # 2. 训练/加载 Base Model
    print("Training Base Model...")
    # Base Model 需要手动填补数据
    train_filled, valid_filled, test_filled = get_filled_data(train, valid, test, info['sparse_features'], seed=seed)

    # 动态获取类，例如 DCNRecommend
    ModelClass = globals()[f"{backbone_name}Recommend"]
    base_model = ModelClass(info['item_name'], info['sparse_features'], info['dense_features'], seed=seed, k=3, **params)
    base_model.fit(train_filled, valid_filled)  # 训练

    # 3. 训练/加载 CoMICE Model
    print("Training CoMICE Model...")
    # CoMICE Model 内部处理填补，传入原始含 NaN 数据
    comice_model = CoMICERecommend(info['item_name'], info['sparse_features'], info['dense_features'], seed=seed, k=3, backbone=backbone_name, **params)
    comice_model.fit(train, valid)  # 训练

    # 4. 执行可视化
    # 注意：Base Model 我们传入 test_filled，CoMICE 我们传入原始 test
    # 为了函数复用，我们可以先分别提取向量，或者像下面这样灵活处理
    # 这里为了方便演示，我们在 extract_embeddings 内部做了判断，但 Base Model 传入的数据必须是无 NaN 的
    print("Generating Plots...")

    # 提取 Base 向量 (使用 test_filled)
    base_vecs, base_lbls = extract_embeddings(base_model, test_filled, sample_size=2000)

    # 提取 CoMICE 向量 (使用原始 test, extract_embeddings 会调用模型内置 imputer)
    comice_vecs, comice_lbls = extract_embeddings(comice_model, test, sample_size=2000)

    # 手动调用绘图部分，因为数据源不一样
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    base_2d = tsne.fit_transform(base_vecs)
    comice_2d = tsne.fit_transform(comice_vecs)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    palette = {0: 'royalblue', 1: 'crimson'}

    sns.scatterplot(x=base_2d[:, 0], y=base_2d[:, 1], hue=base_lbls, palette=palette, ax=axes[0], s=20, alpha=0.6)
    axes[0].set_title(f"Base {backbone_name}")

    sns.scatterplot(x=comice_2d[:, 0], y=comice_2d[:, 1], hue=comice_lbls, palette=palette, ax=axes[1], s=20, alpha=0.6)
    axes[1].set_title(f"CoMICE {backbone_name}")

    plt.show()
if __name__ == "__main__":
    # 确保你的环境中有必要的类定义
    # 比如复制到 test_NaRatio 下面运行
    run_visualization_experiment()