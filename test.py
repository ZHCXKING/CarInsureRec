import numpy as np
import pandas as pd
import torch
import random
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from .base import BaseRecommender
from src.utils import filling, process_mice_list, round


# %% 1. 核心交互模块：Cross Network (DCN-V2)
class CrossNet(nn.Module):
    """
    Deep & Cross Network V2 的核心模块
    公式: x_{l+1} = x_0 * (W_l * x_l + b_l) + x_l
    """

    def __init__(self, input_dim, num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        self.W = nn.ParameterList([
            nn.Parameter(torch.empty(input_dim, input_dim))
            for _ in range(num_layers)
        ])
        self.b = nn.ParameterList([
            nn.Parameter(torch.empty(input_dim))
            for _ in range(num_layers)
        ])

        for w in self.W:
            nn.init.xavier_uniform_(w)
        for b in self.b:
            nn.init.zeros_(b)

    def forward(self, x_0):
        # x_0: [Batch, Input_Dim]
        x_l = x_0
        for i in range(self.num_layers):
            # 1. Linear Projection: W * x + b
            linear_proj = torch.mm(x_l, self.W[i]) + self.b[i]
            # 2. Explicit Feature Crossing: x_0 * (Projected) + x_l
            x_l = x_0 * linear_proj + x_l
        return x_l


# %% 2. 升级后的推荐模型：DCNv2RecModel
class DCNv2RecModel(nn.Module):
    def __init__(self, num_classes, sparse_dims, dense_count,
                 feature_dim=32, cross_layers=3, hidden_units=[256, 128], output_dim=128):
        super().__init__()
        # ... [前面的 Embedding, CrossNet, DeepNet 定义保持完全不变] ...
        self.num_sparse = len(sparse_dims)
        self.num_dense = dense_count

        # 1. Embeddings & Projections
        self.sparse_embs = nn.ModuleList([
            nn.Embedding(dim, feature_dim) for dim in sparse_dims
        ])
        if self.num_dense > 0:
            self.dense_proj = nn.ModuleList([
                nn.Linear(1, feature_dim) for _ in range(self.num_dense)
            ])

        self.total_input_dim = (self.num_sparse + self.num_dense) * feature_dim

        # 2. Towers
        self.cross_net = CrossNet(self.total_input_dim, num_layers=cross_layers)

        dnn_layers = []
        last_dim = self.total_input_dim
        for hidden in hidden_units:
            dnn_layers.append(nn.Linear(last_dim, hidden))
            dnn_layers.append(nn.BatchNorm1d(hidden))
            dnn_layers.append(nn.ReLU())
            dnn_layers.append(nn.Dropout(0.1))
            last_dim = hidden
        self.deep_net = nn.Sequential(*dnn_layers)

        # --- 修改区域开始 ---

        # 定义融合后的特征维度
        self.fusion_dim = self.total_input_dim + hidden_units[-1]

        # A. 投影头 (Projection Head) - 用于生成稳健特征
        # 注意：这里我们稍微加深一点，确保它有能力提取高级语义
        self.projection_head = nn.Sequential(
            nn.Linear(self.fusion_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

        # B. 门控机制 (Gating Mechanism)
        # 利用稳健特征(proj) 生成 门控权重(gate)
        # 输入是 output_dim (proj的维度), 输出是 fusion_dim (x_final的维度)
        self.gate_layer = nn.Sequential(
            nn.Linear(output_dim, self.fusion_dim),
            nn.Sigmoid()  # 输出 0~1 的权重
        )

        # C. 分类头 (Classifier Head)
        # 输入依然是 fusion_dim，但是是经过过滤后的特征
        # 也可以选择 concat，把 input_dim 变成 fusion_dim + output_dim，但Gating通常更优雅
        self.classifier_head = nn.Linear(self.fusion_dim, num_classes)

    def forward(self, x):
        # 1. 特征处理 (不变)
        sparse_x = x[:, :self.num_sparse].long()
        dense_x = x[:, self.num_sparse:]
        embeddings = []
        for i, emb in enumerate(self.sparse_embs):
            embeddings.append(emb(sparse_x[:, i]))
        if self.num_dense > 0:
            for i in range(self.num_dense):
                embeddings.append(self.dense_proj[i](dense_x[:, i].unsqueeze(1)))
        x_0 = torch.cat(embeddings, dim=1)

        # 2. 双塔计算 (不变)
        x_cross = self.cross_net(x_0)
        x_deep = self.deep_net(x_0)

        # 3. 初始融合特征 (Raw Features)
        x_final = torch.cat([x_cross, x_deep], dim=1)  # [B, fusion_dim]

        # --- 修改逻辑开始 ---

        # 4. 生成投影向量 (Robust Features)
        # 注意：这里先不归一化，直接用 Projection Head 的原始输出做 Gating 效果更好
        # 归一化后的 proj_norm 仅用于计算 InfoNCE Loss
        proj_feat = self.projection_head(x_final)
        proj_norm = F.normalize(proj_feat, dim=1)

        # 5. 特征校准 (Feature Calibration)
        # 利用 proj_feat 告诉 x_final：哪些特征是噪音，该抑制；哪些是干货，该保留
        gate = self.gate_layer(proj_feat)  # [B, fusion_dim], 值在 0~1 之间

        x_calibrated = x_final * gate  # Element-wise product (哈达玛积)

        # 也可以选择残差连接： x_calibrated = x_final * (1 + gate)
        # 但在去噪场景下，乘法门控 (x * gate) 更能有效“关掉”噪音维度

        # 6. 分类输出
        # 使用校准后的特征进行预测
        logits = self.classifier_head(x_calibrated)

        return logits, proj_norm


# %% 3. 主类更新 (CoMICERecommend)
class CoMICERecommend(BaseRecommender):
    def __init__(self, user_name: list, item_name: str, date_name: str | None = None,
                 sparse_features: list | None = None, dense_features: list | None = None, standard_bool: bool = False,
                 seed: int = 42, k: int = 3, **kwargs):
        if (user_name is None) or (item_name is None):
            raise ValueError('user_name and item_name are required')
        super().__init__('CoMICE', user_name, item_name, date_name, sparse_features, dense_features, standard_bool, seed, k)
        default_params = {
            'm': 3,
            'lr': 1e-4,
            'batch_size': 64,
            'feature_dim': 32,  # 新增: 特征嵌入维度 (原 embed_dim)
            'proj_dim': 128,  # 新增: 对比学习投影维度
            'epochs': 200,
            'lambda_topk': 0.5,  # 建议调大一点以优化Ranking
            'lambda_nce': 1.0,
            'temperature': 0.1,
            'margin': 0.2,
            'alpha': 1.0,
            'mice_method': 'iterative_SVM'
        }
        self.kwargs.update(default_params)
        self.kwargs.update(kwargs)
        self.user_name = sparse_features + dense_features
        self.out_dim = None
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.optimizer = None
        self.imputers = []

    # ... [train_one_epoch 保持不变] ...
    def train_one_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        topk_loss_fn = TopKMarginLoss(k=self.k, margin=self.kwargs['margin']).to(self.device)
        for x_view1, x_view2, y in dataloader:
            x_view1, x_view2, y = x_view1.to(self.device), x_view2.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            logits1, proj1 = self.model(x_view1)
            logits2, proj2 = self.model(x_view2)

            # Loss计算部分保持原逻辑不变...
            loss_ce = 0.5 * (F.cross_entropy(logits1, y, label_smoothing=0.1) + F.cross_entropy(logits2, y, label_smoothing=0.1))
            loss_topk = 0.5 * (topk_loss_fn(logits1, y) + topk_loss_fn(logits2, y))

            batch_size = x_view1.size(0)
            features = torch.cat([proj1, proj2], dim=0)
            sim_matrix = torch.matmul(features, features.T) / self.kwargs['temperature']
            labels_contrastive = torch.cat([torch.arange(batch_size) + batch_size, torch.arange(batch_size)]).to(self.device)
            mask = torch.eye(2 * batch_size, dtype=torch.bool).to(self.device)
            sim_matrix.masked_fill_(mask, -9e15)
            loss_nce = F.cross_entropy(sim_matrix, labels_contrastive)

            loss = (loss_ce + self.kwargs['lambda_topk'] * loss_topk + self.kwargs['lambda_nce'] * loss_nce)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        print(f"Loss: {total_loss / len(dataloader):.4f}")

    def fit(self, train_data: pd.DataFrame):
        self.out_dim = train_data[self.item_name].nunique()
        self.unique_item = list(range(self.out_dim))
        train_data = self._mapping(train_data, fit_bool=True)
        if self.standard_bool:
            train_data = self._standardize(train_data, fit_bool=True)

        rng = np.random.default_rng(seed=self.seed)
        rints = rng.choice(a=self.kwargs['m'] * 10, size=self.kwargs['m'], replace=False)
        train_data_sets = []
        for i in range(self.kwargs['m']):
            data, imputer = filling(train_data, method=self.kwargs['mice_method'], seed=rints[i])
            train_data_sets.append(data)
            self.imputers.append(imputer)

        dense_count = len(self.dense_features)
        sparse_dims = [self.vocabulary_sizes[col] for col in self.sparse_features]
        x_train_tensor, y_train_tensor = process_mice_list(train_data_sets, self.user_name, self.item_name)

        train_loader = DataLoader(
            RecDataset(x_train_tensor, y_train_tensor, mode='train'),
            batch_size=self.kwargs['batch_size'],
            shuffle=True
        )

        # --- 实例化新的 DCN-V2 模型 ---
        self.model = DCNv2RecModel(
            num_classes=self.out_dim,
            sparse_dims=sparse_dims,
            dense_count=dense_count,
            feature_dim=self.kwargs['feature_dim'],  # 建议 32 或 64
            output_dim=self.kwargs['proj_dim']  # 建议 128
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.kwargs['lr'], weight_decay=1e-5)

        for epoch in range(self.kwargs['epochs']):
            self.train_one_epoch(train_loader)
        self.is_trained = True

    # ... [get_proba 保持不变] ...
    def get_proba(self, test_data: pd.DataFrame):
        # 保持原有的推理逻辑，因为模型输入输出接口未变
        test_data = self._mapping(test_data, fit_bool=False)
        if self.standard_bool:
            test_data = self._standardize(test_data, fit_bool=False)
        test_data_sets = []
        for i in range(self.kwargs['m']):
            data = self.imputers[i].transform(test_data)
            data = round(data)
            test_data_sets.append(data)
        x_test_tensor, y_test_tensor = process_mice_list(test_data_sets, self.user_name, self.item_name)
        test_loader = DataLoader(RecDataset(x_test_tensor, y_test_tensor, mode='eval'), batch_size=self.kwargs['batch_size'], shuffle=False)
        self.model.eval()
        all_scores = []
        with torch.no_grad():
            for x_all, y in test_loader:
                x_all = x_all.to(self.device)
                batch_size, m, d = x_all.shape
                x_flat = x_all.view(-1, d)
                logits, _ = self.model(x_flat)
                logits = logits.view(batch_size, m, -1)
                mean_logits = logits.mean(dim=1)
                var_logits = logits.var(dim=1, unbiased=False)
                risk_scores = mean_logits / (1.0 + self.kwargs['alpha'] * var_logits)
                all_scores.append(risk_scores.cpu().numpy())
        finall_scores = np.concatenate(all_scores, axis=0)
        result = pd.DataFrame(finall_scores, index=test_data.index, columns=self.unique_item)
        return result


# ... [RecDataset 和 TopKMarginLoss 保持不变] ...
class RecDataset(Dataset):
    def __init__(self, X_imputed, y, mode='train'):
        self.X = X_imputed
        self.y = y
        self.mode = mode
        self.N, self.m, self.d = X_imputed.shape

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        if self.mode == 'train':
            idx1, idx2 = random.sample(range(self.m), 2)
            x_view1 = self.X[idx, idx1, :]
            x_view2 = self.X[idx, idx2, :]
            label = self.y[idx]
            return x_view1, x_view2, label
        elif self.mode == 'eval':
            x_all = self.X[idx, :, :]
            label = self.y[idx]
            return x_all, label
        else:
            raise ValueError('mode is not supported')


class TopKMarginLoss(nn.Module):
    def __init__(self, k=5, margin=0.2):
        super().__init__()
        self.k = k
        self.margin = margin

    def forward(self, logits, targets):
        B = logits.size(0)
        topk_vals, topk_idx = torch.topk(logits, self.k, dim=1)
        true_logits = logits[torch.arange(B), targets]
        mask = topk_idx != targets.unsqueeze(1)
        neg_logits = topk_vals.masked_fill(~mask, -1e9).max(dim=1).values
        loss = F.relu(self.margin - true_logits + neg_logits)
        return loss.mean()