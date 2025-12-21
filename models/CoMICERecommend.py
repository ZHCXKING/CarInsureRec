# %%
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
from .base import BaseRecommender
from src.utils import filling, process_mice_list, round
from src.network import DCNv2Backbone, DeepFMBackbone, WideDeepBackbone, CoMICEHead, CoMICEModel, set_seed
# %%
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
            'batch_size': 256,
            'feature_dim': 64,
            'proj_dim': 64,
            'epochs': 500,
            'lambda_nce': 1.0,
            'temperature': 0.1,
            'mice_method': 'MICE_RF',
            'backbone': 'DCNv2',
            'cross_layers': 3,
            'hidden_units': [256, 128]
        }
        self.kwargs.update(default_params)
        self.kwargs.update(kwargs)
        self.user_name = sparse_features + dense_features
        self.out_dim = None
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.optimizer = None
        self.imputers = []
        set_seed(self.seed)
    # %%
    def _build_model(self, sparse_dims, dense_count, num_classes):
        backbone_type = self.kwargs['backbone']
        if backbone_type == 'DCNv2':
            backbone = DCNv2Backbone(sparse_dims, dense_count, self.kwargs['feature_dim'], self.kwargs['cross_layers'], self.kwargs['hidden_units'])
        elif backbone_type == 'DeepFM':
            backbone = DeepFMBackbone(sparse_dims, dense_count, self.kwargs['feature_dim'], self.kwargs['hidden_units'])
        elif backbone_type == 'WideDeep':
            backbone = WideDeepBackbone(sparse_dims, dense_count, self.kwargs['feature_dim'], self.kwargs['hidden_units'])
        else:
            raise ValueError(f"Unsupported backbone: {backbone_type}")
        head = CoMICEHead(input_dim=backbone.output_dim, num_classes=num_classes, proj_dim=self.kwargs['proj_dim'])
        return CoMICEModel(backbone, head).to(self.device)
    # %%
    def train_one_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        m = self.kwargs['m']
        for x_all, y in dataloader:
            # x_all shape: [batch_size, m, input_dim]
            batch_size = x_all.size(0)
            x_all, y = x_all.to(self.device), y.to(self.device)
            # 将 batch 和 m 维度合并，一次性通过模型以提高效率
            # x_flat shape: [batch_size * m, input_dim]
            x_flat = x_all.view(-1, x_all.size(-1))
            logits_flat, proj_flat = self.model(x_flat)
            # 1. 分类损失 (Cross Entropy): 计算所有 m 个视角的平均损失
            # y 需要扩展以匹配 logits_flat
            y_expanded = y.repeat_interleave(m)
            loss_ce = F.cross_entropy(logits_flat, y_expanded)
            # 2. 对比损失 (Multi-view InfoNCE)
            # proj_flat shape: [batch_size * m, proj_dim]
            # 计算特征相似度矩阵 [BM, BM]
            sim_matrix = torch.matmul(proj_flat, proj_flat.T) / self.kwargs['temperature']
            # 构建正样本掩码：同一行的不同插补版本互为正样本
            # 只有当两个样本来自同一个原始 index 时，mask 为 1
            labels_idx = torch.arange(batch_size).to(self.device).repeat_interleave(m)
            mask = (labels_idx.unsqueeze(0) == labels_idx.unsqueeze(1)).fill_diagonal_(False)
            # 这里采用一种简化的 Multi-view InfoNCE 实现：
            # 对于每一个视角，其正样本是来自同一 ID 的其他 m-1 个视角
            # 我们将非正样本（不同 ID）作为负样本
            # 为了方便计算，我们通过 log_softmax 处理相似度矩阵
            exp_sim = torch.exp(sim_matrix)
            # 掩盖掉自对比 (Self-similarity)
            diag_mask = torch.eye(batch_size * m, device=self.device).bool()
            exp_sim = exp_sim.masked_fill(diag_mask, 0)
            # 计算每个维度的 InfoNCE: log(sum(pos_sim) / sum(all_sim))
            pos_sim = (exp_sim * mask).sum(dim=1)
            all_sim = exp_sim.sum(dim=1)
            loss_nce = -torch.log(pos_sim / (all_sim + 1e-8) + 1e-8).mean()
            # 总损失
            loss = loss_ce + self.kwargs['lambda_nce'] * loss_nce
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        print(f"Loss: {total_loss / len(dataloader):.4f}")
    # %%
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
        # 处理成 [N, M, D] 的张量
        x_train_tensor, y_train_tensor = process_mice_list(train_data_sets, self.user_name, self.item_name)
        train_loader = DataLoader(
            RecDataset(x_train_tensor, y_train_tensor),
            batch_size=self.kwargs['batch_size'],
            shuffle=True
        )
        self.model = self._build_model(sparse_dims, dense_count, self.out_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.kwargs['lr'])
        for epoch in range(self.kwargs['epochs']):
            self.train_one_epoch(train_loader)
        self.is_trained = True
    # %%
    def get_proba(self, test_data: pd.DataFrame):
        test_data = self._mapping(test_data, fit_bool=False)
        if self.standard_bool:
            test_data = self._standardize(test_data, fit_bool=False)
        test_data_sets = []
        data = self.imputers[0].transform(test_data)
        data = round(data)
        test_data_sets.append(data)
        x_test_tensor, y_test_tensor = process_mice_list(test_data_sets, self.user_name, self.item_name)
        test_loader = DataLoader(
            RecDataset(x_test_tensor, y_test_tensor),
            batch_size=self.kwargs['batch_size'],
            shuffle=False
        )
        self.model.eval()
        all_scores = []
        with torch.no_grad():
            for x_all, _ in test_loader:
                x_all = x_all.to(self.device)
                batch_size, m, d = x_all.shape
                x_flat = x_all.view(-1, d)
                logits, _ = self.model(x_flat)
                probs = torch.softmax(logits, dim=1)
                all_scores.append(probs.cpu().numpy())
        final_scores = np.concatenate(all_scores, axis=0)
        return pd.DataFrame(final_scores, index=test_data.index, columns=self.unique_item)
# %%
class RecDataset(Dataset):
    def __init__(self, X_imputed, y):
        """
        X_imputed: Tensor of shape [N, m, d]
        y: Tensor of shape [N]
        """
        self.X = X_imputed
        self.y = y
        self.N, self.m, self.d = X_imputed.shape
    def __len__(self):
        return self.N
    def __getitem__(self, idx):
        # 无论 train 还是 eval，现在都返回该样本的所有 m 个插补视角
        x_all = self.X[idx, :, :]  # [m, d]
        label = self.y[idx]
        return x_all, label
