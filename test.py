import numpy as np
import pandas as pd
import torch
import random
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torch.nn import functional as F
from .base import BaseRecommender
from src.utils import filling, process_mice_list, round
from src.network import DCNv2Backbone, DeepFMBackbone, WideDeepBackbone, CoMICEHead, CoMICEModel, set_seed
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
            'proj_dim': 32,
            'epochs': 500,
            'lambda_nce': 1.0,
            'temperature': 0.1,
            'alpha': 1.0,
            'mice_method': 'iterative_Ga',
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

    def train_one_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        m = self.kwargs['m']
        for x_all, y in dataloader:
            # x_all shape: [batch_size, m, input_dim]
            batch_size = x_all.size(0)
            x_all, y = x_all.to(self.device), y.to(self.device)
            # 将 batch 和 m 维度合并
            x_flat = x_all.view(-1, x_all.size(-1))
            logits_flat, proj_flat = self.model(x_flat)
            # 1. 分类损失 (Cross Entropy)
            y_expanded = y.repeat_interleave(m)  # [batch_size * m]
            loss_ce = F.cross_entropy(logits_flat, y_expanded, label_smoothing=0.1)
            # 2. 有监督对比损失 (Supervised Contrastive Loss)
            # 计算相似度矩阵 [BM, BM]
            # proj_flat 已经经过了 F.normalize，所以 matmul 等同于余弦相似度
            sim_matrix = torch.matmul(proj_flat, proj_flat.T) / self.kwargs['temperature']
            # ----------------- 构建 SupCon Mask -----------------
            # 这里的逻辑是：如果 y_i == y_j，则 mask[i,j] = 1
            y_expanded = y_expanded.view(-1, 1)
            mask = torch.eq(y_expanded, y_expanded.T).float().to(self.device)
            # 排除自身对比 (Self-contrast mask)
            # 对角线设为 0，因为一个视角不应该和自己进行对比损失计算
            logits_mask = torch.scatter(
                torch.ones_like(mask), 1, torch.arange(batch_size * m).view(-1, 1).to(self.device), 0)
            mask = mask * logits_mask
            # 计算 Log-Sum-Exp (InfoNCE 分母部分)
            # 为了数值稳定性，减去每行的最大值
            logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
            logits = sim_matrix - logits_max.detach()
            # exp(sim) 并排除对角线
            exp_logits = torch.exp(logits) * logits_mask
            # 计算 log(分子/分母) -> log(分子) - log(分母)
            # mask * logits 提取出所有正样本的相似度
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
            # 计算每个锚点(anchor)对其所有正样本的平均对数似然
            # mask.sum(1) 是每个样本在 batch 中拥有的正样本数量（含其他视角和同类样本）
            mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
            # 只有当该样本在 batch 内存在正样本时才计算损失
            valid_indices = mask.sum(1) > 0
            loss_nce = -mean_log_prob_pos[valid_indices].mean()
            # 总损失
            loss = loss_ce + self.kwargs['lambda_nce'] * loss_nce
            self.optimizer.zero_grad()
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
    def get_proba(self, test_data: pd.DataFrame):
        test_data = self._mapping(test_data, fit_bool=False)
        if self.standard_bool:
            test_data = self._standardize(test_data, fit_bool=False)
        test_data_sets = []
        for i in range(self.kwargs['m']):
            data = self.imputers[i].transform(test_data)
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
                logits = logits.view(batch_size, m, -1)
                mean_logits = logits.mean(dim=1)
                var_logits = logits.var(dim=1, unbiased=False)
                # 风险敏感评分公式
                risk_scores = mean_logits / (1.0 + self.kwargs['alpha'] * var_logits)
                all_scores.append(risk_scores.cpu().numpy())
        final_scores = np.concatenate(all_scores, axis=0)
        return pd.DataFrame(final_scores, index=test_data.index, columns=self.unique_item)
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