import numpy as np
import pandas as pd
import torch
import random
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torch.nn import functional as F
from .base import BaseRecommender
from src.utils import filling, process_mice_list, round
from src.network import DCNv2Backbone, DeepFMBackbone, WideDeepBackbone, CoMICEHead, CoMICEModel
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
            'feature_dim': 32,  # 特征嵌入维度
            'proj_dim': 128,  # 对比学习投影维度
            'epochs': 200,
            'lambda_topk': 0.5,
            'lambda_nce': 1.0,
            'temperature': 0.1,
            'margin': 0.2,
            'alpha': 1.0,
            'mice_method': 'iterative_SVM',
            'backbone': 'DCNv2',  # 新增：选择模型骨架 ['DCNv2', 'DeepFM', 'WideDeep']
            'cross_layers': 3,  # DCNv2 专用参数
            'hidden_units': [256, 128]  # 通用DNN参数
        }
        self.kwargs.update(default_params)
        self.kwargs.update(kwargs)
        self.user_name = sparse_features + dense_features
        self.out_dim = None
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.optimizer = None
        self.imputers = []
    def _build_model(self, sparse_dims, dense_count, num_classes):
        """内部方法：根据参数构建 Backbone + CoMICEHead"""
        backbone_type = self.kwargs['backbone']
        # 1. 选择并初始化 Backbone
        if backbone_type == 'DCNv2':
            backbone = DCNv2Backbone(
                sparse_dims, dense_count,
                feature_dim=self.kwargs['feature_dim'],
                cross_layers=self.kwargs['cross_layers'],
                hidden_units=self.kwargs['hidden_units']
            )
        elif backbone_type == 'DeepFM':
            backbone = DeepFMBackbone(
                sparse_dims, dense_count,
                feature_dim=self.kwargs['feature_dim'],
                hidden_units=self.kwargs['hidden_units']
            )
        elif backbone_type == 'WideDeep':
            backbone = WideDeepBackbone(
                sparse_dims, dense_count,
                feature_dim=self.kwargs['feature_dim'],
                hidden_units=self.kwargs['hidden_units']
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone_type}")
        # 2. 初始化 CoMICE Head
        head = CoMICEHead(
            input_dim=backbone.output_dim,
            num_classes=num_classes,
            proj_dim=self.kwargs['proj_dim']
        )
        # 3. 组装
        return CoMICEModel(backbone, head).to(self.device)
    def train_one_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        # 假设 TopKMarginLoss 类已经在外部定义或在同一文件中
        topk_loss_fn = TopKMarginLoss(k=self.k, margin=self.kwargs['margin']).to(self.device)
        for x_view1, x_view2, y in dataloader:
            x_view1, x_view2, y = x_view1.to(self.device), x_view2.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            # 模型输出 (logits, proj_norm)
            logits1, proj1 = self.model(x_view1)
            logits2, proj2 = self.model(x_view2)
            # Loss 计算 (逻辑保持不变)
            loss_ce = 0.5 * (F.cross_entropy(logits1, y, label_smoothing=0.1) +
                             F.cross_entropy(logits2, y, label_smoothing=0.1))

            loss_topk = 0.5 * (topk_loss_fn(logits1, y) + topk_loss_fn(logits2, y))
            # InfoNCE Loss
            batch_size = x_view1.size(0)
            features = torch.cat([proj1, proj2], dim=0)
            sim_matrix = torch.matmul(features, features.T) / self.kwargs['temperature']
            labels_contrastive = torch.cat([torch.arange(batch_size) + batch_size,
                                            torch.arange(batch_size)]).to(self.device)
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
        # --- 使用动态构建方法 ---
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
            RecDataset(x_test_tensor, y_test_tensor, mode='eval'),
            batch_size=self.kwargs['batch_size'],
            shuffle=False
        )
        self.model.eval()
        all_scores = []
        with torch.no_grad():
            for x_all, y in test_loader:
                x_all = x_all.to(self.device)
                batch_size, m, d = x_all.shape
                x_flat = x_all.view(-1, d)
                # 模型返回 (logits, proj), 我们只需要 logits
                logits, _ = self.model(x_flat)
                logits = logits.view(batch_size, m, -1)
                mean_logits = logits.mean(dim=1)
                var_logits = logits.var(dim=1, unbiased=False)
                risk_scores = mean_logits / (1.0 + self.kwargs['alpha'] * var_logits)
                all_scores.append(risk_scores.cpu().numpy())
        finall_scores = np.concatenate(all_scores, axis=0)
        result = pd.DataFrame(finall_scores, index=test_data.index, columns=self.unique_item)
        return result
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