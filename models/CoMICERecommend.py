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
            'm': 1,
            'lr': 1e-4,
            'batch_size': 128,
            'feature_dim': 64,
            'proj_dim': 32,
            'epochs': 200,
            'lambda_nce': 1.0,
            'temperature': 0.1,
            'mice_method': 'MICE_Ga',
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
        total_loss, loss_ce, loss_nce = 0.0, 0.0, 0.0
        m = self.kwargs['m']
        for x_all, y in dataloader:
            # x_all shape: [batch_size, m, input_dim]
            batch_size = x_all.size(0)
            x_all, y = x_all.to(self.device), y.to(self.device)
            # 1. 扁平化输入 [Batch * m, Dim]
            x_flat = x_all.view(-1, x_all.size(-1))
            # 2. 前向传播
            logits_flat, proj_flat = self.model(x_flat)
            # --- 扩展标签 ---
            # y_expanded shape: [Batch * m]
            # 例如 batch y=[0, 1], m=2 -> y_expanded=[0, 0, 1, 1]
            y_expanded = y.repeat_interleave(m)
            # --- Loss 1: Cross Entropy (分类损失) ---
            loss_ce = F.cross_entropy(logits_flat, y_expanded)
            # --- Loss 2: Supervised Contrastive Loss (有监督对比损失) ---
            # A. 计算相似度矩阵 [BM, BM]
            sim_matrix = torch.matmul(proj_flat, proj_flat.T) / self.kwargs['temperature']
            # [数值稳定技巧] 减去每行的最大值，防止 exp 后溢出
            sim_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
            sim_matrix = sim_matrix - sim_max.detach()
            # B. 构建有监督 Mask
            # 只要标签相同 (y_i == y_j) 即为正样本
            # mask shape: [BM, BM]
            mask = torch.eq(y_expanded.unsqueeze(0), y_expanded.unsqueeze(1)).float()
            # C. 排除自身 (Diagonal)
            # 自身与自身的对比不产生梯度，且会主导 loss，需剔除
            mask.fill_diagonal_(0)
            # D. 计算 Exp
            exp_sim = torch.exp(sim_matrix)
            # 同样在 exp_sim 中剔除对角线（为了计算分母 sum(all_sim) 时不包含自身）
            diag_mask = torch.eye(batch_size * m, device=self.device).bool()
            exp_sim = exp_sim.masked_fill(diag_mask, 0)
            # E. 计算 InfoNCE
            # 分子：所有同类样本（正样本）的 exp 之和
            pos_sim = (exp_sim * mask).sum(dim=1)
            # 分母：所有样本（负样本+正样本，但不含自身）的 exp 之和
            all_sim = exp_sim.sum(dim=1)
            # 计算 Log Probability
            # 加 eps 防止 log(0)
            eps = 1e-8
            # 这里的逻辑是：log(Sum_Pos / Sum_All)
            # 注意：如果某样本在 Batch 里没有同类（mask行为全0），pos_sim 为 0，这会导致 loss 极大。
            # 为了稳健性，可以只对“有正样本”的行计算 loss，或者加上 eps
            log_prob = torch.log(pos_sim / (all_sim + eps) + eps)
            # loss_nce 取负均值
            loss_nce = -log_prob.mean()
            # 总损失
            loss = loss_ce + self.kwargs['lambda_nce'] * loss_nce
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        print(f"loss_ce: {loss_ce:.4f}, loss_nce: {loss_nce:.4f}")
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
        # 1. 基础预处理
        test_data = self._mapping(test_data, fit_bool=False)
        if self.standard_bool:
            test_data = self._standardize(test_data, fit_bool=False)
        # 2. 生成多视角插补数据 (Multiple Imputation)
        # 遍历训练时拟合好的所有 imputer，生成 m 个版本的测试集
        test_data_sets = []
        for imputer in self.imputers:
            # 使用训练好的 imputer 转换测试数据
            data = imputer.transform(test_data)
            # CRITICAL: 必须进行 round 操作，确保浮点数转为与训练一致的整数索引
            data = round(data)
            test_data_sets.append(data)
        # 3. 转换为 Tensor [N, m, d]
        # process_mice_list 会将列表堆叠为 [N, m, d] 的形状
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
                # x_all shape: [batch_size, m, d]
                batch_size, m, d = x_all.shape
                # 4. 扁平化以进行批量推理
                x_flat = x_all.view(-1, d)  # [batch_size * m, d]
                # 前向传播
                logits, _ = self.model(x_flat)  # logits: [batch_size * m, num_classes]
                # 5. 计算概率 (Softmax)
                probs_flat = torch.softmax(logits, dim=1)
                # 6. 聚合多视角结果 (Ensemble / Soft Voting)
                # 将形状还原为 [batch_size, m, num_classes]
                probs_view = probs_flat.view(batch_size, m, -1)
                # 在 m 维度取平均值，得到该 batch 每个样本的最终概率
                # shape: [batch_size, num_classes]
                probs_mean = probs_view.mean(dim=1)
                all_scores.append(probs_mean.cpu().numpy())
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
