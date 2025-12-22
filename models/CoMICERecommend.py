# %%
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
from .base import BaseRecommender
from src.utils import filling, round
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
            'lr': 1e-4,
            'batch_size': 128,
            'feature_dim': 64,
            'proj_dim': 32,
            'epochs': 200,
            'lambda_nce': 1.0,
            'temperature': 0.1,
            'mice_method': 'MICE_Ga',
            'backbone': 'DeepFM',
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
        self.imputer = None  # 变为单个 imputer
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
        # 这里可以使用你优化过的 CoMICEHead
        head = CoMICEHead(input_dim=backbone.output_dim, num_classes=num_classes, proj_dim=self.kwargs['proj_dim'])
        return CoMICEModel(backbone, head).to(self.device)
    # %%
    def train_one_epoch(self, dataloader):
        self.model.train()
        total_loss, loss_ce, loss_nce = 0.0, 0.0, 0.0
        for x, y in dataloader:
            # x shape: [batch_size, input_dim]
            # y shape: [batch_size]
            batch_size = x.size(0)
            x, y = x.to(self.device), y.to(self.device)
            # 2. 前向传播
            # 直接输入 x，不需要 view/flatten，因为已经是 [B, D]
            logits, proj = self.model(x)
            # --- Loss 1: Cross Entropy (分类损失) ---
            # 不需要 y.repeat_interleave，因为是一对一
            loss_ce = F.cross_entropy(logits, y)
            # --- Loss 2: Supervised Contrastive Loss (有监督对比损失) ---
            # 这里的 batch 内所有样本互为对比对象
            # A. 计算相似度矩阵 [B, B]
            sim_matrix = torch.matmul(proj, proj.T) / self.kwargs['temperature']
            # [数值稳定技巧] 减去每行的最大值
            sim_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
            sim_matrix = sim_matrix - sim_max.detach()
            # B. 构建有监督 Mask
            # 只要标签相同 (y_i == y_j) 即为正样本
            # mask shape: [B, B]
            mask = torch.eq(y.unsqueeze(0), y.unsqueeze(1)).float()
            # C. 排除自身 (Diagonal)
            mask.fill_diagonal_(0)
            # D. 计算 Exp
            exp_sim = torch.exp(sim_matrix)
            # 剔除对角线
            diag_mask = torch.eye(batch_size, device=self.device).bool()
            exp_sim = exp_sim.masked_fill(diag_mask, 0)
            # E. 计算 InfoNCE
            # 分子：所有同类样本（正样本）的 exp 之和
            pos_sim = (exp_sim * mask).sum(dim=1)
            # 分母：所有样本（负样本+正样本，但不含自身）的 exp 之和
            all_sim = exp_sim.sum(dim=1)
            # 计算 Log Probability
            eps = 1e-8
            # 如果 batch 内没有同类样本，pos_sim 可能为 0，导致 log(0)。
            # 实际上 SupCon Loss 通常只对有正样本的 anchor 计算 loss。
            # 这里简单处理：加 eps
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
        # 1. 映射和标准化
        train_data = self._mapping(train_data, fit_bool=True)
        if self.standard_bool:
            train_data = self._standardize(train_data, fit_bool=True)
        rng = np.random.default_rng(seed=self.seed)
        seed_val = rng.choice(1000)
        # 2. 插补 (Filling 内部已经包含了 round)
        # 注意：这里我们只进行一次插补，因为 m=1
        data_imputed, self.imputer = filling(train_data, method=self.kwargs['mice_method'], seed=seed_val)
        # 3. 手动转换为 Tensor (替代 process_mice_list)
        # 提取特征列和标签列
        X_df = data_imputed[self.user_name]
        y_df = data_imputed[self.item_name]
        # 转换为 Tensor
        # 特征通常是 float (Backbone 内部会将 sparse 列转为 long，但输入通常保持 float 格式以便统一处理)
        x_train_tensor = torch.tensor(X_df.values, dtype=torch.float32)
        # 标签是分类索引，必须是 long
        y_train_tensor = torch.tensor(y_df.values, dtype=torch.long)
        dense_count = len(self.dense_features)
        sparse_dims = [self.vocabulary_sizes[col] for col in self.sparse_features]
        # 4. 创建 Dataset 和 DataLoader
        train_loader = DataLoader(
            RecDataset(x_train_tensor, y_train_tensor),
            batch_size=self.kwargs['batch_size'],
            shuffle=True
        )
        # 5. 构建模型和优化器
        self.model = self._build_model(sparse_dims, dense_count, self.out_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.kwargs['lr'])
        # 6. 训练循环
        for epoch in range(self.kwargs['epochs']):
            self.train_one_epoch(train_loader)
        self.is_trained = True
    # %%
    def get_proba(self, test_data: pd.DataFrame):
        test_data = self._mapping(test_data, fit_bool=False)
        if self.standard_bool:
            test_data = self._standardize(test_data, fit_bool=False)
        # 1. 插补 (Transform)
        data = self.imputer.transform(test_data)
        # 2. 取整 (Round) - 必须显式调用，因为 transform 不包含 round
        data = round(data)
        # 3. 手动转换为 Tensor (替代 process_mice_list)
        X_df = data[self.user_name]
        x_test_tensor = torch.tensor(X_df.values, dtype=torch.float32)
        # 为了兼容 Dataset，我们需要一个 dummy y，或者如果 test_data 有标签就提取标签
        y_df = data[self.item_name]
        y_test_tensor = torch.tensor(y_df.values, dtype=torch.long)
        test_loader = DataLoader(
            RecDataset(x_test_tensor, y_test_tensor),
            batch_size=self.kwargs['batch_size'],
            shuffle=False
        )
        self.model.eval()
        all_scores = []
        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(self.device)
                # x shape: [B, d]
                logits, _ = self.model(x)
                probs = torch.softmax(logits, dim=1)
                all_scores.append(probs.cpu().numpy())
        final_scores = np.concatenate(all_scores, axis=0)
        return pd.DataFrame(final_scores, index=test_data.index, columns=self.unique_item)
# %%
class RecDataset(Dataset):
    def __init__(self, X_imputed, y):
        """
        X_imputed: Tensor of shape [N, d] (Single View)
        y: Tensor of shape [N]
        """
        self.X = X_imputed
        self.y = y
        self.N = X_imputed.shape[0]
    def __len__(self):
        return self.N
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]