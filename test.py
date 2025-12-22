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
    def __init__(self, item_name: str, sparse_features: list, dense_features: list,
                 standard_bool: bool = True, seed: int = 42, k: int = 3, **kwargs):
        super().__init__('CoMICE', item_name, sparse_features, dense_features, standard_bool, seed, k)
        default_params = {
            'lr': 1e-4,
            'batch_size': 512,
            'feature_dim': 64,
            'proj_dim': 32,
            'epochs': 200,
            'lambda_nce': 1.0,
            'temperature': 0.1,
            'mice_method': 'MICE_Ga',
            'backbone': 'DCNv2',
            'cross_layers': 3,
            'hidden_units': [256, 128],
            'm': 3  # <--- 新增参数 m: 视图数量
        }
        self.kwargs.update(default_params)
        self.kwargs.update(kwargs)
        self.user_name = sparse_features + dense_features
        self.out_dim = None
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.optimizer = None
        self.imputer = None  # 仅保存主要的一个 imputer 用于测试
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

        for x, y in dataloader:
            # x shape: [Batch, m, Features]
            # y shape: [Batch]
            batch_size = x.size(0)
            m = self.kwargs['m']

            # 1. 整理多视图数据
            # 将 m 个视图展平，视为独立的样本输入模型
            # x shape becomes: [Batch * m, Features]
            x = x.view(batch_size * m, -1).to(self.device)

            # 2. 扩展标签
            # y_expanded shape: [Batch * m]
            # 例如 m=2, y=[0, 1] -> y_expanded=[0, 0, 1, 1]
            # 这样同一各样本的不同视图会有相同的标签
            y = y.to(self.device)
            y_expanded = y.repeat_interleave(m)

            # 3. 前向传播
            logits, proj = self.model(x)

            # 4. 计算分类损失 (在所有视图上)
            loss_ce = F.cross_entropy(logits, y_expanded)

            # 5. 计算对比损失 (NCE)
            # 相似度矩阵: [B*m, B*m]
            sim_matrix = torch.matmul(proj, proj.T) / self.kwargs['temperature']

            # 构建正样本 Mask
            # 只要标签相同（y_expanded相等），即为正样本。
            # 这涵盖了两种情况：
            # 1. 同一个样本的不同视图 (Augmentation/View Positives)
            # 2. 不同样本但属于同一类别 (Supervised Positives)
            mask = torch.eq(y_expanded.unsqueeze(0), y_expanded.unsqueeze(1)).float()
            mask.fill_diagonal_(0)  # 自己不能作为自己的正样本

            exp_sim = torch.exp(sim_matrix)
            diag_mask = torch.eye(batch_size * m, device=self.device).bool()
            exp_sim = exp_sim.masked_fill(diag_mask, 0)  # 分母中也不包含自己

            pos_sim = (exp_sim * mask).sum(dim=1)
            all_sim = exp_sim.sum(dim=1)

            eps = 1e-8
            # 注意：如果某个样本在Batch内没有其他正样本（且m=1），pos_sim可能为0。
            # 但这里 m >= 1，且包含同一样本的其他视图，所以 pos_sim 理论上不会为0（除非m=1且Batch内无同类）
            log_prob = torch.log(pos_sim / (all_sim + eps) + eps)
            loss_nce = -log_prob.mean()

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

        # 准备列表存储 m 个视图的数据
        X_views_list = []
        y_values = None

        for i in range(self.kwargs['m']):
            # 复制数据以防污染
            df_temp = train_data.copy()

            # 使用不同的随机种子生成不同的填补视图
            # 注意：filling 函数内部调用 select_interpolation，后者使用 seed
            current_seed = self.seed + i

            df_filled, imputer_instance = filling(df_temp, method=self.kwargs['mice_method'], seed=current_seed)
            df_filled = round(df_filled, self.sparse_features)

            # 关键：只在第1个视图上 fit 预处理参数 (mapping, scaling)
            # 后续视图使用相同的参数 transform，确保特征空间对齐
            if i == 0:
                self.imputer = imputer_instance  # 保存第一个imputer用于测试阶段
                df_filled = self._mapping(df_filled, fit_bool=True)
                if self.standard_bool:
                    df_filled = self._standardize(df_filled, fit_bool=True)
                # 保存标签 (所有视图标签一致)
                y_values = df_filled[self.item_name].values
            else:
                df_filled = self._mapping(df_filled, fit_bool=False)
                if self.standard_bool:
                    df_filled = self._standardize(df_filled, fit_bool=False)

            X_views_list.append(df_filled[self.user_name].values)

        # 堆叠数据: [N, m, D]
        x_train_numpy = np.stack(X_views_list, axis=1)

        x_train_tensor = torch.tensor(x_train_numpy, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_values, dtype=torch.long)

        dense_count = len(self.dense_features)
        sparse_dims = [self.vocabulary_sizes[col] for col in self.sparse_features]

        # Dataset 接收 3D Tensor
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
        if not self.is_trained:
            raise ValueError('model is not trained')

        # 测试阶段：使用 1 个视图 (self.imputer, 即训练时的第0个imputer)
        test_data = self.imputer.transform(test_data)
        test_data = round(test_data, self.sparse_features)
        test_data = self._mapping(test_data, fit_bool=False)
        if self.standard_bool:
            test_data = self._standardize(test_data, fit_bool=False)

        X_df = test_data[self.user_name]
        y_df = test_data[self.item_name]

        # 测试数据 shape: [N, D] (1个视图)
        # RecDataset 兼容处理：如果输入是 2D，getitem 返回 [D]，DataLoader batch 后为 [Batch, D]
        x_test_tensor = torch.tensor(X_df.values, dtype=torch.float32)
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

                # 如果是测试集(1个视图)，x shape 是 [Batch, Features]
                # 不需要像训练那样 reshape，直接输入模型即可
                logits, _ = self.model(x)
                probs = torch.softmax(logits, dim=1)
                all_scores.append(probs.cpu().numpy())

        final_scores = np.concatenate(all_scores, axis=0)
        result = pd.DataFrame(final_scores, index=test_data.index, columns=self.unique_item)
        return result
# %%
class RecDataset(Dataset):
    def __init__(self, X_imputed, y):
        # X_imputed 可以是 [N, m, D] (训练) 或 [N, D] (测试)
        self.X = X_imputed
        self.y = y
        self.N = X_imputed.shape[0]
    def __len__(self):
        return self.N
    def __getitem__(self, idx):
        # 返回单个样本
        # 训练时返回 [m, D], DataLoader 组装成 [Batch, m, D]
        # 测试时返回 [D], DataLoader 组装成 [Batch, D]
        return self.X[idx], self.y[idx]