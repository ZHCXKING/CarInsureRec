import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import copy
import random
import os
# 假设 BaseRecommender 在同级目录或父级目录，如果无法导入请自行调整
# from .base import BaseRecommender
# 为了代码独立运行，这里提供一个简化的 BaseRecommender 占位，实际使用请替换为你的 import
class BaseRecommender:
    def __init__(self, model_name, item_name, sparse_features, dense_features, standard_bool, seed, k):
        self.model_name = model_name
        self.item_name = item_name
        self.sparse_features = sparse_features
        self.dense_features = dense_features
        self.standard_bool = standard_bool
        self.seed = seed
        self.k = k
        self.vocabulary_sizes = {}  # 需要在 _mapping 中填充
        self.kwargs = {}
        self.is_trained = False
    def _mapping(self, data, fit_bool=True):
        # 简易实现：Label Encoding
        data = data.copy()
        for col in self.sparse_features:
            if fit_bool:
                uniques = sorted(data[col].unique())
                self.vocabulary_sizes[col] = len(uniques) + 1  # +1 for unknown
                self.mapping_dict = {v: i + 1 for i, v in enumerate(uniques)}  # 简化的 mapping 保存
                # 实际生产代码需要更严谨的 mapping 保存机制
            # 这里简化，假设 fit_bool=False 时也能处理，实际需保存 encoder
            data[col] = data[col].astype('category').cat.codes + 1

        # Target Encoding
        if fit_bool:
            self.target_map = {v: i for i, v in enumerate(sorted(data[self.item_name].unique()))}
        data[self.item_name] = data[self.item_name].map(self.target_map).fillna(0).astype(int)
        return data
    def _standardize(self, data, fit_bool=True):
        # 简易实现：StandardScaler
        data = data.copy()
        if not self.dense_features: return data
        if fit_bool:
            self.dense_mean = data[self.dense_features].mean()
            self.dense_std = data[self.dense_features].std() + 1e-8
        data[self.dense_features] = (data[self.dense_features] - self.dense_mean) / self.dense_std
        return data
class RecDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# ================== 核心组件 (Layers) ==================

class DCNv2Layer(nn.Module):
    """
    DCN V2 Cross Layer (Low-Rank Version)
    Formula: x_{l+1} = x_0 \odot (U_l V_l^T x_l + b_l) + x_l
    """
    def __init__(self, input_dim, low_rank=64):
        super().__init__()
        self.input_dim = input_dim
        # V projection: d -> r
        self.v_linear = nn.Linear(input_dim, low_rank, bias=False)
        # U projection: r -> d (includes bias b_l)
        self.u_linear = nn.Linear(low_rank, input_dim, bias=True)

        # Init
        nn.init.xavier_uniform_(self.v_linear.weight)
        nn.init.xavier_uniform_(self.u_linear.weight)
    def forward(self, x_l, x_0):
        # x_l, x_0: [Batch, d]
        # Project down: [B, r]
        v_out = self.v_linear(x_l)
        # Project up and add bias: [B, d] -> (W x_l + b)
        u_out = self.u_linear(v_out)
        # Hadamard product with x_0 and add residual
        return x_0 * u_out + x_l
class DNN(nn.Module):
    def __init__(self, input_dim, hidden_units, dropout=0.0):
        super().__init__()
        layers = []
        in_dim = input_dim
        for hidden in hidden_units:
            layers.extend([
                nn.Linear(in_dim, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden
        self.net = nn.Sequential(*layers)
        self.output_dim = in_dim
    def forward(self, x):
        return self.net(x)
class TabularAugmentation(nn.Module):
    def __init__(self, num_sparse, num_dense, mask_prob=0.2, noise_std=0.0):
        super().__init__()
        self.num_sparse = num_sparse
        self.num_dense = num_dense
        self.mask_prob = mask_prob
        self.noise_std = noise_std
    def forward(self, x):
        if not self.training:
            return x
        x_aug = x.clone()

        # 1. Mask Sparse Features (前 num_sparse 列)
        if self.num_sparse > 0 and self.mask_prob > 0:
            sparse_part = x_aug[:, :self.num_sparse]
            mask = torch.rand_like(sparse_part, dtype=torch.float) < self.mask_prob
            sparse_part[mask] = 0.0
            x_aug[:, :self.num_sparse] = sparse_part

        # 2. Add Noise to Dense Features (后 num_dense 列)
        if self.num_dense > 0 and self.noise_std > 0:
            dense_part = x_aug[:, self.num_sparse:]
            noise = torch.randn_like(dense_part) * self.noise_std
            x_aug[:, self.num_sparse:] = dense_part + noise

        return x_aug
class CoMICEHead(nn.Module):
    def __init__(self, input_dim, num_classes, proj_dim=128, hidden_dim=256, dropout_rate=0.1):
        super().__init__()
        # Projection Head for Contrastive Learning
        self.projection_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim)
        )
        # Classifier Head for Recommendation Task
        self.classifier_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
    def forward(self, feature_vector):
        # Contrastive Branch
        proj_feat = self.projection_head(feature_vector)
        proj_norm = F.normalize(proj_feat, dim=1)
        # Classification Branch
        logits = self.classifier_head(feature_vector)
        return logits, proj_norm
# ================== 核心模型 (Backbone & Wrapper) ==================

class HybridBackbone(nn.Module):
    def __init__(self, sparse_dims, dense_count, feature_dim=32,
                 cross_layers=3, attention_layers=2, num_heads=2,
                 hidden_units=[256, 128], dropout=0.1,
                 low_rank_dim=64):
        super().__init__()
        self.num_sparse = len(sparse_dims)
        self.num_dense = dense_count

        # 1. Embedding Layer
        # 为每个稀疏特征建立 Embedding
        self.sparse_embs = nn.ModuleList([nn.Embedding(d, feature_dim) for d in sparse_dims])
        # 将每个稠密特征投影到同样的维度
        if dense_count > 0:
            self.dense_proj = nn.ModuleList([nn.Linear(1, feature_dim) for _ in range(dense_count)])

        self.num_fields = self.num_sparse + self.num_dense
        self.total_input_dim = self.num_fields * feature_dim

        # 2. Cross Tower (DCNv2)
        # 输入维度: Flatten后的 Embedding 拼接
        self.dcn_layers = nn.ModuleList([
            DCNv2Layer(self.total_input_dim, low_rank=low_rank_dim)
            for _ in range(cross_layers)
        ])

        # 3. Deep Tower (DNN)
        # 输入维度: Flatten后的 Embedding 拼接
        self.deep_net = DNN(self.total_input_dim, hidden_units, dropout=dropout)

        # 4. Attention Tower (Transformer)
        # 输入维度: [Batch, Fields, Emb_Dim] (保留字段结构)
        # 去掉了 SeNet，直接处理 Embedding
        encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=num_heads,
                                                   dim_feedforward=feature_dim * 2,
                                                   dropout=dropout, batch_first=True)
        self.att_encoder = nn.TransformerEncoder(encoder_layer, num_layers=attention_layers)

        # Attention Output Dimensions (Pooling: Mean + Max)
        self.att_out_dim = feature_dim * 2

        # Normalization Layers
        self.ln_cross = nn.LayerNorm(self.total_input_dim)
        self.ln_deep = nn.LayerNorm(self.deep_net.output_dim)
        self.ln_att = nn.LayerNorm(self.att_out_dim)

        # Final Output Dimension = Cross + Deep + Attention
        self.output_dim = self.total_input_dim + self.deep_net.output_dim + self.att_out_dim
    def forward(self, x):
        # --- 1. Embedding Lookup & Projection ---
        # 处理 Sparse
        sparse_x = x[:, :self.num_sparse].long()
        embs = [emb(sparse_x[:, i]) for i, emb in enumerate(self.sparse_embs)]

        # 处理 Dense
        if self.num_dense > 0:
            dense_x = x[:, self.num_sparse:]
            embs.extend([self.dense_proj[i](dense_x[:, i].unsqueeze(1)) for i in range(self.num_dense)])

        # Stacked: [Batch, Fields, Emb_Dim]
        stacked_embs = torch.stack(embs, dim=1)

        # Flattened: [Batch, Fields * Emb_Dim] -> 用于 DCN 和 DNN
        flat_input = stacked_embs.view(x.size(0), -1)

        # --- 2. Cross Tower (DCNv2) ---
        # DCNv2 requires x_0 (original input) for residual connection
        x_0 = flat_input
        cross_out = x_0
        for layer in self.dcn_layers:
            cross_out = layer(cross_out, x_0)
        cross_out = self.ln_cross(cross_out)

        # --- 3. Deep Tower (DNN) ---
        deep_out = self.deep_net(flat_input)
        deep_out = self.ln_deep(deep_out)

        # --- 4. Attention Tower (Transformer) ---
        # Input: [Batch, Fields, Emb_Dim]
        att_feat = self.att_encoder(stacked_embs)

        # Pooling: Global Avg + Global Max
        avg_pool = torch.mean(att_feat, dim=1)
        max_pool = torch.max(att_feat, dim=1)[0]
        att_out = torch.cat([avg_pool, max_pool], dim=1)
        att_out = self.ln_att(att_out)

        # --- 5. Fusion ---
        final_out = torch.cat([cross_out, deep_out, att_out], dim=1)
        return final_out
class CoMICENet(nn.Module):
    """
    Wrapper model that connects Backbone and Head
    """
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head
    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)
# ================== 主推荐类 (Main Class) ==================

class CoMICERecommend(BaseRecommender):
    def __init__(self, item_name: str, sparse_features: list, dense_features: list,
                 standard_bool: bool = True, seed: int = 42, k: int = 3, **kwargs):
        super().__init__('CoMICE', item_name, sparse_features, dense_features, standard_bool, seed, k)

        default_params = {
            'lr': 1e-3,
            'batch_size': 1024,
            'epochs': 200,
            'lambda_nce': 1.0,
            'temperature': 0.1,
            'proj_dim': 32,
            # Backbone params
            'feature_dim': 32,
            'hidden_units': [256, 128],
            'dropout': 0.1,
            'cross_layers': 3,
            'low_rank_dim': 64,  # DCN low rank
            'attention_layers': 2,
            'num_heads': 2,
            # Augmentation params
            'mask_prob': 0.1,
            'noise_std': 0.01
        }
        self.kwargs.update(default_params)
        self.kwargs.update(kwargs)

        self.user_name = sparse_features + dense_features
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        set_seed(self.seed)
    def _build_model(self, sparse_dims, dense_count, num_classes):
        # 直接构建 HybridBackbone，不再进行条件判断
        backbone = HybridBackbone(
            sparse_dims=sparse_dims,
            dense_count=dense_count,
            feature_dim=self.kwargs['feature_dim'],
            cross_layers=self.kwargs['cross_layers'],
            attention_layers=self.kwargs['attention_layers'],
            num_heads=self.kwargs['num_heads'],
            hidden_units=self.kwargs['hidden_units'],
            dropout=self.kwargs['dropout'],
            low_rank_dim=self.kwargs['low_rank_dim']
        )

        head = CoMICEHead(
            input_dim=backbone.output_dim,
            num_classes=num_classes,
            proj_dim=self.kwargs['proj_dim'],
            dropout_rate=self.kwargs['dropout']
        )

        model = CoMICENet(backbone, head).to(self.device)

        # 初始化增强模块
        self.augmenter = TabularAugmentation(
            num_sparse=len(self.sparse_features),
            num_dense=len(self.dense_features),
            mask_prob=self.kwargs['mask_prob'],
            noise_std=self.kwargs['noise_std']
        ).to(self.device)

        return model
    def train_one_epoch(self, dataloader):
        self.model.train()
        total_loss, ce_acc, nce_acc = 0.0, 0.0, 0.0

        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            batch_size = x.size(0)

            # --- Task 1: Main Classification Task ---
            logits_clean, _ = self.model(x)
            loss_ce = F.cross_entropy(logits_clean, y)

            # --- Task 2: Label-Aware Self-Supervised Task ---
            # 1. Augmentation
            x_view1 = self.augmenter(x)
            x_view2 = self.augmenter(x)

            # 2. Projection
            _, proj1 = self.model(x_view1)
            _, proj2 = self.model(x_view2)
            features = torch.cat([proj1, proj2], dim=0)  # [2B, D]

            # 3. Similarity Matrix
            sim_matrix = torch.matmul(features, features.T) / self.kwargs['temperature']
            sim_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
            sim_matrix = sim_matrix - sim_max.detach()  # Stability

            # 4. 构建 Mask
            # Label Mask
            y_cat = torch.cat([y, y], dim=0)
            mask_label = torch.eq(y_cat.unsqueeze(0), y_cat.unsqueeze(1)).float()

            # Identity Mask
            mask_self = torch.eye(2 * batch_size, device=self.device)

            # Positive Mask (SimCLR definition: i and i+B)
            mask_pos = torch.zeros((2 * batch_size, 2 * batch_size), device=self.device)
            mask_pos[torch.arange(batch_size), torch.arange(batch_size) + batch_size] = 1
            mask_pos[torch.arange(batch_size) + batch_size, torch.arange(batch_size)] = 1

            # Negative Mask Calculation (FN-Cancellation)
            mask_valid_neg = (1 - mask_label) + mask_pos
            mask_valid_neg = mask_valid_neg * (1 - mask_self)
            mask_valid_neg = (mask_valid_neg > 0).float()

            # 5. Compute Loss
            exp_sim = torch.exp(sim_matrix)
            pos_sim = (exp_sim * mask_pos).sum(dim=1)
            neg_sim_sum = (exp_sim * mask_valid_neg).sum(dim=1)

            log_prob = -torch.log(pos_sim / (neg_sim_sum + 1e-8) + 1e-8)
            loss_nce = log_prob.mean()

            # Total Loss
            loss = loss_ce + self.kwargs['lambda_nce'] * loss_nce

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            ce_acc += loss_ce.item()
            nce_acc += loss_nce.item()

        n = len(dataloader)
        return total_loss / n, ce_acc / n, nce_acc / n
    def fit(self, train_data: pd.DataFrame, valid_data: pd.DataFrame = None, patience: int = 10):
        self.out_dim = train_data[self.item_name].nunique()
        self.unique_item = list(range(self.out_dim))

        # 1. 处理训练数据
        train_data = self._mapping(train_data, fit_bool=True)
        if self.standard_bool:
            train_data = self._standardize(train_data, fit_bool=True)

        X_tensor = torch.tensor(train_data[self.user_name].values, dtype=torch.float32)
        y_tensor = torch.tensor(train_data[self.item_name].values, dtype=torch.long)

        sparse_dims = [self.vocabulary_sizes[col] for col in self.sparse_features]
        dense_count = len(self.dense_features)

        train_loader = DataLoader(RecDataset(X_tensor, y_tensor), batch_size=self.kwargs['batch_size'], shuffle=True)

        # 2. 处理验证数据
        valid_loader = None
        if valid_data is not None:
            valid_data = self._mapping(valid_data, fit_bool=False)
            if self.standard_bool:
                valid_data = self._standardize(valid_data, fit_bool=False)
            X_val = torch.tensor(valid_data[self.user_name].values, dtype=torch.float32)
            y_val = torch.tensor(valid_data[self.item_name].values, dtype=torch.long)
            valid_loader = DataLoader(RecDataset(X_val, y_val), batch_size=self.kwargs['batch_size'], shuffle=False)

        # 3. 初始化模型和优化器
        self.model = self._build_model(sparse_dims, dense_count, self.out_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.kwargs['lr'])

        # 4. 早停变量
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_weights = None

        # 5. 训练循环
        for epoch in range(self.kwargs['epochs']):
            avg_loss, avg_ce, avg_nce = self.train_one_epoch(train_loader)
            log_msg = f"Epoch {epoch + 1}: Train Loss {avg_loss:.4f} (CE: {avg_ce:.4f}, NCE: {avg_nce:.4f})"

            if valid_loader is not None:
                self.model.eval()
                val_loss_sum = 0.0
                with torch.no_grad():
                    for x_v, y_v in valid_loader:
                        x_v, y_v = x_v.to(self.device), y_v.to(self.device)
                        logits_v, _ = self.model(x_v)
                        loss_v = F.cross_entropy(logits_v, y_v)
                        val_loss_sum += loss_v.item()

                avg_val_loss = val_loss_sum / len(valid_loader)
                log_msg += f" | Val CE Loss: {avg_val_loss:.4f}"

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    best_model_weights = copy.deepcopy(self.model.state_dict())
                else:
                    patience_counter += 1

            print(log_msg)

            if valid_loader is not None and patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                self.model.load_state_dict(best_model_weights)
                break

        if valid_loader is not None and best_model_weights is not None and patience_counter < patience:
            self.model.load_state_dict(best_model_weights)

        self.is_trained = True
    def get_proba(self, test_data: pd.DataFrame):
        if not self.is_trained:
            raise ValueError('Model not trained')

        test_data = self._mapping(test_data, fit_bool=False)
        if self.standard_bool:
            test_data = self._standardize(test_data, fit_bool=False)

        X_tensor = torch.tensor(test_data[self.user_name].values, dtype=torch.float32)
        y_tensor = torch.tensor(test_data[self.item_name].values, dtype=torch.long)

        loader = DataLoader(RecDataset(X_tensor, y_tensor), batch_size=self.kwargs['batch_size'], shuffle=False)
        self.model.eval()

        all_probs = []
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(self.device)
                logits, _ = self.model(x)
                all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())

        return pd.DataFrame(np.concatenate(all_probs, axis=0), index=test_data.index, columns=self.unique_item)