# %%
import pandas as pd
import copy
from torch.utils.data import DataLoader
from .base import BaseRecommender
from src.network import *
# %%
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
            'backbone': 'Hybrid',
            'feature_dim': 32,
            'hidden_units': [256, 128],
            'dropout': 0.1,
            'cross_layers': 3,
            'attention_layers': 3,
            'num_heads': 2,
            'mask_prob': 0.1,  # 新增：稀疏特征 Mask 概率
            'noise_std': 0.01  # 新增：稠密特征噪声标准差
        }
        self.kwargs.update(default_params)
        self.kwargs.update(kwargs)
        self.user_name = sparse_features + dense_features
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        set_seed(self.seed)
    # %%
    def _build_model(self, sparse_dims, dense_count, num_classes):
        common_args = {
            'sparse_dims': sparse_dims,
            'dense_count': dense_count,
            'feature_dim': self.kwargs['feature_dim'],
            'hidden_units': self.kwargs['hidden_units'],
            'dropout': self.kwargs['dropout']
        }
        backbone_name = self.kwargs['backbone']
        if backbone_name == 'Hybrid':
            backbone = HybridBackbone(
                cross_layers=self.kwargs['cross_layers'],
                attention_layers=self.kwargs['attention_layers'],
                num_heads=self.kwargs['num_heads'],
                **common_args
            )
        elif backbone_name == 'DCNv2':
            backbone = DCNv2Backbone(
                cross_layers=self.kwargs['cross_layers'],
                **common_args
            )
        elif backbone_name == 'AutoInt':
            backbone = AutoIntBackbone(
                attention_layers=self.kwargs['attention_layers'],
                num_heads=self.kwargs['num_heads'],
                **common_args
            )
        elif backbone_name == 'FiBiNET':
            backbone = FiBiNETBackbone(**common_args)
        elif backbone_name == 'DeepFM':
            backbone = DeepFMBackbone(**common_args)
        elif backbone_name == 'WideDeep':
            backbone = WideDeepBackbone(**common_args)
        else:
            raise ValueError(f"Backbone '{backbone_name}' not supported. ")
        head = CoMICEHead(backbone.output_dim, num_classes, proj_dim=self.kwargs['proj_dim'])
        model = CoMICEModel(backbone, head).to(self.device)
        # 初始化增强模块
        self.augmenter = TabularAugmentation(
            num_sparse=len(self.sparse_features),
            num_dense=len(self.dense_features),
            mask_prob=self.kwargs['mask_prob'],
            noise_std=self.kwargs['noise_std']
        ).to(self.device)
        return model
    # %% 修改：返回 loss 而不是直接打印，方便在 fit 中统一管理日志
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
            # 4. 构建 Mask (创新点在这里)
            # Label Mask: 标记出 Batch 中 label 相同的样本对
            # y_cat: [y, y] -> [2B]
            y_cat = torch.cat([y, y], dim=0)
            # mask_label[i, j] = 1 if label[i] == label[j]
            mask_label = torch.eq(y_cat.unsqueeze(0), y_cat.unsqueeze(1)).float()
            # Identity Mask: 排除自身 (对角线)
            mask_self = torch.eye(2 * batch_size, device=self.device)
            # Positive Mask: 仅指 SimCLR 定义的正样本 (i, i+B) 和 (i+B, i)
            # 这是我们想要在分子中拉近的目标
            mask_pos = torch.zeros((2 * batch_size, 2 * batch_size), device=self.device)
            mask_pos[torch.arange(batch_size), torch.arange(batch_size) + batch_size] = 1
            mask_pos[torch.arange(batch_size) + batch_size, torch.arange(batch_size)] = 1
            # Negative Mask (分母的 Mask):
            # 原始 SimCLR: 只要不是自己，都是负样本 -> (1 - mask_self)
            # 创新改进 FN-Cancellation: 只要不是自己，且 *label不同*，才是负样本
            # 也就是说：如果 label 相同 (mask_label==1)，我们既不把它当正样本(分子)，也不把它当负样本(分母)，直接忽略
            # 逻辑：
            # 真正的负样本 = (Label不同) OR (是自身对应的增强视图-即原本的正样本对)
            # 注意：SimCLR公式中分母通常包含正样本项，为了标准实现，我们通常保留正样本在分母，剔除其他同label的项
            # 定义：允许在分母中出现的项 = (Label不同的项) + (SimCLR定义的正样本对)
            mask_valid_neg = (1 - mask_label) + mask_pos
            # 确保对角线不参与
            mask_valid_neg = mask_valid_neg * (1 - mask_self)
            # 限制为 0/1
            mask_valid_neg = (mask_valid_neg > 0).float()
            # 5. Compute Loss
            exp_sim = torch.exp(sim_matrix)
            # 分子: exp(sim(pos))
            pos_sim = (exp_sim * mask_pos).sum(dim=1)
            # 分母: sum(exp(sim(negatives)))
            # 只累加 valid_neg mask 为 1 的部分
            neg_sim_sum = (exp_sim * mask_valid_neg).sum(dim=1)
            # Log Prob
            # loss = -log ( pos / (pos + negs) )
            # 注意：SimCLR 标准公式分母包含正样本项。
            # mask_valid_neg 包含了 mask_pos，所以 neg_sim_sum 已经包含了 pos_sim
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
    # %% 修改后的 fit 方法，包含 Early Stopping
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
        # 2. 处理验证数据 (如果存在)
        valid_loader = None
        if valid_data is not None:
            # 关键：使用 fit_bool=False 防止数据泄露
            valid_data = self._mapping(valid_data, fit_bool=False)
            if self.standard_bool:
                valid_data = self._standardize(valid_data, fit_bool=False)
            X_val = torch.tensor(valid_data[self.user_name].values, dtype=torch.float32)
            y_val = torch.tensor(valid_data[self.item_name].values, dtype=torch.long)
            # 验证集不需要 shuffle，也不需要 batch_size 特别大，沿用即可
            valid_loader = DataLoader(RecDataset(X_val, y_val), batch_size=self.kwargs['batch_size'], shuffle=False)
        # 3. 初始化模型和优化器
        self.model = self._build_model(sparse_dims, dense_count, self.out_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.kwargs['lr'])
        # 4. 早停变量初始化
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_weights = None
        # 5. 训练循环
        for epoch in range(self.kwargs['epochs']):
            # 训练
            avg_loss, avg_ce, avg_nce = self.train_one_epoch(train_loader)
            log_msg = f"Epoch {epoch + 1}: Train Loss {avg_loss:.4f} (CE: {avg_ce:.4f}, NCE: {avg_nce:.4f})"
            # 验证
            if valid_loader is not None:
                self.model.eval()
                val_loss_sum = 0.0
                with torch.no_grad():
                    for x_v, y_v in valid_loader:
                        x_v, y_v = x_v.to(self.device), y_v.to(self.device)
                        logits_v, _ = self.model(x_v)  # CoMICE 返回 logits 和 proj
                        # 仅使用 CrossEntropy 作为早停指标（关注分类准确性）
                        loss_v = F.cross_entropy(logits_v, y_v)
                        val_loss_sum += loss_v.item()
                avg_val_loss = val_loss_sum / len(valid_loader)
                # Check Early Stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    best_model_weights = copy.deepcopy(self.model.state_dict())
                else:
                    patience_counter += 1
            print(log_msg)
            # 触发早停
            if valid_loader is not None and patience_counter >= patience:
                self.model.load_state_dict(best_model_weights)
                break
        # 如果训练跑满 epochs 且有验证集，恢复最佳权重
        if valid_loader is not None and best_model_weights is not None and patience_counter < patience:
            self.model.load_state_dict(best_model_weights)
        self.is_trained = True
    # %%
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