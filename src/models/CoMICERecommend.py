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
            'lr': 1e-4,
            'batch_size': 512,
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
        return CoMICEModel(backbone, head).to(self.device)
    # %% 修改：返回 loss 而不是直接打印，方便在 fit 中统一管理日志
    def train_one_epoch(self, dataloader):
        self.model.train()
        total_loss, ce_acc, nce_acc = 0.0, 0.0, 0.0
        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            logits, proj = self.model(x)
            # 1. Cross Entropy Loss (Main Task)
            loss_ce = F.cross_entropy(logits, y)
            # 2. Contrastive Loss (Auxiliary Task)
            batch_size = x.size(0)
            sim_matrix = torch.matmul(proj, proj.T) / self.kwargs['temperature']
            sim_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
            sim_matrix = sim_matrix - sim_max.detach()
            # Mask: Identify positive pairs (same label)
            mask = torch.eq(y.unsqueeze(0), y.unsqueeze(1)).float()
            mask.fill_diagonal_(0)
            exp_sim = torch.exp(sim_matrix)
            exp_sim = exp_sim.masked_fill(torch.eye(batch_size, device=self.device).bool(), 0)
            pos_sim = (exp_sim * mask).sum(dim=1)
            all_sim = exp_sim.sum(dim=1)
            log_prob = torch.log(pos_sim / (all_sim + 1e-8) + 1e-8)
            valid_mask = mask.sum(dim=1) > 0
            if valid_mask.sum() > 0:
                loss_nce = -log_prob[valid_mask].mean()
            else:
                loss_nce = torch.tensor(0.0).to(self.device)
            # Total Loss
            loss = loss_ce + self.kwargs['lambda_nce'] * loss_nce
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            ce_acc += loss_ce.item()
            nce_acc += loss_nce.item()
        # 返回平均损失
        n = len(dataloader)
        return total_loss / n, ce_acc / n, nce_acc / n
    # %% 修改后的 fit 方法，包含 Early Stopping
    def fit(self, train_data: pd.DataFrame, valid_data: pd.DataFrame = None, patience: int = 5):
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