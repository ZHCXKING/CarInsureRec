# %%
import pandas as pd
import copy
from torch.utils.data import DataLoader
from .base import BaseRecommender
from src.network import *
# %%
class StandardModel(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.classifier_head = nn.Linear(backbone.output_dim, num_classes)
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier_head(features)
        return logits
# %%
class NetworkRecommender(BaseRecommender):
    def __init__(self, model_name: str, backbone_class, item_name: str, sparse_features: list, dense_features: list,
                 standard_bool: bool = True, seed: int = 42, k: int = 3, **kwargs):
        super().__init__(model_name, item_name, sparse_features, dense_features, standard_bool, seed, k)
        self.backbone_class = backbone_class
        default_params = {
            'lr': 1e-3,
            'batch_size': 1024,
            'feature_dim': 32,
            'epochs': 200,
            'hidden_units': [256, 128],
            'cross_layers': 3,
            'dropout': 0.1,
            'attention_layers': 3,
            'num_heads': 2
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
        # 修正原代码中的逻辑错误: check if class is in list
        if self.backbone_class == HybridBackbone:
            backbone = HybridBackbone(
                cross_layers=self.kwargs['cross_layers'],
                attention_layers=self.kwargs['attention_layers'],
                num_heads=self.kwargs['num_heads'],
                **common_args
            )
        elif self.backbone_class == DCNv2Backbone:
            backbone = self.backbone_class(
                cross_layers=self.kwargs['cross_layers'],
                **common_args
            )
        elif self.backbone_class == AutoIntBackbone:
            backbone = self.backbone_class(
                attention_layers=self.kwargs['attention_layers'],
                num_heads=self.kwargs['num_heads'],
                **common_args
            )
        else:
            backbone = self.backbone_class(**common_args)
        return StandardModel(backbone, num_classes).to(self.device)
    # %% 修改后的 fit 方法，包含 Early Stopping
    def fit(self, train_data: pd.DataFrame, valid_data: pd.DataFrame = None, patience: int = 10):
        """
        :param patience: 容忍多少个 epoch 验证集 loss 不下降
        """
        self.out_dim = train_data[self.item_name].nunique()
        self.unique_item = list(range(self.out_dim))
        # 1. 处理训练数据 (fit_bool=True)
        train_data = self._mapping(train_data, fit_bool=True)
        if self.standard_bool:
            train_data = self._standardize(train_data, fit_bool=True)
        X_train = torch.tensor(train_data[self.user_name].values, dtype=torch.float32)
        y_train = torch.tensor(train_data[self.item_name].values, dtype=torch.long)
        sparse_dims = [self.vocabulary_sizes[col] for col in self.sparse_features]
        dense_count = len(self.dense_features)
        train_loader = DataLoader(RecDataset(X_train, y_train), batch_size=self.kwargs['batch_size'], shuffle=True)
        # 2. 处理验证数据 (如果存在)
        valid_loader = None
        if valid_data is not None:
            # 注意：验证集必须使用 fit_bool=False，沿用训练集的 mapping 和 scaler
            valid_data = self._mapping(valid_data, fit_bool=False)
            if self.standard_bool:
                valid_data = self._standardize(valid_data, fit_bool=False)
            X_val = torch.tensor(valid_data[self.user_name].values, dtype=torch.float32)
            y_val = torch.tensor(valid_data[self.item_name].values, dtype=torch.long)
            # 验证集不需要 shuffle
            valid_loader = DataLoader(RecDataset(X_val, y_val), batch_size=self.kwargs['batch_size'], shuffle=False)
        # 3. 初始化模型
        self.model = self._build_model(sparse_dims, dense_count, self.out_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.kwargs['lr'])
        criterion = nn.CrossEntropyLoss()
        # 4. 早停相关变量初始化
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_weights = None
        # 5. 训练循环
        for epoch in range(self.kwargs['epochs']):
            # --- Training ---
            self.model.train()
            train_loss = 0.0
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(x)
                loss = criterion(logits, y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            avg_train_loss = train_loss / len(train_loader)
            # --- Validation & Early Stopping ---
            if valid_loader is not None:
                self.model.eval()
                valid_loss = 0.0
                with torch.no_grad():
                    for x_v, y_v in valid_loader:
                        x_v, y_v = x_v.to(self.device), y_v.to(self.device)
                        logits_v = self.model(x_v)
                        loss_v = criterion(logits_v, y_v)
                        valid_loss += loss_v.item()
                avg_val_loss = valid_loss / len(valid_loader)
                print(f"Epoch {epoch + 1}: Train Loss {avg_train_loss:.4f} | Val Loss {avg_val_loss:.4f}")
                # 检查是否更优
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    # 在内存中保存最佳模型权重
                    best_model_weights = copy.deepcopy(self.model.state_dict())
                else:
                    patience_counter += 1
                # 触发早停
                if patience_counter >= patience:
                    self.model.load_state_dict(best_model_weights)
                    break
            else:
                # 如果没有验证集，只打印训练 Loss
                print(f"Epoch {epoch + 1}: Train Loss {avg_train_loss:.4f}")
        # 如果训练结束且保存了最佳权重（针对没有触发早停但跑完了所有epoch的情况），恢复最佳权重
        if best_model_weights is not None:
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
                logits = self.model(x)
                all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        final_probs = np.concatenate(all_probs, axis=0)
        return pd.DataFrame(final_probs, index=test_data.index, columns=self.unique_item)
# %%
class HybridRecommend(NetworkRecommender):
    def __init__(self, item_name, sparse_features, dense_features, **kwargs):
        super().__init__('Hybrid', HybridBackbone, item_name, sparse_features, dense_features, **kwargs)
# %%
class DCNv2Recommend(NetworkRecommender):
    def __init__(self, item_name, sparse_features, dense_features, **kwargs):
        super().__init__('DCNv2', DCNv2Backbone, item_name, sparse_features, dense_features, **kwargs)
# %%
class DeepFMRecommend(NetworkRecommender):
    def __init__(self, item_name, sparse_features, dense_features, **kwargs):
        super().__init__('DeepFM', DeepFMBackbone, item_name, sparse_features, dense_features, **kwargs)
# %%
class WideDeepRecommend(NetworkRecommender):
    def __init__(self, item_name, sparse_features, dense_features, **kwargs):
        super().__init__('WideDeep', WideDeepBackbone, item_name, sparse_features, dense_features, **kwargs)
# %%
class AutoIntRecommend(NetworkRecommender):
    def __init__(self, item_name, sparse_features, dense_features, **kwargs):
        super().__init__('AutoInt', AutoIntBackbone, item_name, sparse_features, dense_features, **kwargs)
# %%
class FiBiNETRecommend(NetworkRecommender):
    def __init__(self, item_name, sparse_features, dense_features, **kwargs):
        super().__init__('FiBiNET', FiBiNETBackbone, item_name, sparse_features, dense_features, **kwargs)