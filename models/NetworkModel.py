# %%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from .base import BaseRecommender
from src.utils import filling, round
from src.network import DCNv2Backbone, DeepFMBackbone, WideDeepBackbone, set_seed
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
class StandardDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.N = X.shape[0]
    def __len__(self):
        return self.N
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
# %%
class NetworkRecommender(BaseRecommender):
    def __init__(self, model_name: str, backbone_class, user_name: list, item_name: str,
                 date_name: str | None = None, sparse_features: list | None = None,
                 dense_features: list | None = None, standard_bool: bool = False,
                 seed: int = 42, k: int = 3, **kwargs):
        if (user_name is None) or (item_name is None):
            raise ValueError('user_name and item_name are required')
        super().__init__(model_name, user_name, item_name, date_name, sparse_features, dense_features, standard_bool, seed, k)
        self.backbone_class = backbone_class
        default_params = {
            'lr': 1e-4,
            'batch_size': 128,
            'feature_dim': 64,
            'epochs': 200,
            'hidden_units': [256, 128],
            'cross_layers': 3,
            'mice_method': 'MICE_Ga'
        }
        self.kwargs.update(default_params)
        self.kwargs.update(kwargs)
        self.user_name = sparse_features + dense_features
        self.out_dim = None
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.optimizer = None
        self.imputer = None
        set_seed(self.seed)
    # %%
    def _build_model(self, sparse_dims, dense_count, num_classes):
        if self.backbone_class == DCNv2Backbone:
            backbone = DCNv2Backbone(
                sparse_dims, dense_count,
                feature_dim=self.kwargs['feature_dim'],
                cross_layers=self.kwargs['cross_layers'],
                hidden_units=self.kwargs['hidden_units']
            )
        else:
            backbone = self.backbone_class(
                sparse_dims, dense_count,
                feature_dim=self.kwargs['feature_dim'],
                hidden_units=self.kwargs['hidden_units']
            )
        return StandardModel(backbone, num_classes).to(self.device)
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
        # 2. 插补 (filling 包含 round)
        data_imputed, self.imputer = filling(train_data, method=self.kwargs['mice_method'], seed=seed_val)
        dense_count = len(self.dense_features)
        sparse_dims = [self.vocabulary_sizes[col] for col in self.sparse_features]
        # 3. 手动转换为 Tensor (替代 process_mice_list)
        # 提取特征
        X_df = data_imputed[self.user_name]
        x_train = torch.tensor(X_df.values, dtype=torch.float32)
        # 提取标签
        y_df = data_imputed[self.item_name]
        y_train = torch.tensor(y_df.values, dtype=torch.long)
        # 4. 创建 DataLoader
        train_loader = DataLoader(
            StandardDataset(x_train, y_train),
            batch_size=self.kwargs['batch_size'],
            shuffle=True
        )
        self.model = self._build_model(sparse_dims, dense_count, self.out_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.kwargs['lr'])
        criterion = nn.CrossEntropyLoss()
        # 5. 训练
        for epoch in range(self.kwargs['epochs']):
            self.model.train()
            total_loss = 0.0
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(x)
                loss = criterion(logits, y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}: Loss {total_loss / len(train_loader):.4f}")
        self.is_trained = True
    # %%
    def get_proba(self, test_data: pd.DataFrame):
        if not self.is_trained:
            raise ValueError('model is not trained')
        test_data = self._mapping(test_data, fit_bool=False)
        if self.standard_bool:
            test_data = self._standardize(test_data, fit_bool=False)
        # 1. 插补
        data_imputed = self.imputer.transform(test_data)
        # 2. 取整 (显式调用)
        data_imputed = round(data_imputed)
        # 3. 手动转换为 Tensor (替代 process_mice_list)
        X_df = data_imputed[self.user_name]
        x_test = torch.tensor(X_df.values, dtype=torch.float32)
        # 处理标签 (如果有的话，没有则 dummy)
        if self.item_name in data_imputed.columns:
            y_test = torch.tensor(data_imputed[self.item_name].values, dtype=torch.long)
        else:
            y_test = torch.zeros(len(x_test), dtype=torch.long)
        test_loader = DataLoader(
            StandardDataset(x_test, y_test),
            batch_size=self.kwargs['batch_size'],
            shuffle=False
        )
        self.model.eval()
        all_probs = []
        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(self.device)
                logits = self.model(x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                all_probs.append(probs)
        final_probs = np.concatenate(all_probs, axis=0)
        result = pd.DataFrame(final_probs, index=test_data.index, columns=self.unique_item)
        return result
# %%
class DCNv2Recommend(NetworkRecommender):
    def __init__(self, user_name: list, item_name: str, date_name: str | None = None,
                 sparse_features: list | None = None, dense_features: list | None = None,
                 standard_bool: bool = False, seed: int = 42, k: int = 3, **kwargs):
        super().__init__('DCNv2', DCNv2Backbone, user_name, item_name, date_name,
                         sparse_features, dense_features, standard_bool, seed, k, **kwargs)
# %%
class DeepFMRecommend(NetworkRecommender):
    def __init__(self, user_name: list, item_name: str, date_name: str | None = None,
                 sparse_features: list | None = None, dense_features: list | None = None,
                 standard_bool: bool = False, seed: int = 42, k: int = 3, **kwargs):
        super().__init__('DeepFM', DeepFMBackbone, user_name, item_name, date_name,
                         sparse_features, dense_features, standard_bool, seed, k, **kwargs)
# %%
class WideDeepRecommend(NetworkRecommender):
    def __init__(self, user_name: list, item_name: str, date_name: str | None = None,
                 sparse_features: list | None = None, dense_features: list | None = None,
                 standard_bool: bool = False, seed: int = 42, k: int = 3, **kwargs):
        super().__init__('WideDeep', WideDeepBackbone, user_name, item_name, date_name,
                         sparse_features, dense_features, standard_bool, seed, k, **kwargs)
