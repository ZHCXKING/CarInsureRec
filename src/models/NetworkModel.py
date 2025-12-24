# %%
import pandas as pd
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
            'lr': 1e-4,
            'batch_size': 512,
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
        if self.backbone_class == [DCNv2Backbone, HybridBackbone]:
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
    # %%
    def fit(self, train_data: pd.DataFrame):
        self.out_dim = train_data[self.item_name].nunique()
        self.unique_item = list(range(self.out_dim))
        train_data = self._mapping(train_data, fit_bool=True)
        if self.standard_bool:
            train_data = self._standardize(train_data, fit_bool=True)
        X_tensor = torch.tensor(train_data[self.user_name].values, dtype=torch.float32)
        y_tensor = torch.tensor(train_data[self.item_name].values, dtype=torch.long)
        sparse_dims = [self.vocabulary_sizes[col] for col in self.sparse_features]
        dense_count = len(self.dense_features)
        train_loader = DataLoader(RecDataset(X_tensor, y_tensor), batch_size=self.kwargs['batch_size'], shuffle=True)
        self.model = self._build_model(sparse_dims, dense_count, self.out_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.kwargs['lr'])
        criterion = nn.CrossEntropyLoss()
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
            print(f"Epoch {epoch + 1}: Loss {total_loss / len(train_loader):.4f}")
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