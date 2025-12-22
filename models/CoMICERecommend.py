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
            'temperature': 0.2,
            'mice_method': 'MICE_RF',
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
        self.imputer = None
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
            batch_size = x.size(0)
            x, y = x.to(self.device), y.to(self.device)
            logits, proj = self.model(x)
            loss_ce = F.cross_entropy(logits, y)
            sim_matrix = torch.matmul(proj, proj.T) / self.kwargs['temperature']
            sim_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
            sim_matrix = sim_matrix - sim_max.detach()
            mask = torch.eq(y.unsqueeze(0), y.unsqueeze(1)).float()
            mask.fill_diagonal_(0)
            exp_sim = torch.exp(sim_matrix)
            diag_mask = torch.eye(batch_size, device=self.device).bool()
            exp_sim = exp_sim.masked_fill(diag_mask, 0)
            pos_sim = (exp_sim * mask).sum(dim=1)
            all_sim = exp_sim.sum(dim=1)
            eps = 1e-8
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
        train_data, self.imputer = filling(train_data, method=self.kwargs['mice_method'], seed=self.seed)
        train_data = round(train_data, self.sparse_features)
        train_data = self._mapping(train_data, fit_bool=True)
        if self.standard_bool:
            train_data = self._standardize(train_data, fit_bool=True)
        X_df = train_data[self.user_name]
        y_df = train_data[self.item_name]
        x_train_tensor = torch.tensor(X_df.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_df.values, dtype=torch.long)
        dense_count = len(self.dense_features)
        sparse_dims = [self.vocabulary_sizes[col] for col in self.sparse_features]
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
        test_data = self.imputer.transform(test_data)
        test_data = round(test_data, self.sparse_features)
        test_data = self._mapping(test_data, fit_bool=False)
        if self.standard_bool:
            test_data = self._standardize(test_data, fit_bool=False)
        X_df = test_data[self.user_name]
        x_test_tensor = torch.tensor(X_df.values, dtype=torch.float32)
        y_df = test_data[self.item_name]
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
                logits, _ = self.model(x)
                probs = torch.softmax(logits, dim=1)
                all_scores.append(probs.cpu().numpy())
        final_scores = np.concatenate(all_scores, axis=0)
        result = pd.DataFrame(final_scores, index=test_data.index, columns=self.unique_item)
        return result
# %%
class RecDataset(Dataset):
    def __init__(self, X_imputed, y):
        self.X = X_imputed
        self.y = y
        self.N = X_imputed.shape[0]
    def __len__(self):
        return self.N
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]