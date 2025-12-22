# %%
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from .base import BaseRecommender
from src.utils import filling, round
from src.network import HybridBackbone, CoMICEHead, CoMICEModel, set_seed, RecDataset
# %%
class CoMICERecommend(BaseRecommender):
    def __init__(self, item_name: str, sparse_features: list, dense_features: list,
                 standard_bool: bool = True, seed: int = 42, k: int = 3, **kwargs):
        super().__init__('CoMICE', item_name, sparse_features, dense_features, standard_bool, seed, k)
        default_params = {
            'lr': 1e-4,
            'batch_size': 512,
            'feature_dim': 32,
            'proj_dim': 32,
            'epochs': 200,
            'lambda_nce': 1.0,
            'temperature': 0.1,
            'mice_method': 'MICE_RF',
            'cross_layers': 3,
            'hidden_units': [256, 128],
            'use_attention': True,
            'dropout': 0.1
        }
        self.kwargs.update(default_params)
        self.kwargs.update(kwargs)
        self.user_name = sparse_features + dense_features
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        set_seed(self.seed)
    # %%
    def _build_model(self, sparse_dims, dense_count, num_classes):
        common_args = {
            'sparse_dims': sparse_dims, 'dense_count': dense_count,
            'feature_dim': self.kwargs['feature_dim'],
            'hidden_units': self.kwargs['hidden_units'],
            'dropout': self.kwargs['dropout']
        }
        backbone = HybridBackbone(cross_layers=self.kwargs['cross_layers'],
                                  use_attention=self.kwargs['use_attention'],
                                  **common_args)
        head = CoMICEHead(backbone.output_dim, num_classes, proj_dim=self.kwargs['proj_dim'])
        return CoMICEModel(backbone, head).to(self.device)
    # %%
    def train_one_epoch(self, dataloader):
        self.model.train()
        total_loss, ce_acc, nce_acc = 0.0, 0.0, 0.0
        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            logits, proj = self.model(x)
            loss_ce = F.cross_entropy(logits, y)
            batch_size = x.size(0)
            sim_matrix = torch.matmul(proj, proj.T) / self.kwargs['temperature']
            sim_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
            sim_matrix = sim_matrix - sim_max.detach()
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
            loss = loss_ce + self.kwargs['lambda_nce'] * loss_nce
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            ce_acc += loss_ce.item()
            nce_acc += loss_nce.item()
        print(f"Loss: {total_loss / len(dataloader):.4f} (CE: {ce_acc / len(dataloader):.4f}, NCE: {nce_acc / len(dataloader):.4f})")
    # %%
    def fit(self, train_data: pd.DataFrame):
        self.out_dim = train_data[self.item_name].nunique()
        self.unique_item = list(range(self.out_dim))
        train_data, self.imputer = filling(train_data, method=self.kwargs['mice_method'], seed=self.seed)
        train_data = round(train_data, self.sparse_features)
        train_data = self._mapping(train_data, fit_bool=True)
        if self.standard_bool:
            train_data = self._standardize(train_data, fit_bool=True)
        X_tensor = torch.tensor(train_data[self.user_name].values, dtype=torch.float32)
        y_tensor = torch.tensor(train_data[self.item_name].values, dtype=torch.long)
        sparse_dims = [self.vocabulary_sizes[col] for col in self.sparse_features]
        dense_count = len(self.dense_features)
        loader = DataLoader(RecDataset(X_tensor, y_tensor), batch_size=self.kwargs['batch_size'], shuffle=True)
        self.model = self._build_model(sparse_dims, dense_count, self.out_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.kwargs['lr'])
        for epoch in range(self.kwargs['epochs']):
            self.train_one_epoch(loader)
        self.is_trained = True
    # %%
    def get_proba(self, test_data: pd.DataFrame):
        if not self.is_trained:
            raise ValueError('Model not trained')
        test_data = self.imputer.transform(test_data)
        test_data = round(test_data, self.sparse_features)
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