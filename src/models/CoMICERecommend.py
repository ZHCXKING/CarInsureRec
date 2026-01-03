# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import copy
from torch.utils.data import DataLoader, Dataset
from src.utils import filling, round
from .base import BaseRecommender
from src.network import set_seed, DCNBackbone, DCNv2Backbone, AutoIntBackbone, FiBiNETBackbone, DeepFMBackbone, WideDeepBackbone, RecDataset
# %%
def pairwise_label_aware_loss(z1, z2, y, temperature=0.1):
    batch_size = z1.shape[0]
    device = z1.device
    features = torch.cat([z1, z2], dim=0)
    labels = torch.cat([y, y], dim=0)
    features = F.normalize(features, dim=1)
    sim_matrix = torch.matmul(features, features.T) / temperature
    sim_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
    sim_matrix = sim_matrix - sim_max.detach()
    mask_label = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1)).float()
    mask_self = torch.eye(2 * batch_size, device=device)
    mask_pos = torch.zeros((2 * batch_size, 2 * batch_size), device=device)
    mask_pos[torch.arange(batch_size), torch.arange(batch_size) + batch_size] = 1
    mask_pos[torch.arange(batch_size) + batch_size, torch.arange(batch_size)] = 1
    mask_valid_neg = (1 - mask_label) + mask_pos
    mask_valid_neg = mask_valid_neg * (1 - mask_self)
    mask_valid_neg = (mask_valid_neg > 0).float()
    exp_sim = torch.exp(sim_matrix)
    pos_sim = (exp_sim * mask_pos).sum(dim=1)
    neg_sim_sum = (exp_sim * mask_valid_neg).sum(dim=1)
    log_prob = -torch.log(pos_sim / (neg_sim_sum + 1e-8) + 1e-8)
    return log_prob.mean()
def multiview_label_aware_loss(projections_list, y, temperature=0.1):
    num_views = len(projections_list)
    if num_views < 2:
        return torch.tensor(0.0).to(projections_list[0].device)
    total_loss = 0.0
    pair_count = 0
    for i in range(num_views):
        for j in range(i + 1, num_views):
            loss = pairwise_label_aware_loss(projections_list[i], projections_list[j], y, temperature)
            total_loss += loss
            pair_count += 1
    return total_loss / pair_count
class MultiViewRecDataset(Dataset):
    def __init__(self, X_list, y):
        self.X_list = X_list
        self.y = y
        self.N = self.X_list[0].shape[0]
        for x in self.X_list:
            assert x.shape[0] == self.N
    def __len__(self):
        return self.N
    def __getitem__(self, idx):
        views = [x[idx] for x in self.X_list]
        return (*views, self.y[idx])
# %%
class ContrastiveModel(nn.Module):
    def __init__(self, backbone, num_classes, proj_dim=32, is_binary=False):
        super().__init__()
        self.backbone = backbone
        output_dim = getattr(backbone, 'output_dim', None)
        self.classifier_head = nn.Linear(output_dim, num_classes)
        self.projection_head = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, proj_dim)
        )
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier_head(features)
        proj = self.projection_head(features)
        return logits, proj
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
            'backbone': 'DCN',
            'feature_dim': 32,
            'hidden_units': [256, 128],
            'dropout': 0.1,
            'cross_layers': 3,
            'low_rank': 64,
            'attention_layers': 3,
            'num_heads': 2,
            'mice_method': 'MICE_NB',
            'num_views': 3
        }
        self.kwargs.update(default_params)
        self.kwargs.update(kwargs)
        self.user_name = sparse_features + dense_features
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        set_seed(self.seed)
    def _build_model(self, sparse_dims, dense_count, num_classes):
        common_args = {
            'sparse_dims': sparse_dims,
            'dense_count': dense_count,
            'feature_dim': self.kwargs['feature_dim'],
            'hidden_units': self.kwargs['hidden_units'],
            'dropout': self.kwargs['dropout']
        }
        backbone_name = self.kwargs['backbone']
        if backbone_name == 'DCN':
            backbone = DCNBackbone(cross_layers=self.kwargs['cross_layers'], **common_args)
        elif backbone_name == 'DCNv2':
            backbone = DCNv2Backbone(cross_layers=self.kwargs['cross_layers'], low_rank=self.kwargs['low_rank'], **common_args)
        elif backbone_name == 'AutoInt':
            backbone = AutoIntBackbone(attention_layers=self.kwargs['attention_layers'], num_heads=self.kwargs['num_heads'], **common_args)
        elif backbone_name == 'FiBiNET':
            backbone = FiBiNETBackbone(**common_args)
        elif backbone_name == 'DeepFM':
            backbone = DeepFMBackbone(**common_args)
        elif backbone_name == 'WideDeep':
            backbone = WideDeepBackbone(**common_args)
        else:
            raise ValueError(f"Backbone '{backbone_name}' not supported. ")
        is_binary = (num_classes == 2)
        model = ContrastiveModel(backbone, num_classes, self.kwargs['proj_dim'], is_binary)
        return model.to(self.device)
    def train_one_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        total_ce = 0.0
        total_nce = 0.0
        for batch in dataloader:
            views_x = batch[:-1]
            y = batch[-1].to(self.device)
            views_x = [v.to(self.device) for v in views_x]
            self.optimizer.zero_grad()
            logits_list = []
            projs_list = []
            for x in views_x:
                l, p = self.model(x)
                logits_list.append(l)
                projs_list.append(p)
            ce_loss = 0.0
            for l in logits_list:
                ce_loss += F.cross_entropy(l, y)
            ce_loss = ce_loss / len(views_x)
            nce_loss = multiview_label_aware_loss(
                projs_list,
                y,
                temperature=self.kwargs['temperature']
            )
            loss = ce_loss + self.kwargs['lambda_nce'] * nce_loss
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            total_ce += ce_loss.item()
            total_nce += nce_loss.item()
        n = len(dataloader)
        return total_loss / n, total_ce / n, total_nce / n
    def fit(self, train_data: pd.DataFrame, valid_data: pd.DataFrame = None, patience: int = 10):
        self.out_dim = train_data[self.item_name].nunique()
        self.unique_item = list(range(self.out_dim))
        num_views = self.kwargs['num_views']
        print(f"Generating {num_views} MICE views...")
        views_dataframes = []
        for i in range(num_views):
            current_seed = self.seed + i
            df_filled, imputer = filling(train_data.copy(), method=self.kwargs['mice_method'], seed=current_seed)
            df_filled = round(df_filled, self.sparse_features)
            views_dataframes.append(df_filled)
            if i == 0:
                self.imputer = imputer
        print("Preprocessing views...")
        views_dataframes[0] = self._mapping(views_dataframes[0], fit_bool=True)
        if self.standard_bool:
            views_dataframes[0] = self._standardize(views_dataframes[0], fit_bool=True)
        for i in range(1, num_views):
            views_dataframes[i] = self._mapping(views_dataframes[i], fit_bool=False)
            if self.standard_bool:
                views_dataframes[i] = self._standardize(views_dataframes[i], fit_bool=False)
        X_tensors = [torch.tensor(df[self.user_name].values, dtype=torch.float32) for df in views_dataframes]
        y_tensor = torch.tensor(views_dataframes[0][self.item_name].values, dtype=torch.long)
        sparse_dims = [self.vocabulary_sizes[col] for col in self.sparse_features]
        dense_count = len(self.dense_features)
        train_loader = DataLoader(
            MultiViewRecDataset(X_tensors, y_tensor),
            batch_size=self.kwargs['batch_size'],
            shuffle=True
        )
        valid_loader = None
        if valid_data is not None:
            val_filled = self.imputer.transform(valid_data.copy())
            val_filled = round(val_filled, self.sparse_features)
            val_filled = self._mapping(val_filled, fit_bool=False)
            if self.standard_bool:
                val_filled = self._standardize(val_filled, fit_bool=False)
            X_val = torch.tensor(val_filled[self.user_name].values, dtype=torch.float32)
            y_val = torch.tensor(val_filled[self.item_name].values, dtype=torch.long)
            valid_loader = DataLoader(RecDataset(X_val, y_val), batch_size=self.kwargs['batch_size'], shuffle=False)
        self.model = self._build_model(sparse_dims, dense_count, self.out_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.kwargs['lr'])
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_weights = None
        print(f"Start Contrastive Training with {num_views} views (Label-Aware)...")
        for epoch in range(self.kwargs['epochs']):
            avg_loss, avg_ce, avg_nce = self.train_one_epoch(train_loader)
            log_msg = f"Epoch {epoch + 1}: Total {avg_loss:.4f} | CE {avg_ce:.4f} | NCE {avg_nce:.4f}"
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
                log_msg += f" | Val CE {avg_val_loss:.4f}"
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    best_model_weights = copy.deepcopy(self.model.state_dict())
                else:
                    patience_counter += 1
            print(log_msg)
            if valid_loader is not None and patience_counter >= patience:
                print("Early stopping triggered")
                self.model.load_state_dict(best_model_weights)
                break
        if valid_loader is not None and best_model_weights is not None and patience_counter < patience:
            self.model.load_state_dict(best_model_weights)
        self.is_trained = True
    def get_proba(self, test_data: pd.DataFrame):
        if not self.is_trained:
            raise ValueError('Model not trained')
        test_filled = self.imputer.transform(test_data.copy())
        test_filled = round(test_filled, self.sparse_features)
        test_filled = self._mapping(test_filled, fit_bool=False)
        if self.standard_bool:
            test_filled = self._standardize(test_filled, fit_bool=False)
        X_tensor = torch.tensor(test_filled[self.user_name].values, dtype=torch.float32)
        y_tensor = torch.zeros(len(test_filled), dtype=torch.long)
        loader = DataLoader(RecDataset(X_tensor, y_tensor), batch_size=self.kwargs['batch_size'], shuffle=False)
        self.model.eval()
        all_probs = []
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(self.device)
                logits, _ = self.model(x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                all_probs.append(probs)
        final_probs = np.concatenate(all_probs, axis=0)
        return pd.DataFrame(final_probs, index=test_data.index, columns=self.unique_item)