#%%
import torch
import random
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from .base import BaseRecommender
#%%
class RecModel(nn.Module):
    def __init__(self, num_classes, sparse_dims, dense_count, embed_dim:int=128):
        super().__init__()
        self.embs = nn.ModuleList([
            nn.Embedding(num_embeddings=dim, embedding_dim=16)
            for dim in sparse_dims
        ])
        self.input_dim = len(sparse_dims) * 16 + dense_count
        self.backbone = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            #nn.Dropout(0.3),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, embed_dim)
        )
        self.classifier_head = nn.Linear(embed_dim, num_classes)
        self.projection_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
    #%%
    def forward(self, x):
        sparse_x = x[:, :len(self.embs)].long()  # [B, num_sparse]
        dense_x = x[:, len(self.embs):]  # [B, dense_count]
        emb_outs = []
        for i, emb_layer in enumerate(self.embs):
            emb_outs.append(emb_layer(sparse_x[:, i]))
        x_concat = torch.cat(emb_outs + [dense_x], dim=1)  # [B, input_dim]
        z = self.backbone(x_concat)
        logits = self.classifier_head(z)
        proj = self.projection_head(z)
        proj = F.normalize(proj, dim=1)
        return logits, proj
#%%
class RecDataset(Dataset):
    def __init__(self, X_imputed, y, mode='train'):
        """
        X_imputed: Tensor [N, m, d] - N个样本, m个插补, d个特征
        y: Tensor [N] - 标签
        mode: 'train' 或 'eval'
        """
        self.X = X_imputed
        self.y = y
        self.mode = mode
        self.N, self.m, self.d = X_imputed.shape
    def __len__(self):
        return self.N
    def __getitem__(self, idx):
        if self.mode == 'train':
            idx1, idx2 = random.sample(range(self.m), 2)
            x_view1 = self.X[idx, idx1, :] #[d]
            x_view2 = self.X[idx, idx2, :] #[d]
            label = self.y[idx]
            return x_view1, x_view2, label
        elif self.mode == 'eval':
            x_all = self.X[idx, :, :] #[m,d]
            label = self.y[idx]
            return x_all, label
        else:
            raise ValueError('mode is not supported')
# %%
class CoMICERecommend(BaseRecommender):
    def __init__(self, user_name: list, item_name: str, date_name: str | None = None,
                 sparse_features: list | None = None, dense_features: list | None = None, standard_bool: bool = False,
                 seed: int = 42, **kwargs):
        if (user_name is None) or (item_name is None):
            raise ValueError('user_name and item_name are required')
        super().__init__('CoMICE', user_name, item_name, date_name, sparse_features, dense_features, standard_bool, seed)
        default_params = {
            'm': 3,
            'batch_size': 64,
            'embed_dim': 128,
            'epochs': 500,
        }
        self.kwargs.update(default_params)
        self.kwargs.update(kwargs)