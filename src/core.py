#%%
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from src.utils import load
from src.utils import mice_samples
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
            nn.Dropout(0.1),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, embed_dim)
        )
        self.classifier_head = nn.Linear(embed_dim, num_classes)
        self.projection_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
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
def train_one_epoch(model, dataloader, optimizer, device, lambda_nce=1.0, lambda_kl=5.0, lambda_topk=0.5, temperature=0.1, topk=5, margin=0.2):
    model.train()
    total_loss = 0.0
    topk_loss_fn = TopKMarginLoss(k=topk, margin=margin).to(device)
    for x_view1, x_view2, y in dataloader:
        x_view1, x_view2, y = x_view1.to(device), x_view2.to(device), y.to(device)
        optimizer.zero_grad()
        logits1, proj1 = model(x_view1)
        logits2, proj2 = model(x_view2)
        # A. Cross Entropy (Label Smoothing)
        loss_ce = 0.5 * (
            F.cross_entropy(logits1, y, label_smoothing=0.1) +
            F.cross_entropy(logits2, y, label_smoothing=0.1)
        )
        # B. Top-k Margin Loss
        loss_topk = 0.5 * (
            topk_loss_fn(logits1, y) +
            topk_loss_fn(logits2, y)
        )
        # C. InfoNCE Loss
        batch_size = x_view1.size(0)
        features = torch.cat([proj1, proj2], dim=0)
        sim_matrix = torch.matmul(features, features.T) / temperature
        labels_contrastive = torch.cat([
            torch.arange(batch_size) + batch_size,
            torch.arange(batch_size)
        ]).to(device)
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(device)
        sim_matrix.masked_fill_(mask, -9e15)
        loss_nce = F.cross_entropy(sim_matrix, labels_contrastive)
        # D. KL Consistency Loss
        p1 = F.log_softmax(logits1, dim=1)
        p2 = F.log_softmax(logits2, dim=1)
        loss_kl = 0.5 * (
            F.kl_div(p1, p2.exp(), reduction='batchmean') +
            F.kl_div(p2, p1.exp(), reduction='batchmean')
        )
        # Total Loss
        loss = (loss_ce + lambda_topk * loss_topk + lambda_nce * loss_nce + lambda_kl * loss_kl)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    print(
        f"Epoch Loss: {avg_loss:.4f} | "
        f"CE: {loss_ce.item():.4f} | "
        f"TopK: {loss_topk.item():.4f} | "
        f"NCE: {loss_nce.item():.4f} | "
        f"KL: {loss_kl.item():.4f}"
    )
    return avg_loss
#%%
def validate_and_get_probs(model, dataloader, device, alpha=1.0, eps=1e-8):
    """
    返回 risk-aware score，用于排序或评估
    shape: [Total_Samples, Num_Classes]
    """
    model.eval()
    all_scores = []
    with torch.no_grad():
        for x_all, y in dataloader:
            # x_all: [Batch, m, d]
            x_all = x_all.to(device)
            batch_size, m, d = x_all.shape
            # 1. 展平插补维度
            x_flat = x_all.view(-1, d)  # [Batch*m, d]
            # 2. 前向推理
            logits, _ = model(x_flat)   # [Batch*m, C]
            # 3. softmax 概率
            probs = F.softmax(logits, dim=1)  # [Batch*m, C]
            # 4. reshape 回插补维度
            probs = probs.view(batch_size, m, -1)  # [B, m, C]
            # 5. 计算期望 & 方差
            mean_probs = probs.mean(dim=1)                  # μ: [B, C]
            var_probs = probs.var(dim=1, unbiased=False)    # σ²: [B, C]
            # 6. risk-aware score
            risk_scores = mean_probs / (1.0 + alpha * var_probs + eps)
            all_scores.append(risk_scores.cpu().numpy())
    final_scores = np.concatenate(all_scores, axis=0)
    return final_scores
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
#%%
def process_mice_list(df_list, user_name, item_name):
    all_versions = [torch.tensor(df[user_name].values, dtype=torch.float) for df in df_list]
    # 堆叠成 [N, m, d]
    X_imputed = torch.stack(all_versions, dim=1)
    # 标签在所有版本中应该是相同的，取第一个即可
    # 假设标签已经是数值型，如果不是请先进行 LabelEncoding
    y = torch.tensor(df_list[0][item_name].values, dtype=torch.long)
    return X_imputed, y
# %%
class transform_meth():
    def __init__(self, sparse_feature, dense_feature):
        self.sparse_feature = sparse_feature
        self.dense_feature = dense_feature
        self.mapping = {}
        self.vocabulary_sizes = {}
    def _Standardize(self, data: pd.DataFrame, fit_bool: bool):
        if self.dense_feature is None:
            raise ValueError('dense_feature is None')
        if fit_bool:
            self.scaler = StandardScaler()
            data[self.dense_feature] = self.scaler.fit_transform(data[self.dense_feature])
        else:
            data[self.dense_feature] = self.scaler.transform(data[self.dense_feature])
        return data
    # %%
    def _mapping(self, data: pd.DataFrame, fit_bool: bool):
        if fit_bool:
            for col in self.sparse_feature:
                unique = sorted(data[col].unique())
                mapping = {v: i + 1 for i, v in enumerate(unique)}
                self.mapping[col] = mapping
                self.vocabulary_sizes[col] = len(mapping) + 1
                data[col] = data[col].map(lambda x: self.mapping[col].get(x, 0))
        else:
            for col in self.sparse_feature:
                data[col] = data[col].map(lambda x: self.mapping[col].get(x, 0))
        return data
#%%
class TopKMarginLoss(nn.Module):
    def __init__(self, k=5, margin=0.2):
        super().__init__()
        self.k = k
        self.margin = margin
    def forward(self, logits, targets):
        """
        logits: [B, C]
        targets: [B]
        """
        B = logits.size(0)
        # top-k logits
        topk_vals, topk_idx = torch.topk(logits, self.k, dim=1)
        # true class logits
        true_logits = logits[torch.arange(B), targets]
        # mask true label in top-k
        mask = topk_idx != targets.unsqueeze(1)
        # hardest negative in top-k
        neg_logits = topk_vals.masked_fill(~mask, -1e9).max(dim=1).values
        loss = F.relu(self.margin - true_logits + neg_logits)
        return loss.mean()
#%%
m = 5
batch_size=64
embed_dim=128
epochs = 500
train, test = load('original', amount=4000, split_num=250) #original, dropna
#user_name = ['Age', 'DrivingExp', 'Occupation', 'NCD', 'Make', 'Car.year', 'Car.price']
item_name = 'InsCov'
date_name = 'Date'
sparse_features = ['Occupation', 'NCD', 'Make']
dense_features = ['Age', 'Car.year', 'Car.price', 'DrivingExp']
user_name = sparse_features + dense_features #确保类别变量在连续变量之前
#
transform_method = transform_meth(sparse_features, dense_features)
train = transform_method._mapping(train, fit_bool=True)
test = transform_method._mapping(test, fit_bool=False)
train = transform_method._Standardize(train, fit_bool=True)
test = transform_method._Standardize(test, fit_bool=False)
#
train_data_sets, test_data_sets, _ = mice_samples(train, test, method='iterative_NB', m=m, seed=42)
#
dense_count = len(dense_features)
sparse_dims = [transform_method.vocabulary_sizes[col] for col in sparse_features]
#
x_train_tensor, y_train_tensor = process_mice_list(train_data_sets, user_name, item_name)
x_test_tensor, y_test_tensor = process_mice_list(test_data_sets, user_name, item_name)
train_loader = DataLoader(
    RecDataset(x_train_tensor, y_train_tensor, mode='train'),
    batch_size=batch_size, shuffle=True
)
test_loader = DataLoader(
    RecDataset(x_test_tensor, y_test_tensor, mode='eval'),
    batch_size=batch_size, shuffle=False
)
# 自动获取类别数
num_classes = train[item_name].nunique()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RecModel(num_classes, sparse_dims, dense_count, embed_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
for epoch in range(epochs):
    avg_loss = train_one_epoch(model, train_loader, optimizer, device)
    probs = validate_and_get_probs(model, test_loader, device)
    preds = np.argmax(probs, axis=1)
    acc = (preds == y_test_tensor.numpy()).mean()
    print(f"Test Accuracy: {acc:.4f}")