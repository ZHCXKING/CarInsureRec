import torch
import random
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
# %%
def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# %%
class CoMICEHead(nn.Module):
    def __init__(self, input_dim, num_classes, proj_dim=128, hidden_dim=256, dropout_rate=0.1):
        super().__init__()
        self.projection_head = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim, proj_dim)
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(proj_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim // 2, input_dim),
            nn.Sigmoid()
        )
        self.classifier_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    def forward(self, feature_vector):
        proj_feat = self.projection_head(feature_vector)
        proj_norm = F.normalize(proj_feat, dim=1)
        gate = self.gate_mlp(proj_feat)
        feat_calibrated = feature_vector * (1 + gate)
        logits = self.classifier_head(feat_calibrated)
        return logits, proj_norm
# %%
class CoMICEModel(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head
    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)
# %%
class CrossNet(nn.Module):
    def __init__(self, input_dim, num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        self.W = nn.ParameterList([nn.Parameter(torch.empty(input_dim, input_dim)) for _ in range(num_layers)])
        self.b = nn.ParameterList([nn.Parameter(torch.empty(input_dim)) for _ in range(num_layers)])
        for w in self.W: nn.init.xavier_uniform_(w)
        for b in self.b: nn.init.zeros_(b)
    def forward(self, x_0):
        x_l = x_0
        for i in range(self.num_layers):
            x_l = x_0 * (torch.mm(x_l, self.W[i]) + self.b[i]) + x_l
        return x_l
# %%
class DCNv2Backbone(nn.Module):
    def __init__(self, sparse_dims, dense_count, feature_dim=32, cross_layers=3, hidden_units=[256, 128]):
        super().__init__()
        self.num_sparse = len(sparse_dims)
        self.num_dense = dense_count
        self.sparse_embs = nn.ModuleList([nn.Embedding(d, feature_dim) for d in sparse_dims])
        if dense_count > 0:
            self.dense_proj = nn.ModuleList([nn.Linear(1, feature_dim) for _ in range(dense_count)])
        self.total_input_dim = (self.num_sparse + self.num_dense) * feature_dim
        self.cross_net = CrossNet(self.total_input_dim, num_layers=cross_layers)
        layers = []
        last_dim = self.total_input_dim
        for hidden in hidden_units:
            layers.extend([nn.Linear(last_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(0.1)])
            last_dim = hidden
        self.deep_net = nn.Sequential(*layers)
        self.output_dim = self.total_input_dim + hidden_units[-1]
    # %%
    def forward(self, x):
        sparse_x = x[:, :self.num_sparse].long()
        dense_x = x[:, self.num_sparse:]
        embs = [emb(sparse_x[:, i]) for i, emb in enumerate(self.sparse_embs)]
        if self.num_dense > 0:
            embs.extend([self.dense_proj[i](dense_x[:, i].unsqueeze(1)) for i in range(self.num_dense)])
        x_0 = torch.cat(embs, dim=1)
        return torch.cat([self.cross_net(x_0), self.deep_net(x_0)], dim=1)
# %%
class DeepFMBackbone(nn.Module):
    def __init__(self, sparse_dims, dense_count, feature_dim=16, hidden_units=[256, 128]):
        super().__init__()
        self.num_sparse = len(sparse_dims)
        self.num_dense = dense_count
        self.embeddings = nn.ModuleList([nn.Embedding(d, feature_dim) for d in sparse_dims])
        self.linear_sparse = nn.ModuleList([nn.Embedding(d, 1) for d in sparse_dims])
        if dense_count > 0: self.linear_dense = nn.Linear(dense_count, 1)
        input_dim = self.num_sparse * feature_dim + dense_count
        layers = []
        last_dim = input_dim
        for hidden in hidden_units:
            layers.extend([nn.Linear(last_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(0.1)])
            last_dim = hidden
        self.dnn = nn.Sequential(*layers)
        self.output_dim = hidden_units[-1]
        self.adapter = nn.Linear(1, self.output_dim)
    # %%
    def forward(self, x):
        sparse_x = x[:, :self.num_sparse].long()
        dense_x = x[:, self.num_sparse:]
        lin_out = sum([emb(sparse_x[:, i]) for i, emb in enumerate(self.linear_sparse)])
        if self.num_dense > 0: lin_out += self.linear_dense(dense_x)
        emb_stack = torch.stack([emb(sparse_x[:, i]) for i, emb in enumerate(self.embeddings)], dim=1)
        sum_sq = torch.pow(torch.sum(emb_stack, dim=1), 2)
        sq_sum = torch.sum(torch.pow(emb_stack, 2), dim=1)
        fm_out = 0.5 * torch.sum(sum_sq - sq_sum, dim=1, keepdim=True)
        dnn_in = emb_stack.view(x.size(0), -1)
        if self.num_dense > 0: dnn_in = torch.cat([dnn_in, dense_x], dim=1)
        dnn_out = self.dnn(dnn_in)
        return dnn_out + self.adapter(lin_out + fm_out)
# %%
class WideDeepBackbone(nn.Module):
    def __init__(self, sparse_dims, dense_count, feature_dim=16, hidden_units=[256, 128]):
        super().__init__()
        self.num_sparse = len(sparse_dims)
        self.num_dense = dense_count
        self.wide_sparse = nn.ModuleList([nn.Embedding(d, 1) for d in sparse_dims])
        if dense_count > 0: self.wide_dense = nn.Linear(dense_count, 1)
        self.deep_embs = nn.ModuleList([nn.Embedding(d, feature_dim) for d in sparse_dims])
        input_dim = self.num_sparse * feature_dim + dense_count
        layers = []
        last_dim = input_dim
        for hidden in hidden_units:
            layers.extend([nn.Linear(last_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(0.1)])
            last_dim = hidden
        self.dnn = nn.Sequential(*layers)
        self.output_dim = hidden_units[-1]
        self.adapter = nn.Linear(1, self.output_dim)
    # %%
    def forward(self, x):
        sparse_x = x[:, :self.num_sparse].long()
        dense_x = x[:, self.num_sparse:]
        wide_out = sum([emb(sparse_x[:, i]) for i, emb in enumerate(self.wide_sparse)])
        if self.num_dense > 0: wide_out += self.wide_dense(dense_x)
        deep_in = torch.cat([emb(sparse_x[:, i]) for i, emb in enumerate(self.deep_embs)], dim=1)
        if self.num_dense > 0: deep_in = torch.cat([deep_in, dense_x], dim=1)
        deep_out = self.dnn(deep_in)
        return deep_out + self.adapter(wide_out)
