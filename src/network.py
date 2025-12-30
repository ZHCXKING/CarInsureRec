import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import random
import os
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
class FactorizationMachine(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, inputs):
        sum_of_square = torch.sum(inputs, dim=1) ** 2
        square_of_sum = torch.sum(inputs ** 2, dim=1)
        ix = sum_of_square - square_of_sum
        return 0.5 * torch.sum(ix, dim=1, keepdim=True)
class CrossNetwork(nn.Module):
    def __init__(self, input_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.W = nn.ParameterList([nn.Parameter(torch.randn(input_dim)) for _ in range(num_layers)])
        self.b = nn.ParameterList([nn.Parameter(torch.zeros(input_dim)) for _ in range(num_layers)])
        for w in self.W: nn.init.xavier_uniform_(w.unsqueeze(0))
    def forward(self, x_0):
        x_l = x_0
        for i in range(self.num_layers):
            dot_prod = torch.sum(x_l * self.W[i], dim=1, keepdim=True)
            x_l = x_0 * dot_prod + self.b[i] + x_l
        return x_l
class DCNv2Layer(nn.Module):
    def __init__(self, input_dim, low_rank=64):
        super().__init__()
        self.input_dim = input_dim
        self.v_linear = nn.Linear(input_dim, low_rank, bias=False)
        self.u_linear = nn.Linear(low_rank, input_dim, bias=True)
        nn.init.xavier_uniform_(self.v_linear.weight)
        nn.init.xavier_uniform_(self.u_linear.weight)
    def forward(self, x_l, x_0):
        v_out = self.v_linear(x_l)
        u_out = self.u_linear(v_out)
        return x_0 * u_out + x_l
class AutoIntLayer(nn.Module):
    def __init__(self, embed_dim, num_heads=2, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
        self.W_res = nn.Linear(embed_dim, embed_dim)
        self.relu = nn.ReLU()
    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        res_out = self.W_res(x)
        out = self.relu(attn_out + res_out)
        return out
class SENETLayer(nn.Module):
    def __init__(self, num_fields, reduction_ratio=3):
        super().__init__()
        reduced_dim = max(1, num_fields // reduction_ratio)
        self.excitation = nn.Sequential(
            nn.Linear(num_fields, reduced_dim, bias=False),
            nn.ReLU(),
            nn.Linear(reduced_dim, num_fields, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        z = torch.mean(x, dim=2)
        a = self.excitation(z)
        return x * a.unsqueeze(-1)
class BilinearInteraction(nn.Module):
    def __init__(self, embed_dim, num_fields, type='Field-All'):
        super().__init__()
        self.type = type
        if type == 'Field-All':
            self.W = nn.Parameter(torch.randn(embed_dim, embed_dim))
            nn.init.xavier_uniform_(self.W)
        elif type == 'Field-Each':
            self.W = nn.Parameter(torch.randn(num_fields, embed_dim, embed_dim))
            nn.init.xavier_uniform_(self.W)
    def forward(self, inputs):
        B, F, E = inputs.shape
        if self.type == 'Field-All':
            vid = torch.matmul(inputs, self.W)
        elif self.type == 'Field-Each':
            vid = torch.matmul(inputs.unsqueeze(2), self.W).squeeze(2)
        else:
            raise NotImplementedError
        p = vid.unsqueeze(2) * inputs.unsqueeze(1)
        triu_idx = torch.triu_indices(F, F, offset=1).to(inputs.device)
        inter = p[:, triu_idx[0], triu_idx[1], :]
        return inter.reshape(B, -1)
class DNN(nn.Module):
    def __init__(self, input_dim, hidden_units, dropout=0.0):
        super().__init__()
        layers = []
        in_dim = input_dim
        for hidden in hidden_units:
            layers.extend([
                nn.Linear(in_dim, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden
        self.net = nn.Sequential(*layers)
        self.output_dim = in_dim
    def forward(self, x):
        return self.net(x)
# %%
class WideDeepBackbone(nn.Module):
    def __init__(self, sparse_dims, dense_count, feature_dim=16, hidden_units=[256, 128], dropout=0.1):
        super().__init__()
        self.num_sparse = len(sparse_dims)
        self.num_dense = dense_count
        self.wide_sparse = nn.ModuleList([nn.Embedding(d, 1) for d in sparse_dims])
        if dense_count > 0:
            self.wide_dense = nn.Linear(dense_count, 1)
        self.deep_embs = nn.ModuleList([nn.Embedding(d, feature_dim) for d in sparse_dims])
        deep_input_dim = self.num_sparse * feature_dim + dense_count
        self.dnn = DNN(deep_input_dim, hidden_units, dropout)
        dnn_out_dim = hidden_units[-1]
        self.output_dim = 1
        self.final_linear = nn.Linear(dnn_out_dim, 1)
    def forward(self, x):
        sparse_x = x[:, :self.num_sparse].long()
        dense_x = x[:, self.num_sparse:]
        wide_out = sum([emb(sparse_x[:, i]) for i, emb in enumerate(self.wide_sparse)])
        if self.num_dense > 0:
            wide_out += self.wide_dense(dense_x)
        deep_emb = torch.cat([emb(sparse_x[:, i]) for i, emb in enumerate(self.deep_embs)], dim=1)
        deep_in = torch.cat([deep_emb, dense_x], dim=1) if self.num_dense > 0 else deep_emb
        deep_out = self.dnn(deep_in)
        deep_logit = self.final_linear(deep_out)
        return wide_out + deep_logit
class DeepFMBackbone(nn.Module):
    def __init__(self, sparse_dims, dense_count, feature_dim=16, hidden_units=[256, 128], dropout=0.1):
        super().__init__()
        self.num_sparse = len(sparse_dims)
        self.num_dense = dense_count
        self.embeddings = nn.ModuleList([nn.Embedding(d, feature_dim) for d in sparse_dims])
        if dense_count > 0:
            self.dense_embs = nn.ModuleList([nn.Linear(1, feature_dim) for _ in range(dense_count)])
        self.fm_1st_sparse = nn.ModuleList([nn.Embedding(d, 1) for d in sparse_dims])
        if dense_count > 0:
            self.fm_1st_dense = nn.ModuleList([nn.Linear(1, 1) for _ in range(dense_count)])
        total_fields = self.num_sparse + dense_count
        dnn_input_dim = total_fields * feature_dim
        self.dnn = DNN(dnn_input_dim, hidden_units, dropout)
        self.dnn_linear = nn.Linear(hidden_units[-1], 1)
        self.fm_2nd = FactorizationMachine()
        self.output_dim = 1
    def forward(self, x):
        sparse_x = x[:, :self.num_sparse].long()
        dense_x = x[:, self.num_sparse:]
        sparse_embeds = [self.embeddings[i](sparse_x[:, i]) for i in range(self.num_sparse)]
        dense_embeds = []
        if self.num_dense > 0:
            dense_embeds = [self.dense_embs[i](dense_x[:, i].unsqueeze(1)) for i in range(self.num_dense)]
        all_embeds = sparse_embeds + dense_embeds
        embed_stack = torch.stack(all_embeds, dim=1)
        fm_1st = sum([self.fm_1st_sparse[i](sparse_x[:, i]) for i in range(self.num_sparse)])
        if self.num_dense > 0:
            fm_1st += sum([self.fm_1st_dense[i](dense_x[:, i].unsqueeze(1)) for i in range(self.num_dense)])
        fm_2nd = self.fm_2nd(embed_stack)
        dnn_in = embed_stack.view(x.size(0), -1)
        dnn_out = self.dnn(dnn_in)
        dnn_score = self.dnn_linear(dnn_out)
        return fm_1st + fm_2nd + dnn_score
class DCNBackbone(nn.Module):
    def __init__(self, sparse_dims, dense_count, feature_dim=16, cross_layers=3, hidden_units=[256, 128], dropout=0.1):
        super().__init__()
        self.num_sparse = len(sparse_dims)
        self.num_dense = dense_count
        self.embeddings = nn.ModuleList([nn.Embedding(d, feature_dim) for d in sparse_dims])
        self.input_dim = self.num_sparse * feature_dim + dense_count
        self.cross_net = CrossNetwork(self.input_dim, cross_layers)
        self.dnn = DNN(self.input_dim, hidden_units, dropout)
        self.output_dim = self.input_dim + hidden_units[-1]
    def forward(self, x):
        sparse_x = x[:, :self.num_sparse].long()
        dense_x = x[:, self.num_sparse:]
        sparse_embeds = [self.embeddings[i](sparse_x[:, i]) for i in range(self.num_sparse)]
        sparse_flat = torch.cat(sparse_embeds, dim=1)
        if self.num_dense > 0:
            x_0 = torch.cat([sparse_flat, dense_x], dim=1)
        else:
            x_0 = sparse_flat
        cross_out = self.cross_net(x_0)
        deep_out = self.dnn(x_0)
        return torch.cat([cross_out, deep_out], dim=1)
class DCNv2Backbone(nn.Module):
    def __init__(self, sparse_dims, dense_count, feature_dim=16, cross_layers=3, hidden_units=[256, 128], dropout=0.1, low_rank=64):
        super().__init__()
        self.num_sparse = len(sparse_dims)
        self.num_dense = dense_count
        self.embeddings = nn.ModuleList([nn.Embedding(d, feature_dim) for d in sparse_dims])
        self.input_dim = self.num_sparse * feature_dim + dense_count
        self.cross_layers = nn.ModuleList([
            DCNv2Layer(self.input_dim, low_rank=low_rank)
            for _ in range(cross_layers)
        ])
        self.dnn = DNN(self.input_dim, hidden_units, dropout)
        self.output_dim = self.input_dim + hidden_units[-1]
    def forward(self, x):
        sparse_x = x[:, :self.num_sparse].long()
        dense_x = x[:, self.num_sparse:]
        sparse_embeds = [self.embeddings[i](sparse_x[:, i]) for i in range(self.num_sparse)]
        sparse_flat = torch.cat(sparse_embeds, dim=1)
        if self.num_dense > 0:
            x_0 = torch.cat([sparse_flat, dense_x], dim=1)
        else:
            x_0 = sparse_flat
        x_l = x_0
        for layer in self.cross_layers:
            x_l = layer(x_l, x_0)
        cross_out = x_l
        deep_out = self.dnn(x_0)
        return torch.cat([cross_out, deep_out], dim=1)
class AutoIntBackbone(nn.Module):
    def __init__(self, sparse_dims, dense_count, feature_dim=32, attention_layers=3, num_heads=2, hidden_units=[256, 128], dropout=0.1):
        super().__init__()
        self.num_sparse = len(sparse_dims)
        self.num_dense = dense_count
        self.sparse_embs = nn.ModuleList([nn.Embedding(d, feature_dim) for d in sparse_dims])
        if dense_count > 0:
            self.dense_proj = nn.ModuleList([nn.Linear(1, feature_dim) for _ in range(dense_count)])
        self.att_layers = nn.ModuleList([
            AutoIntLayer(feature_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(attention_layers)
        ])
        num_fields = self.num_sparse + dense_count
        self.flat_dim = num_fields * feature_dim
        self.dnn = DNN(self.flat_dim, hidden_units, dropout)
        self.output_dim = hidden_units[-1]
    def forward(self, x):
        sparse_x = x[:, :self.num_sparse].long()
        dense_x = x[:, self.num_sparse:]
        embs = [self.sparse_embs[i](sparse_x[:, i]) for i in range(self.num_sparse)]
        if self.num_dense > 0:
            embs.extend([self.dense_proj[i](dense_x[:, i].unsqueeze(1)) for i in range(self.num_dense)])
        att_input = torch.stack(embs, dim=1)
        for layer in self.att_layers:
            att_input = layer(att_input)
        flat_input = att_input.reshape(x.size(0), -1)
        output = self.dnn(flat_input)
        return output
class FiBiNETBackbone(nn.Module):
    def __init__(self, sparse_dims, dense_count, feature_dim=32, hidden_units=[256, 128], dropout=0.1):
        super().__init__()
        self.num_sparse = len(sparse_dims)
        self.num_dense = dense_count
        num_fields = self.num_sparse + dense_count
        self.sparse_embs = nn.ModuleList([nn.Embedding(d, feature_dim) for d in sparse_dims])
        if dense_count > 0:
            self.dense_proj = nn.ModuleList([nn.Linear(1, feature_dim) for _ in range(dense_count)])
        self.senet = SENETLayer(num_fields, reduction_ratio=3)
        self.bilinear = BilinearInteraction(feature_dim, num_fields, type='Field-All')
        num_pairs = num_fields * (num_fields - 1) // 2
        deep_input_dim = (num_fields * feature_dim * 2) + (num_pairs * feature_dim * 2)
        self.dnn = DNN(deep_input_dim, hidden_units, dropout)
        self.output_dim = hidden_units[-1]
    def forward(self, x):
        sparse_x = x[:, :self.num_sparse].long()
        dense_x = x[:, self.num_sparse:]
        embs = [self.sparse_embs[i](sparse_x[:, i]) for i in range(self.num_sparse)]
        if self.num_dense > 0:
            embs.extend([self.dense_proj[i](dense_x[:, i].unsqueeze(1)) for i in range(self.num_dense)])
        E = torch.stack(embs, dim=1)
        V = self.senet(E)
        p = self.bilinear(E)
        q = self.bilinear(V)
        E_flat = E.view(E.size(0), -1)
        V_flat = V.view(V.size(0), -1)
        dnn_input = torch.cat([E_flat, V_flat, p, q], dim=1)
        output = self.dnn(dnn_input)
        return output
# %%
class RecDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.N = X.shape[0]
    def __len__(self):
        return self.N
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]