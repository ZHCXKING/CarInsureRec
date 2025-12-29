import torch
import torch.nn as nn
import torch.nn.functional as F
class DCNv2Layer(nn.Module):
    """
    Deep & Cross Network V2 (Low-Rank version for efficiency)
    Explicitly captures high-order feature interactions.
    """
    def __init__(self, input_dim, low_rank=64, num_experts=1):
        super().__init__()
        self.input_dim = input_dim
        # U: [D, R], V: [R, D] -> W = U * V (Rank-deficient approximation)
        self.U = nn.Parameter(torch.Tensor(num_experts, input_dim, low_rank))
        self.V = nn.Parameter(torch.Tensor(num_experts, low_rank, input_dim))
        self.b = nn.Parameter(torch.Tensor(num_experts, input_dim))
        # Gating for experts (if num_experts > 1, basically MoE)
        self.gating = nn.Linear(input_dim, num_experts) if num_experts > 1 else None
        self._init_weights()
    def _init_weights(self):
        nn.init.xavier_uniform_(self.U)
        nn.init.xavier_uniform_(self.V)
        nn.init.zeros_(self.b)
    def forward(self, x):
        # x: [Batch, Input_Dim]
        # x_0 is usually the original input, but in stacked layers x_l is input
        # Standard DCNv2 formula: x_{l+1} = x_0 * (W * x_l + b) + x_l
        # Here we implement one layer logic: out = x * (W * x + b) + x
        # Low-rank: W = U * V
        batch_size = x.shape[0]
        # [Batch, 1, Dim] * [1, Experts, Dim, Low] -> [Batch, Experts, Low] (Optimization via internal MatMul)
        # Simplified for 1 expert:
        xv = torch.mm(x, self.V[0].t())  # [B, R]
        wx = torch.mm(xv, self.U[0].t())  # [B, D]
        output = x * (wx + self.b[0]) + x
        return output
class OptimizedHybridBackbone(nn.Module):
    def __init__(self, sparse_dims, dense_count, feature_dim=32,
                 cross_layers=3, attention_layers=2, num_heads=4,
                 hidden_units=[256, 128], dropout=0.1,
                 low_rank_dim=64):
        super().__init__()
        self.num_sparse = len(sparse_dims)
        self.num_dense = dense_count
        self.feature_dim = feature_dim

        # 1. Embedding Layer
        self.sparse_embs = nn.ModuleList([nn.Embedding(d, feature_dim) for d in sparse_dims])
        if dense_count > 0:
            self.dense_proj = nn.ModuleList([nn.Linear(1, feature_dim) for _ in range(dense_count)])

        self.num_fields = self.num_sparse + self.num_dense
        self.total_input_dim = self.num_fields * feature_dim

        # 2. SeNet (Feature Importance Selection)
        self.senet = SeNetGate(self.num_fields, reduction_ratio=2)

        # 3. Cross Tower (DCNv2 - Explicit Interaction)
        # 保持输入输出维度一致，便于残差
        self.dcn_layers = nn.ModuleList([
            DCNv2Layer(self.total_input_dim, low_rank=low_rank_dim)
            for _ in range(cross_layers)
        ])

        # 4. Deep Tower (DNN - Implicit Interaction)
        deep_layers = []
        in_dim = self.total_input_dim
        for hidden in hidden_units:
            deep_layers.extend([
                nn.Linear(in_dim, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden
        self.deep_net = nn.Sequential(*deep_layers)
        self.deep_out_dim = hidden_units[-1]

        # 5. Attention Tower (Contextual Interaction)
        # 使用 Transformer Encoder Layer 标准实现
        encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=num_heads,
                                                   dim_feedforward=feature_dim * 2,
                                                   dropout=dropout, batch_first=True)
        self.att_encoder = nn.TransformerEncoder(encoder_layer, num_layers=attention_layers)

        # Attention Projection (Pooling后投影)
        # Concat(Mean, Max) -> 2 * feature_dim
        self.att_out_dim = feature_dim * 2

        # Final Norms
        self.ln_cross = nn.LayerNorm(self.total_input_dim)
        self.ln_deep = nn.LayerNorm(self.deep_out_dim)
        self.ln_att = nn.LayerNorm(self.att_out_dim)

        # Final Output Dimension
        self.output_dim = self.total_input_dim + self.deep_out_dim + self.att_out_dim
    def forward(self, x):
        # --- 1. Embedding Lookup ---
        sparse_x = x[:, :self.num_sparse].long()
        embs = [emb(sparse_x[:, i]) for i, emb in enumerate(self.sparse_embs)]

        if self.num_dense > 0:
            dense_x = x[:, self.num_sparse:]
            embs.extend([self.dense_proj[i](dense_x[:, i].unsqueeze(1)) for i in range(self.num_dense)])

        # [Batch, Fields, Emb_Dim]
        stacked_embs = torch.stack(embs, dim=1)

        # --- 2. SeNet Gating ---
        # 动态调整特征权重
        gated_embs = self.senet(stacked_embs)

        # Flatten for Cross and Deep towers: [Batch, Fields * Emb_Dim]
        flat_input = gated_embs.view(x.size(0), -1)

        # --- 3. Cross Tower (DCNv2) ---
        cross_out = flat_input
        for layer in self.dcn_layers:
            cross_out = layer(cross_out)
        cross_out = self.ln_cross(cross_out)  # [Batch, Total_Input_Dim]

        # --- 4. Deep Tower (DNN) ---
        deep_out = self.deep_net(flat_input)
        deep_out = self.ln_deep(deep_out)  # [Batch, Hidden_Last]

        # --- 5. Attention Tower (Transformer) ---
        # Input: [Batch, Fields, Emb_Dim]
        att_feat = self.att_encoder(gated_embs)

        # Pooling Strategy: 避免 Flatten 导致的维度爆炸
        # Global Average Pooling
        avg_pool = torch.mean(att_feat, dim=1)
        # Global Max Pooling
        max_pool = torch.max(att_feat, dim=1)[0]

        att_out = torch.cat([avg_pool, max_pool], dim=1)
        att_out = self.ln_att(att_out)  # [Batch, Emb_Dim * 2]

        # --- 6. Fusion ---
        final_out = torch.cat([cross_out, deep_out, att_out], dim=1)

        return final_out