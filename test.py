# %%
class CoMICEHead(nn.Module):
    def __init__(self, input_dim, num_classes, proj_dim=128, hidden_dim=256, dropout_rate=0.1):
        super().__init__()

        # --- 修改部分 ---
        # 原版：2层 MLP (Linear -> BN -> ReLU -> Linear)
        # 修改版：单层线性映射。直接将 Backbone 输出映射到投影空间。
        # 这样可以大幅减少参数量，并降低辅助任务的计算开销。
        self.projection_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim)
        )

        # 主分类任务通常仍需要 MLP 来保证非线性拟合能力，保持不变
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
        # 投影头直接线性变换
        proj_feat = self.projection_head(feature_vector)
        # 依然需要归一化，因为对比学习依赖余弦相似度
        proj_norm = F.normalize(proj_feat, dim=1)

        # 分类头输出 Logits
        logits = self.classifier_head(feature_vector)

        return logits, proj_norm