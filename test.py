class TabularAugmentation(nn.Module):
    def __init__(self, sparse_idxs, dense_idxs, mask_prob=0.2, noise_std=0.01):
        super().__init__()
        self.sparse_idxs = sparse_idxs  # 稀疏特征在输入tensor中的列索引
        self.dense_idxs = dense_idxs  # 稠密特征在输入tensor中的列索引
        self.mask_prob = mask_prob
        self.noise_std = noise_std
    def forward(self, x):
        if not self.training:
            return x
        x_aug = x.clone()
        # 1. Sparse Masking: 随机将部分 Categorical 特征置为 0 (假设0是padding/unknown)
        if len(self.sparse_idxs) > 0:
            mask = torch.rand(x.shape[0], len(self.sparse_idxs), device=x.device) < self.mask_prob
            # 构建这就需要知道 sparse 特征在 x 中的具体位置
            # 假设 x 的前 len(sparse) 列是 sparse
            sparse_part = x_aug[:, self.sparse_idxs]
            sparse_part[mask] = 0
            x_aug[:, self.sparse_idxs] = sparse_part
        # 2. Dense Noise: 添加高斯噪声
        if len(self.dense_idxs) > 0:
            noise = torch.randn(x.shape[0], len(self.dense_idxs), device=x.device) * self.noise_std
            x_aug[:, self.dense_idxs] += noise
        return x_aug
class CoMICERecommend(BaseRecommender):
    def __init__(self, item_name: str, sparse_features: list, dense_features: list,
                 standard_bool: bool = True, seed: int = 42, k: int = 3, **kwargs):
        super().__init__('CoMICE', item_name, sparse_features, dense_features, standard_bool, seed, k)
        default_params = {
            'lr': 1e-3,  # 对比学习通常需要稍大的 LR
            'batch_size': 1024,  # 增大 Batch Size 对比学习至关重要
            'epochs': 50,
            'lambda_nce': 0.1,  # 降低辅助损失权重，避免干扰主分类任务
            'temperature': 0.07,  # 经典的温度系数
            'proj_dim': 64,  # 增大投影维度
            'backbone': 'DCNv2',  # 推荐使用 DCNv2 或 FiBiNET 作为强力骨干
            'feature_dim': 32,
            'hidden_units': [256, 128],
            'dropout': 0.1,
            'mask_prob': 0.15,  # 新增：Mask概率
            'noise_std': 0.05,  # 新增：噪声标准差
        }
        self.kwargs.update(default_params)
        self.kwargs.update(kwargs)
        self.user_name = sparse_features + dense_features
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        set_seed(self.seed)
    def _build_model(self, sparse_dims, dense_count, num_classes):
        # ... (保持原有的 backbone 构建逻辑不变) ...
        # 这里为了节省篇幅省略 backbone 的 if-else 选择代码，与原代码一致
        # 请确保使用 kwargs 中的 backbone 参数
        # 复制原代码的 backbone 选择逻辑
        common_args = {
            'sparse_dims': sparse_dims,
            'dense_count': dense_count,
            'feature_dim': self.kwargs['feature_dim'],
            'hidden_units': self.kwargs['hidden_units'],
            'dropout': self.kwargs['dropout']
        }
        backbone_name = self.kwargs['backbone']
        if backbone_name == 'Hybrid':
            backbone = HybridBackbone(cross_layers=self.kwargs.get('cross_layers', 3), **common_args)
        elif backbone_name == 'DCNv2':
            backbone = DCNv2Backbone(cross_layers=self.kwargs.get('cross_layers', 3), **common_args)
        elif backbone_name == 'FiBiNET':
            backbone = FiBiNETBackbone(**common_args)
        else:
            # 默认 fallback
            backbone = DCNv2Backbone(cross_layers=3, **common_args)

        head = CoMICEHead(backbone.output_dim, num_classes, proj_dim=self.kwargs['proj_dim'])
        model = CoMICEModel(backbone, head)

        # 初始化增强模块
        sparse_idxs = list(range(len(self.sparse_features)))
        dense_idxs = list(range(len(self.sparse_features), len(self.sparse_features) + len(self.dense_features)))
        self.augmenter = TabularAugmentation(
            sparse_idxs, dense_idxs,
            mask_prob=self.kwargs['mask_prob'],
            noise_std=self.kwargs['noise_std']
        ).to(self.device)

        return model.to(self.device)
    def train_one_epoch(self, dataloader):
        self.model.train()
        total_loss, ce_acc, nce_acc = 0.0, 0.0, 0.0
        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            # --- 优化策略：基于增强的对比 ---
            # 1. 原始输入 -> 用于分类 (Main Task)
            logits_clean, _ = self.model(x)
            loss_ce = F.cross_entropy(logits_clean, y)
            # 2. 增强输入 -> 用于对比 (Auxiliary Task)
            # 生成两个视角的增强数据 (View 1, View 2)
            x_aug1 = self.augmenter(x)
            x_aug2 = self.augmenter(x)
            _, proj1 = self.model(x_aug1)
            _, proj2 = self.model(x_aug2)
            # 拼接 proj 和 labels 用于 Supervised Contrastive Loss
            # 现在的 batch size 变成了 2N
            features = torch.cat([proj1, proj2], dim=0)
            labels = torch.cat([y, y], dim=0)
            batch_size = labels.shape[0]  # 2N
            # 计算相似度矩阵
            sim_matrix = torch.matmul(features, features.T) / self.kwargs['temperature']
            # 数值稳定性处理
            sim_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
            sim_matrix = sim_matrix - sim_max.detach()
            # Mask 1: 自身对自身的 mask (对角线)
            logits_mask = torch.scatter(
                torch.ones_like(sim_matrix),
                1,
                torch.arange(batch_size, device=self.device).view(-1, 1),
                0
            )
            # Mask 2: 正样本 mask (Label 相同)
            # label_matrix: [2N, 2N], (i,j)=1 if label[i]==label[j]
            label_matrix = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1)).float()
            mask = label_matrix * logits_mask  # 排除对角线
            # 计算 Log Prob
            exp_sim = torch.exp(sim_matrix) * logits_mask
            log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
            # 计算 Mean Log-Likelihood over positive pairs
            # 只计算那些确实存在正样本的行
            mask_sum = mask.sum(dim=1)
            mask_valid_idx = mask_sum > 0
            if mask_valid_idx.any():
                mean_log_prob_pos = (mask * log_prob).sum(dim=1)[mask_valid_idx] / mask_sum[mask_valid_idx]
                loss_nce = -mean_log_prob_pos.mean()
            else:
                loss_nce = torch.tensor(0.0, device=self.device)
            # Total Loss
            loss = loss_ce + self.kwargs['lambda_nce'] * loss_nce
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            ce_acc += loss_ce.item()
            nce_acc += loss_nce.item()
        n = len(dataloader)
        return total_loss / n, ce_acc / n, nce_acc / n
    # fit 和 get_proba 方法保持不变，
    # 但建议 fit 中 patience 设为 10，因为对比学习收敛可能稍慢但后期更稳