# %%
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from .base import BaseRecommender  # 假设这个在你本地文件里


# %%
# 1. 定义原生的 PyTorch DeepFM 模型
class DeepFMModel(nn.Module):
    def __init__(self, sparse_feature_info, dense_feature_count, num_classes,
                 embedding_dim=16, hidden_units=[128, 64], dropout=0.5):
        """
        :param sparse_feature_info: List[Tuple], 格式为 [(feat_name, vocab_size), ...]
        :param dense_feature_count: Int, 连续特征的数量
        :param num_classes: Int, 分类数量 (Item总数)
        :param embedding_dim: Int, 隐向量维度
        """
        super(DeepFMModel, self).__init__()
        self.num_classes = num_classes
        self.sparse_feature_info = sparse_feature_info
        self.dense_count = dense_feature_count

        # --- 1. Shared Embeddings (用于FM和DNN) ---
        # 建立一个ModuleList，索引对应sparse_feature_info的顺序
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embedding_dim)
            for _, vocab_size in sparse_feature_info
        ])

        # --- 2. Linear Part (First Order) ---
        # 也就是 Wide 部分。对于多分类，我们需要为每个类别学习一个权重
        # Sparse: 每个特征值的权重是 (vocab_size, num_classes)
        self.linear_sparse = nn.ModuleList([
            nn.Embedding(vocab_size, num_classes)
            for _, vocab_size in sparse_feature_info
        ])
        # Dense: 线性层 (dense_count -> num_classes)
        if self.dense_count > 0:
            self.linear_dense = nn.Linear(dense_feature_count, num_classes)

        # --- 3. FM Part (Second Order) ---
        # FM计算的是标量交互，这里我们通过一个线性层将其映射到 num_classes
        # 或者仅仅将其作为一种特征增强。标准DeepFM用于CTR(输出1维)，
        # 这里为了适配多分类推荐，我们对FM的标量输出做一个投影。
        self.fm_projection = nn.Linear(1, num_classes)

        # --- 4. DNN Part (High Order) ---
        # 输入维度 = 所有sparse特征的emb维度之和 + dense特征数量
        input_dim = len(sparse_feature_info) * embedding_dim + dense_feature_count

        layers = []
        last_dim = input_dim
        for hidden in hidden_units:
            layers.append(nn.Linear(last_dim, hidden))
            layers.append(nn.BatchNorm1d(hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            last_dim = hidden
        self.dnn = nn.Sequential(*layers)

        # DNN Output Head
        self.dnn_linear = nn.Linear(last_dim, num_classes)

    def forward(self, X_sparse, X_dense):
        """
        X_sparse: [Batch, Num_Sparse_Feats] (Long)
        X_dense:  [Batch, Num_Dense_Feats]  (Float)
        """
        batch_size = X_sparse.size(0)

        # =========================
        # A. Linear Component
        # =========================
        linear_logit = torch.zeros(batch_size, self.num_classes, device=X_sparse.device)
        # Sparse Linear
        for i, emb in enumerate(self.linear_sparse):
            # emb(idx) -> [B, num_classes]
            linear_logit += emb(X_sparse[:, i])
        # Dense Linear
        if self.dense_count > 0:
            linear_logit += self.linear_dense(X_dense)

        # =========================
        # B. Embedding Lookup
        # =========================
        # 获取所有sparse特征的k维向量
        sparse_embs = []  # List of [B, k]
        for i, emb in enumerate(self.embeddings):
            sparse_embs.append(emb(X_sparse[:, i]))

        # 堆叠成 [B, Num_Sparse, k] 用于FM计算
        sparse_embs_stack = torch.stack(sparse_embs, dim=1)

        # =========================
        # C. FM Component
        # =========================
        # Formula: 0.5 * [ (Sum v_i)^2 - Sum (v_i^2) ]
        square_of_sum = torch.pow(torch.sum(sparse_embs_stack, dim=1), 2)  # [B, k]
        sum_of_square = torch.sum(torch.pow(sparse_embs_stack, 2), dim=1)  # [B, k]
        cross_term = 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)  # [B, 1]

        # 将标量FM交互映射到分类空间
        fm_logit = self.fm_projection(cross_term)  # [B, num_classes]

        # =========================
        # D. DNN Component
        # =========================
        # Flatten embeddings: [B, Num_Sparse * k]
        dnn_input_sparse = sparse_embs_stack.view(batch_size, -1)

        if self.dense_count > 0:
            dnn_input = torch.cat([dnn_input_sparse, X_dense], dim=1)
        else:
            dnn_input = dnn_input_sparse

        dnn_output = self.dnn(dnn_input)
        dnn_logit = self.dnn_linear(dnn_output)  # [B, num_classes]

        # =========================
        # E. Final Sum
        # =========================
        total_logit = linear_logit + fm_logit + dnn_logit
        return total_logit


# %%
# 2. 包装推荐器类
class DeepFMRecommend(BaseRecommender):
    def __init__(self, user_name: list, item_name: str, date_name: str | None = None,
                 sparse_features: list | None = None, dense_features: list | None = None, standard_bool: bool = False,
                 seed: int = 42, k: int = 3, **kwargs):
        if (user_name is None) or (item_name is None):
            raise ValueError('user_name and item_name are required')
        super().__init__('DeepFM', user_name, item_name, date_name, sparse_features, dense_features, standard_bool, seed, k)

        default_params = {
            'embedding_dim': 16,
            'batch_size': 8,
            'epochs': 200,
            'lr': 0.001,
            'hidden_units': [128, 64],
            'dropout': 0.2
        }
        self.kwargs.update(default_params)
        self.kwargs.update(kwargs)

        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.out_dim = None

        # 移除 rec4torch 的 feature_columns 依赖
        # 只需要记录信息即可
        self.sparse_feature_info = []  # List[(name, vocab_size)]


    def fit(self, train_data: pd.DataFrame):
        # 1. 基础数据处理 (Mapping + Standardization)
        # BaseRecommender 会把 sparse列转成 0~N 的整数，dense列标准化
        self.out_dim = train_data[self.item_name].nunique()
        self.unique_item = list(range(self.out_dim))

        train_data = self._mapping(train_data, fit_bool=True)
        if self.standard_bool:
            train_data = self._standardize(train_data, fit_bool=True)

        # 2. 准备特征信息
        self.sparse_feature_info = [
            (feat, self.vocabulary_sizes[feat])
            for feat in self.sparse_features
        ]

        # 3. 转换为 Tensor
        # 提取 Sparse 特征
        if len(self.sparse_features) > 0:
            X_sparse = torch.tensor(train_data[self.sparse_features].values, dtype=torch.long)
        else:
            # 如果没有sparse特征，给一个空的占位
            X_sparse = torch.zeros((len(train_data), 0), dtype=torch.long)

        # 提取 Dense 特征
        if len(self.dense_features) > 0:
            X_dense = torch.tensor(train_data[self.dense_features].values, dtype=torch.float)
        else:
            X_dense = torch.zeros((len(train_data), 0), dtype=torch.float)

        # 提取 Label
        y = torch.tensor(train_data[self.item_name].values, dtype=torch.long)

        # 4. DataLoader
        train_dataset = TensorDataset(X_sparse, X_dense, y)
        train_loader = DataLoader(train_dataset, batch_size=self.kwargs['batch_size'], shuffle=True)

        # 5. 初始化模型
        self.model = DeepFMModel(
            sparse_feature_info=self.sparse_feature_info,
            dense_feature_count=len(self.dense_features),
            num_classes=self.out_dim,
            embedding_dim=self.kwargs['embedding_dim'],
            hidden_units=self.kwargs['hidden_units'],
            dropout=self.kwargs['dropout']
        ).to(self.device)

        # 6. 训练循环
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.kwargs['lr'])

        self.model.train()
        print(f"Start Training DeepFM on {self.device}...")

        for epoch in range(self.kwargs['epochs']):
            total_loss = 0.0
            for batch_sparse, batch_dense, batch_y in train_loader:
                batch_sparse, batch_dense, batch_y = batch_sparse.to(self.device), batch_dense.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                logits = self.model(batch_sparse, batch_dense)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            # 可选：打印每个 epoch 的 loss
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.kwargs['epochs']}, Loss: {total_loss / len(train_loader):.4f}")

        self.is_trained = True

    def get_proba(self, test_data: pd.DataFrame):
        if not self.is_trained:
            raise ValueError('model is not trained')

        # 1. 基础处理
        test_data = self._mapping(test_data, fit_bool=False)
        if self.standard_bool:
            test_data = self._standardize(test_data, fit_bool=False)

        # 2. 转换为 Tensor
        if len(self.sparse_features) > 0:
            X_sparse = torch.tensor(test_data[self.sparse_features].values, dtype=torch.long)
        else:
            X_sparse = torch.zeros((len(test_data), 0), dtype=torch.long)

        if len(self.dense_features) > 0:
            X_dense = torch.tensor(test_data[self.dense_features].values, dtype=torch.float)
        else:
            X_dense = torch.zeros((len(test_data), 0), dtype=torch.float)

        X_sparse = X_sparse.to(self.device)
        X_dense = X_dense.to(self.device)

        # 3. 推理
        self.model.eval()
        with torch.no_grad():
            # 考虑到测试集可能很大，建议这里也用 DataLoader 或分批处理，
            # 但为了保持与原代码逻辑一致（一次性predict），这里直接传
            logits = self.model(X_sparse, X_dense)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        # 4. 结果包装
        result = pd.DataFrame(probs, index=test_data.index, columns=self.unique_item)
        return result