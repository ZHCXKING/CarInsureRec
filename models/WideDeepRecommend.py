# %%
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from .base import BaseRecommender

# %%
# 1. 定义原生的 PyTorch Wide & Deep 模型
class WideDeepModel(nn.Module):
    def __init__(self, sparse_feature_info, dense_feature_count, num_classes,
                 embedding_dim=16, hidden_units=[128, 64], dropout=0.5):
        """
        :param sparse_feature_info: List[Tuple], 格式为 [(feat_name, vocab_size), ...]
        :param dense_feature_count: Int, 连续特征的数量
        :param num_classes: Int, 分类数量 (Item总数)
        :param embedding_dim: Int, Deep侧的隐向量维度
        """
        super(WideDeepModel, self).__init__()
        self.num_classes = num_classes
        self.dense_count = dense_feature_count

        # =========================
        # Wide Part (Linear)
        # =========================
        # 稀疏特征的 Linear 部分：实际上是为每个特征值学习一个对应类别的 bias
        # Embedding(vocab_size, num_classes)
        self.wide_sparse = nn.ModuleList([
            nn.Embedding(vocab_size, num_classes)
            for _, vocab_size in sparse_feature_info
        ])

        # 稠密特征的 Linear 部分
        if self.dense_count > 0:
            self.wide_dense = nn.Linear(dense_feature_count, num_classes)

        # =========================
        # Deep Part (DNN)
        # =========================
        # Deep侧的 Embedding：用于将稀疏特征转为稠密向量
        self.deep_embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embedding_dim)
            for _, vocab_size in sparse_feature_info
        ])

        # 计算 DNN 输入维度 = (稀疏特征数 * emb_dim) + 稠密特征数
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

        # Deep侧的输出头
        self.deep_linear = nn.Linear(last_dim, num_classes)

    def forward(self, X_sparse, X_dense):
        """
        X_sparse: [Batch, Num_Sparse_Feats] (Long)
        X_dense:  [Batch, Num_Dense_Feats]  (Float)
        """
        batch_size = X_sparse.size(0)

        # --- 1. Wide Part Forward ---
        wide_logit = torch.zeros(batch_size, self.num_classes, device=X_sparse.device)

        # Wide Sparse
        for i, emb in enumerate(self.wide_sparse):
            wide_logit += emb(X_sparse[:, i])

        # Wide Dense
        if self.dense_count > 0:
            wide_logit += self.wide_dense(X_dense)

        # --- 2. Deep Part Forward ---
        # Lookup Embeddings
        deep_emb_list = []
        for i, emb in enumerate(self.deep_embeddings):
            deep_emb_list.append(emb(X_sparse[:, i]))  # [B, emb_dim]

        # Concat Embeddings: [B, num_sparse * emb_dim]
        deep_input_emb = torch.cat(deep_emb_list, dim=1)

        # Concat with Dense Features
        if self.dense_count > 0:
            dnn_input = torch.cat([deep_input_emb, X_dense], dim=1)
        else:
            dnn_input = deep_input_emb

        dnn_output = self.dnn(dnn_input)
        deep_logit = self.deep_linear(dnn_output)

        # --- 3. Final Sum ---
        total_logit = wide_logit + deep_logit
        return total_logit


# %%
# 2. 包装推荐器类
class WideDeepRecommend(BaseRecommender):
    def __init__(self, user_name: list, item_name: str, date_name: str | None = None,
                 sparse_features: list | None = None, dense_features: list | None = None, standard_bool: bool = False,
                 seed: int = 42, k: int = 3, **kwargs):
        if (user_name is None) or (item_name is None):
            raise ValueError('user_name and item_name are required')
        super().__init__('WideDeep', user_name, item_name, date_name, sparse_features, dense_features, standard_bool, seed, k)

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
        self.sparse_feature_info = []

    # %%
    def fit(self, train_data: pd.DataFrame):
        self.out_dim = train_data[self.item_name].nunique()
        self.unique_item = list(range(self.out_dim))

        # 1. 数据映射与标准化
        train_data = self._mapping(train_data, fit_bool=True)
        if self.standard_bool:
            train_data = self._standardize(train_data, fit_bool=True)

        # 2. 记录特征维度信息
        self.sparse_feature_info = [
            (feat, self.vocabulary_sizes[feat])
            for feat in self.sparse_features
        ]

        # 3. 准备 Tensor 数据
        # Sparse
        if len(self.sparse_features) > 0:
            X_sparse = torch.tensor(train_data[self.sparse_features].values, dtype=torch.long)
        else:
            X_sparse = torch.zeros((len(train_data), 0), dtype=torch.long)

        # Dense
        if len(self.dense_features) > 0:
            X_dense = torch.tensor(train_data[self.dense_features].values, dtype=torch.float)
        else:
            X_dense = torch.zeros((len(train_data), 0), dtype=torch.float)

        # Label
        y = torch.tensor(train_data[self.item_name].values, dtype=torch.long)

        # 4. DataLoader
        train_dataset = TensorDataset(X_sparse, X_dense, y)
        train_loader = DataLoader(train_dataset, batch_size=self.kwargs['batch_size'], shuffle=True)

        # 5. 初始化模型
        self.model = WideDeepModel(
            sparse_feature_info=self.sparse_feature_info,
            dense_feature_count=len(self.dense_features),
            num_classes=self.out_dim,
            embedding_dim=self.kwargs['embedding_dim'],
            hidden_units=self.kwargs['hidden_units'],
            dropout=self.kwargs['dropout']
        ).to(self.device)

        # 6. 训练配置
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.kwargs['lr'])

        # 7. 训练循环
        self.model.train()
        print(f"Start Training WideDeep on {self.device}...")

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

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.kwargs['epochs']}, Loss: {total_loss / len(train_loader):.4f}")

        self.is_trained = True

    # %%
    def get_proba(self, test_data: pd.DataFrame):
        if not self.is_trained:
            raise ValueError('model is not trained')

        # 1. 预处理
        test_data = self._mapping(test_data, fit_bool=False)
        if self.standard_bool:
            test_data = self._standardize(test_data, fit_bool=False)

        # 2. 准备 Tensor
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
            logits = self.model(X_sparse, X_dense)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        # 4. 返回结果
        result = pd.DataFrame(probs, index=test_data.index, columns=self.unique_item)
        return result