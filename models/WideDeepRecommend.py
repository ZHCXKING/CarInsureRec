# %%
import pandas as pd
import torch
from .base import BaseRecommender
from scipy.special import softmax
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim
from rec4torch.models import WideDeep
from rec4torch.inputs import SparseFeat, DenseFeat, build_input_array


# https://github.com/Tongjilibo/rec4torch
# %%
class WideDeepRecommend(BaseRecommender):
    def __init__(self, user_name: list, item_name: str, date_name: str | None = None,
                 sparse_features: list | None = None, dense_features: list | None = None, standard_bool: bool = False,
                 seed: int = 42, **kwargs):
        if (user_name is None) or (item_name is None):
            raise ValueError('user_name and item_name are required')
        super().__init__('WideDeep', user_name, item_name, date_name, sparse_features, dense_features, standard_bool, seed)
        default_params = {
            'embedding_dim': 16,
            'dimension': 1,
            'batch_size': 8,
            'epochs': 200,
            'lr': 0.001
        }
        self.kwargs.update(default_params)
        self.kwargs.update(kwargs)
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.out_dim = None
        self.linear_feature_columns = None
        self.dnn_feature_columns = None
        torch.manual_seed(self.seed)

    # %%
    def fit(self, train_data: pd.DataFrame):
        self.out_dim = train_data[self.item_name].nunique()
        self.unique_item = list(range(self.out_dim))
        train_data = self._mapping(train_data, fit_bool=True)
        if self.standard_bool:
            train_data = self._Standardize(train_data, fit_bool=True)
        sparse_feature_columns = [
            SparseFeat(feat, self.vocabulary_sizes[feat], embedding_dim=self.kwargs['embedding_dim'])
            for feat in self.sparse_feature
        ]
        dense_feature_columns = [
            DenseFeat(feat, dimension=self.kwargs['dimension'])
            for feat in self.dense_feature
        ]
        feature_columns = sparse_feature_columns + dense_feature_columns
        self.linear_feature_columns = feature_columns
        self.dnn_feature_columns = feature_columns
        X, y = build_input_array(train_data, self.linear_feature_columns + self.dnn_feature_columns, target=self.item_name)
        X = torch.tensor(X, dtype=torch.float, device=self.device)
        y = torch.tensor(y, dtype=torch.int64, device=self.device)
        train_dataset = TensorDataset(X, y)
        train_loader = DataLoader(train_dataset, batch_size=self.kwargs['batch_size'], shuffle=True)
        self.model = WideDeep(self.linear_feature_columns, self.dnn_feature_columns, out_dim=self.out_dim)
        self.model.to(self.device)
        self.model.compile(
            loss=nn.CrossEntropyLoss(),
            optimizer=optim.Adam(self.model.parameters(), lr=self.kwargs['lr'])
        )
        self.model.fit(train_loader, epochs=self.kwargs['epochs'])
        self.is_trained = True

    # %%
    def get_proba(self, test_data: pd.DataFrame):
        if not self.is_trained:
            raise ValueError('model is not trained')
        test_data = self._mapping(test_data, fit_bool=False)
        if self.standard_bool:
            test_data = self._Standardize(test_data, fit_bool=False)
        X, _ = build_input_array(test_data, self.linear_feature_columns + self.dnn_feature_columns, target=self.item_name)
        X = torch.tensor(X, dtype=torch.float, device=self.device)
        y = self.model.predict(X).cpu().numpy()
        y = softmax(y, axis=1)
        result = pd.DataFrame(y, index=test_data.index, columns=self.unique_item)
        return result
