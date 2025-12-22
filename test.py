# %%
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from .base import BaseRecommender
from src.utils import filling, round
# 假设上面的网络定义在 src.network 中
from src.network import HybridBackbone, AutoIntBackbone, DCNv2Backbone, DeepFMBackbone, WideDeepBackbone, CoMICEHead, CoMICEModel, StandardModel, set_seed, RecDataset
# %%
class NetworkRecommender(BaseRecommender):
    def __init__(self, model_name: str, backbone_class, item_name: str, sparse_features: list, dense_features: list,
                 standard_bool: bool = True, seed: int = 42, k: int = 3, **kwargs):
        super().__init__(model_name, item_name, sparse_features, dense_features, standard_bool, seed, k)
        self.backbone_class = backbone_class

        # 在 default_params 中包含 dropout，默认为 0.1
        default_params = {
            'lr': 1e-4,
            'batch_size': 512,
            'feature_dim': 32,
            'epochs': 200,
            'hidden_units': [256, 128],
            'cross_layers': 3,
            'attention_layers': 3,
            'num_heads': 2,
            'dropout': 0.1,  # 所有子类模型 (DeepFM, WideDeep, DCN, AutoInt) 默认都会用到
            'mice_method': 'MICE_RF'
        }
        self.kwargs.update(default_params)
        self.kwargs.update(kwargs)

        self.user_name = sparse_features + dense_features
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        set_seed(self.seed)
    def _build_model(self, sparse_dims, dense_count, num_classes):
        # 将 dropout 加入 common_args，确保所有 Backbone 都能接收到
        common_args = {
            'sparse_dims': sparse_dims,
            'dense_count': dense_count,
            'feature_dim': self.kwargs['feature_dim'],
            'hidden_units': self.kwargs['hidden_units'],
            'dropout': self.kwargs['dropout']
        }

        if self.backbone_class == DCNv2Backbone:
            backbone = self.backbone_class(
                cross_layers=self.kwargs['cross_layers'],
                **common_args
            )
        elif self.backbone_class == AutoIntBackbone:
            backbone = self.backbone_class(
                attention_layers=self.kwargs['attention_layers'],
                num_heads=self.kwargs['num_heads'],
                **common_args
            )
        else:
            # DeepFM 和 WideDeep 现在也通过 common_args 接收 dropout
            backbone = self.backbone_class(**common_args)

        return StandardModel(backbone, num_classes).to(self.device)
    # ... fit 和 get_proba 方法保持不变 ...
# %%
class DCNv2Recommend(NetworkRecommender):
    def __init__(self, item_name, sparse_features, dense_features, **kwargs):
        super().__init__('DCNv2', DCNv2Backbone, item_name, sparse_features, dense_features, **kwargs)
class DeepFMRecommend(NetworkRecommender):
    def __init__(self, item_name, sparse_features, dense_features, **kwargs):
        super().__init__('DeepFM', DeepFMBackbone, item_name, sparse_features, dense_features, **kwargs)
class WideDeepRecommend(NetworkRecommender):
    def __init__(self, item_name, sparse_features, dense_features, **kwargs):
        super().__init__('WideDeep', WideDeepBackbone, item_name, sparse_features, dense_features, **kwargs)
class AutoIntRecommend(NetworkRecommender):
    def __init__(self, item_name, sparse_features, dense_features, **kwargs):
        super().__init__('AutoInt', AutoIntBackbone, item_name, sparse_features, dense_features, **kwargs)
# %%
class CoMICERecommend(BaseRecommender):
    def __init__(self, item_name: str, sparse_features: list, dense_features: list,
                 standard_bool: bool = True, seed: int = 42, k: int = 3, **kwargs):
        super().__init__('CoMICE', item_name, sparse_features, dense_features, standard_bool, seed, k)

        # 将 dropout 集成到 default_params 中
        default_params = {
            'lr': 1e-4,
            'batch_size': 512,
            'feature_dim': 32,
            'proj_dim': 32,
            'epochs': 200,
            'lambda_nce': 1.0,
            'temperature': 0.1,
            'mice_method': 'MICE_RF',
            'cross_layers': 3,
            'hidden_units': [256, 128],
            'use_attention': True,
            'dropout': 0.1  # 新增默认参数
        }
        self.kwargs.update(default_params)
        self.kwargs.update(kwargs)
        self.user_name = sparse_features + dense_features
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        set_seed(self.seed)
    def _build_model(self, sparse_dims, dense_count, num_classes):
        common_args = {
            'sparse_dims': sparse_dims, 'dense_count': dense_count,
            'feature_dim': self.kwargs['feature_dim'],
            'hidden_units': self.kwargs['hidden_units'],
            'dropout': self.kwargs['dropout']  # 传递给 Backbone
        }
        # HybridBackbone 现在会接收并使用 dropout
        backbone = HybridBackbone(cross_layers=self.kwargs['cross_layers'],
                                  use_attention=self.kwargs['use_attention'],
                                  **common_args)

        head = CoMICEHead(backbone.output_dim, num_classes, proj_dim=self.kwargs['proj_dim'])
        return CoMICEModel(backbone, head).to(self.device)
    # ... train_one_epoch, fit, get_proba 方法保持不变 ...