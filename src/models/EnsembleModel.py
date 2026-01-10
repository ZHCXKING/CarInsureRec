# %%
import pandas as pd
from .base import BaseRecommender
from .NetworkModel import DCNRecommend, DCNv2Recommend, DeepFMRecommend, WideDeepRecommend, AutoIntRecommend, FiBiNETRecommend
from src.utils import filling, round
import numpy as np
# %%
class EnsembleRecommend(BaseRecommender):
    def __init__(self, item_name: str, sparse_features: list, dense_features: list,
                 standard_bool: bool = True, seed: int = 42, k: int = 3, **kwargs):
        super().__init__('Ensemble', item_name, sparse_features, dense_features, standard_bool, seed, k)
        default_params = {
            'num_views': 3,
            'mice_method': 'MICE_NB',
            'backbone': 'DCN'
        }
        self.kwargs.update(default_params)
        self.kwargs.update(kwargs)
        self.models = []
        self.imputers = []
    def fit(self, train_data: pd.DataFrame, valid_data: pd.DataFrame = None):
        self.out_dim = train_data[self.item_name].nunique()
        self.unique_item = list(range(self.out_dim))
        for i in range(self.kwargs['num_views']):
            current_seed = self.seed + i
            train_filled, imputer = filling(train_data.copy(), method=self.kwargs['mice_method'], seed=current_seed)
            train_filled = round(train_filled, self.sparse_features)
            valid_filled = None
            if valid_data is not None:
                valid_filled = imputer.transform(valid_data.copy())
                valid_filled = round(valid_filled, self.sparse_features)
            ModelClass = globals()[f"{self.kwargs['backbone']}Recommend"]
            model = ModelClass(
                self.item_name, self.sparse_features, self.dense_features,
                standard_bool=self.standard_bool,
                seed=current_seed,  # 模型初始化也要用不同的种子以增加差异性
                k=self.k,
                **self.kwargs
            )
            if valid_filled is not None:
                model.fit(train_filled, valid_filled)
            else:
                model.fit(train_filled)
            self.models.append(model)
            self.imputers.append(imputer)
        self.is_trained = True
    def get_proba(self, test_data: pd.DataFrame):
        if not self.is_trained:
            raise ValueError('Model not trained')
        all_preds = []
        for i, model in enumerate(self.models):
            imputer = self.imputers[i]
            test_filled = imputer.transform(test_data.copy())
            test_filled = round(test_filled, self.sparse_features)
            pred_df = model.get_proba(test_filled)
            all_preds.append(pred_df.values)
        avg_pred = np.mean(all_preds, axis=0)
        return pd.DataFrame(avg_pred, index=test_data.index, columns=self.unique_item)
# %%
class AugmentRecommend(BaseRecommender):
    def __init__(self, item_name: str, sparse_features: list, dense_features: list,
                 standard_bool: bool = True, seed: int = 42, k: int = 3, **kwargs):
        super().__init__('Augmented', item_name, sparse_features, dense_features, standard_bool, seed, k)
        default_params = {
            'num_views': 3,
            'mice_method': 'MICE_NB',
            'backbone': 'DCN'
        }
        self.run_params = default_params.copy()
        self.run_params.update(kwargs)
        self.kwargs.update(self.run_params)
        self.imputers = []
        self.model = None
    def fit(self, train_data, valid_data=None):
        self.out_dim = train_data[self.item_name].nunique()
        augmented_train_dfs = []
        num_views = self.run_params['num_views']
        mice_method = self.run_params['mice_method']
        self.imputers = []
        print(f"Augmenting training data with {num_views} views")
        for i in range(num_views):
            current_seed = self.seed + i
            df_filled, imputer = filling(train_data.copy(), method=mice_method, seed=current_seed)
            df_filled = round(df_filled, self.sparse_features)
            augmented_train_dfs.append(df_filled)
            self.imputers.append(imputer)
        final_train_data = pd.concat(augmented_train_dfs, axis=0).reset_index(drop=True)
        if valid_data is not None:
            valid_filled = self.imputers[0].transform(valid_data.copy())
            valid_filled = round(valid_filled, self.sparse_features)
        else:
            valid_filled = None
        backbone_name = self.run_params['backbone']
        ModelClass = globals()[f"{backbone_name}Recommend"]
        self.model = ModelClass(
            self.item_name, self.sparse_features, self.dense_features,
            seed=self.seed, k=self.k, **self.run_params
        )
        if valid_filled is not None:
            self.model.fit(final_train_data, valid_filled)
        else:
            self.model.fit(final_train_data)
        self.unique_item = self.model.unique_item
        self.is_trained = True
    def get_proba(self, test_data):
        if not self.is_trained:
            raise ValueError('Model not trained')
        imputer = self.imputers[0]
        test_filled = imputer.transform(test_data.copy())
        test_filled = round(test_filled, self.sparse_features)
        return self.model.get_proba(test_filled)