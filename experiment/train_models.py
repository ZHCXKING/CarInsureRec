# %%
import json
from src.models import *
from src.utils import load, get_filled_data
from pathlib import Path
# %%
root = Path(__file__).parents[0]
datasets = ['AWM', 'HIP', 'VID']
NN_MODELS = ['DCN', 'DCNv2', 'DeepFM', 'WideDeep', 'FiBiNET', 'AutoInt']
TREE_MODELS = ['RF', 'XGB', 'LGBM', 'CatB']
STATISTIC_MODELS = ['LR', 'KNN', 'NB']
seed = 42
amount = None
train_ratio = 0.7
val_ratio = 0.1
def original_train():
    for data_type in datasets:
        train, valid, test, info = load(data_type, amount, train_ratio, val_ratio, is_dropna=False)
        train_filled, valid_filled, test_filled = get_filled_data(train, valid, test, info['sparse_features'])
        for model_name in NN_MODELS + TREE_MODELS + STATISTIC_MODELS:
            param_file = root / data_type / 'params' / (model_name + ".json")
            model_file = root / data_type / 'Original_model' / (model_name + ".pth")
            with open(param_file, 'r') as f:
                params = json.load(f)
            ModelClass = globals()[f"{model_name}Recommend"]
            model = ModelClass(info['item_name'], info['sparse_features'], info['dense_features'], seed=seed, k=3, **params)
            if model_name in NN_MODELS:
                model.fit(train_filled.copy(), valid_filled.copy())
            elif model_name in TREE_MODELS:
                model.fit(train.copy())
            elif model_name in STATISTIC_MODELS:
                model.fit(train_filled.copy())
            model.save(model_file)
            if model_name in NN_MODELS:
                CoMICE_file = root / data_type / 'Original_model' / ('CoMICE_' + model_name + ".pth")
                comice_model = CoMICERecommend(info['item_name'], info['sparse_features'], info['dense_features'], seed=seed, k=3, backbone=model_name, **params)
                comice_model.fit(train.copy(), valid.copy())
                comice_model.save(CoMICE_file)
if __name__ == "__main__":
    original_train()