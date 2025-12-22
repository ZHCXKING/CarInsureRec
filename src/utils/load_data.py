# %%
import json
import pandas as pd
from pathlib import Path
# %%
def _filter_data(df: pd.DataFrame, target_name: str, min_count=2):
    value_counts = df[target_name].value_counts()
    valid_categories = value_counts[value_counts > min_count].index
    df = df[df[target_name].isin(valid_categories)].reset_index(drop=True)
    df[target_name], _ = pd.factorize(df[target_name], sort=True)
    return df
# %%
def _split_guaranteed(df: pd.DataFrame, target_name: str, split_num: int, seed: int = 42):
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    cum_count = df.groupby(target_name).cumcount()
    train_mask = cum_count == 0
    test_mask = cum_count == 1
    rest_mask = cum_count > 1
    n_needed = split_num - train_mask.sum()
    if n_needed < 0:
        raise ValueError(f"split_num is too small")
    rest_indices = df[rest_mask].index
    final_train_idx = df[train_mask].index.union(rest_indices[:n_needed])
    final_test_idx = df[test_mask].index.union(rest_indices[n_needed:])
    return df.loc[final_train_idx], df.loc[final_test_idx]
# %%
def _detail_info(path):
    with open(path, 'r') as f:
        metadata = json.load(f)
    columns = metadata['tables']['table']['columns']
    item_name = 'product_item'
    sparse_features = []
    dense_features = []
    for col, info in columns.items():
        if col == item_name:
            continue
        if info['sdtype'] == 'categorical':
            sparse_features.append(col)
        elif info['sdtype'] == 'numerical':
            dense_features.append(col)
    return {
        'item_name': item_name,
        'sparse_features': sparse_features,
        'dense_features': dense_features
    }
# %%
def load(data_type: str = 'AWM',
         amount: int | None = None,
         split_num: int | None = None,
         target_name: str = 'product_item',
         seed: int = 42):
    root = Path(__file__).parents[2]
    data_path = root / 'data' / data_type / 'All Data.xlsx'
    detail_path = root / 'data' / data_type / 'Metadata.json'
    data = pd.read_excel(data_path, sheet_name='coding', nrows=amount)
    info = _detail_info(detail_path)
    data = _filter_data(data, target_name)
    if split_num is not None:
        if split_num > data.shape[0]:
            raise ValueError('split_num must be smaller than data amount.')
        train_data, test_data = _split_guaranteed(data, target_name, split_num, seed)
        train_data = train_data.reset_index(drop=True)
        test_data = test_data.reset_index(drop=True)
        return train_data, test_data, info
    else:
        data = data.reset_index(drop=True)
        return data, info
