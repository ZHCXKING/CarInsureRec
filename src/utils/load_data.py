# %%
import json
import pandas as pd
from pathlib import Path
# %%
def _filter_data(df: pd.DataFrame, target_name: str, min_count=3):
    value_counts = df[target_name].value_counts()
    valid_categories = value_counts[value_counts >= min_count].index
    df = df[df[target_name].isin(valid_categories)].reset_index(drop=True)
    df[target_name], _ = pd.factorize(df[target_name], sort=True)
    return df
# %%
def _split_guaranteed(df: pd.DataFrame, target_name: str, train_ratio: float, val_ratio: float, seed: int = 42):
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    cum_count = df.groupby(target_name).cumcount()
    must_train_mask = cum_count == 0
    must_val_mask = cum_count == 1
    must_test_mask = cum_count == 2
    rest_mask = cum_count > 2
    rest_df = df[rest_mask]
    total_len = len(df)
    n_train_target = int(total_len * train_ratio)
    n_val_target = int(total_len * val_ratio)
    needed_train = max(0, n_train_target - must_train_mask.sum())
    needed_val = max(0, n_val_target - must_val_mask.sum())
    rest_indices = rest_df.index
    idx_train_end = needed_train
    idx_val_end = needed_train + needed_val
    final_train_idx = df[must_train_mask].index.union(rest_indices[:idx_train_end])
    final_val_idx = df[must_val_mask].index.union(rest_indices[idx_train_end:idx_val_end])
    final_test_idx = df[must_test_mask].index.union(rest_indices[idx_val_end:])
    return df.loc[final_train_idx], df.loc[final_val_idx], df.loc[final_test_idx]
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
         train_ratio: float | None = None,
         val_ratio: float = 0.1,
         target_name: str = 'product_item',
         seed: int = 42):
    root = Path(__file__).parents[2]
    data_path = root / 'data' / data_type / 'All Data.parquet'
    detail_path = root / 'data' / data_type / 'Metadata.json'
    data = pd.read_parquet(data_path)
    if amount is not None:
        data = data.head(amount)
    data = data.reset_index(drop=True)
    info = _detail_info(detail_path)
    data = _filter_data(data, target_name, min_count=3)
    if train_ratio is not None:
        if not (0 < train_ratio + val_ratio < 1):
            raise ValueError('The sum of train_ratio and val_ratio must be between 0 and 1.')
        train_data, val_data, test_data = _split_guaranteed(data, target_name, train_ratio, val_ratio, seed)
        return (
            train_data.reset_index(drop=True),
            val_data.reset_index(drop=True),
            test_data.reset_index(drop=True),
            info
        )
    else:
        data = data.reset_index(drop=True)
        return data, info
