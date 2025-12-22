# %%
import pandas as pd
from sdv.utils import load_synthesizer
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
def load(data_type: str = 'test',
         amount: int | None = None,
         split_num: int | None = None,
         target_name: str = 'InsCov',
         seed: int = 42):
    root = Path(__file__).parents[2]
    if data_type == 'original':
        data = pd.read_excel(root / 'data' / 'AWM' /'All Data.xlsx', sheet_name='coding', nrows=amount)
    elif data_type == 'dropna':
        data = pd.read_excel(root / 'data' / 'AWM' / 'All Data.xlsx', sheet_name='dropping', nrows=amount)
    elif data_type == 'synthetic':
        if amount is None:
            raise ValueError('When loading synthetic data, amount cannot be None.')
        synthesizer = load_synthesizer(filepath=root / 'data' / 'synthesizer.pkl')
        data = synthesizer.sample(num_rows=amount)
    else:
        raise ValueError('data_type must be original, synthetic and dropna.')
    data = _filter_data(data, target_name)
    if split_num is not None:
        if split_num > data.shape[0]:
            raise ValueError('split_num must be smaller than data amount.')
        train_data, test_data = _split_guaranteed(data, target_name, split_num, seed)
        train_data = train_data.reset_index(drop=True)
        test_data = test_data.reset_index(drop=True)
        return train_data, test_data
    else:
        data = data.reset_index(drop=True)
        return data
