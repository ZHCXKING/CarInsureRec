#%%
import pandas as pd
from sdv.utils import load_synthesizer
from src.utils.fillna import filling, split_filling
from sklearn.model_selection import train_test_split
from pathlib import Path
#%%
def _filter_data(df: pd.DataFrame, min_count=2):
    value_counts = df['InsCov'].value_counts()
    valid_categories = value_counts[value_counts > min_count].index
    df = df[df['InsCov'].isin(valid_categories)].reset_index(drop=True)
    df['InsCov'], _ = pd.factorize(df['InsCov'], sort=True)
    return df
#%%
def load(data_type: str='test',
         amount: int|None=None,
         split_num: int|None=None,
         fillna: bool=False,
         fillna_method: str='iterative_NB',
         seed: int=42):
    root = Path(__file__).parents[2]
    if data_type == 'original':
        data = pd.read_excel(root/'data'/'All Data.xlsx', sheet_name='coding', nrows=amount)
    elif data_type == 'test':
        data = pd.read_excel(root/'data'/'All Data.xlsx', sheet_name='filling', nrows=amount)
    elif data_type == 'synthetic':
        if amount is None:
            raise ValueError('When loading synthetic data, amount cannot be None.')
        synthesizer = load_synthesizer(filepath=root/'data'/'synthesizer.pkl')
        data = synthesizer.sample(num_rows=amount)
    else:
        raise ValueError('data_type must be original, synthetic and test.')
    data = _filter_data(data)
    if split_num is not None:
        if split_num > data.shape[0]:
            raise ValueError('split_num must be smaller than data amount.')
        train_data, test_data = train_test_split(data, train_size=split_num, stratify=data['InsCov'], random_state=seed)
        if fillna:
            train_data, test_data = split_filling(train_data, test_data, method=fillna_method)
        return train_data, test_data
    else:
        if fillna:
            data = filling(data, method=fillna_method)
        return data