#%%
import pandas as pd
from sdv.utils import load_synthesizer
from utils.fillna import iterative, split_iterative
#%%
def load_data(data_type: str='synthetic', amount: int|None=None, split_num: int|None=None, fillna: bool=False):
    if data_type == 'original':
        data = pd.read_excel('../data/All Data.xlsx', sheet_name='coding', nrows=amount)
    elif data_type == 'test':
        data = pd.read_excel('../data/All Data.xlsx', sheet_name='filling', nrows=amount)
    elif data_type == 'synthetic':
        if amount is None:
            raise ValueError('When loading synthetic data, amount cannot be None.')
        synthesizer = load_synthesizer(filepath='../data/synthesizer.pkl')
        data = synthesizer.sample(num_rows=amount)
    else:
        raise ValueError('data_type must be original, synthetic and test.')
    if split_num is not None:
        if split_num > data.shape[0]:
            raise ValueError('split_num must be smaller than data amount.)')
        if fillna:
            data_train, data_test = split_iterative(data, split_num)
            return data_train, data_test
        else:
            data_train = data[:split_num].reset_index(drop=True)
            data_test = data[split_num:].reset_index(drop=True)
            return data_train, data_test
    else:
        if fillna:
            data = iterative(data, method='NB')
            return data
        else:
            return data