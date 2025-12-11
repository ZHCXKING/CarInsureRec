#%%
from src.utils.load_data import load
#%%
df = load('synthetic', amount=1000, fillna=True)
print(df)
print(df.info())