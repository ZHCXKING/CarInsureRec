# import pandas as pd
# from sklearn.model_selection import train_test_split
# df = pd.read_excel('data/All Data.xlsx', sheet_name='coding')
# train, test = train_test_split(df, train_size=0.8, stratify=df['InsCov'], random_state=42, shuffle=True)
# train = train.reset_index(drop=True)
# test =  test.reset_index(drop=True)
# print(train)
# print(test)
from src.utils import load
df1, df2 = load('original', amount=1000, split_num=500, fillna=True)
print(df1.info())
print(df2.info())