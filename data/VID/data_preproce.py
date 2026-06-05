# %%
import pandas as pd
import numpy as np
from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer
# %%
feature = ['SEX', 'INSR_END', 'INSR_TYPE', 'INSURED_VALUE', 'PREMIUM', 'PROD_YEAR', 'SEATS_NUM', 'CARRYING_CAPACITY',
           'TYPE_VEHICLE', 'CCM_TON', 'MAKE', 'USAGE', 'CLAIM_PAID']
df1 = pd.read_csv('motor_data11-14lats.csv', usecols=feature)
df2 = pd.read_csv('motor_data14-2018.csv', usecols=feature)
df = pd.concat([df1, df2], ignore_index=True)
df['INSR_TYPE'], _ = pd.factorize(df['INSR_TYPE'], sort=True)
df['INSR_END'] = pd.to_datetime(df['INSR_END'], format='%d-%b-%y')
df['Car.year'] = df['INSR_END'].dt.year - df['PROD_YEAR']
df = df.drop(columns=['INSR_END', 'PROD_YEAR'])
df['product_item'] = (df['USAGE'] + df['TYPE_VEHICLE'])
df['product_item'], _ = pd.factorize(df['product_item'], sort=True)
df = df.drop(columns=['USAGE', 'TYPE_VEHICLE'])
df['MAKE'], _ = pd.factorize(df['MAKE'], sort=True)
df['MAKE'] = df['MAKE'].replace(-1, np.nan)
print(len(df))
print(df['SEX'].max(), df['SEX'].min(), df['SEX'].nunique())
print(df['INSR_TYPE'].max(), df['INSR_TYPE'].min(), df['INSR_TYPE'].nunique())
print(df['MAKE'].max(), df['MAKE'].min(), df['MAKE'].nunique())
print(df['product_item'].max(), df['product_item'].min(), df['product_item'].nunique())
#df.to_parquet('AllData.parquet', index=False)
# %%
missing_percentage = df.isna().sum().sum() / df.size * 100
print(f"Percentage of rows with missing values: {missing_percentage:.2f}%")
df_drop = df.dropna(ignore_index=True)
#df_drop.to_parquet('DropNaData.parquet', index=False)
# %%
# df_sdv = df.sample(n=100000, random_state=42)
# metadata = Metadata.load_from_json(filepath='Metadata.json')
# synthesizer = CTGANSynthesizer(metadata, verbose=True)
# synthesizer.fit(df_sdv)
# synthesizer.save(filepath='synthesizer.pkl')