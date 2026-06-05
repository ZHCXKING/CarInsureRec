# %%
import pandas as pd
import numpy as np
from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer
# %%
Data1_feature = ['Age', 'Driving exp', 'Occupation', 'Coverage', 'Insurer', 'NCD', 'Settlement Date', 'Make', 'Year of manufacturer', 'Sum of insured']
Data2_feature = ['Age', 'DE', 'Occupation', 'Coverage', 'Insur Co', 'NCD', 'Settlement Date', 'Make', 'Man. Year', 'Sum Insured']
feature = ['Age', 'DrivingExp', 'Occupation', 'Coverage', 'Insurer', 'NCD', 'Date', 'Make', 'Car.year', 'Car.price']
df1 = pd.read_excel('Data1.xlsx', usecols=Data1_feature).rename(columns=dict(zip(Data1_feature, feature)))
df2 = pd.read_excel('Data2.xlsx', usecols=Data2_feature).rename(columns=dict(zip(Data2_feature, feature)))
df = pd.concat([df1, df2], ignore_index=True)
df.dropna(subset=['Insurer'], inplace=True, ignore_index=True)
df.drop_duplicates(inplace=True, ignore_index=True)
# %%
df['Age'] = df['Age'].replace({
    '70歲以上': '71',
    '70+': '71'
})
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df['DrivingExp'] = df['DrivingExp'].replace({
    '多於 10 年': '11',
    '暫准駕駛執照 ( P牌 ）': '0',
    '+10 years': '11',
    'Option One': np.nan,
    '>': np.nan
})
df['DrivingExp'] = df['DrivingExp'].str.replace(r'[^0-9]', '', regex=True)
df['DrivingExp'] = pd.to_numeric(df['DrivingExp'], errors='coerce')
df['Occupation'] = df['Occupation'].str.replace(r'\|.*', '', regex=True)
df['Occupation'] = df['Occupation'].str.replace(r'.*? ([\u4e00-\u9fa5].*)', r'\1', regex=True)
df['Occupation'] = df['Occupation'].replace({
    'clekr': '文員',
    'Option One': np.nan
})
df['Occupation'] = df['Occupation'].str.strip()
df['Occupation'], Occupation_index = pd.factorize(df['Occupation'], sort=True)
df['Occupation'] = df['Occupation'].replace(-1, np.nan)
df['product_item'] = np.where(
    df['Insurer'].isna() & df['Coverage'].isna(),
    np.nan,
    df['Insurer'].fillna('Missing').astype(str) + '|' + df['Coverage'].fillna('Missing').astype(str)
)
value_counts = df['product_item'].value_counts()
valid_categories = value_counts[value_counts >= 2].index
df = df[df['product_item'].isin(valid_categories)].reset_index(drop=True)
df['product_item'], _ = pd.factorize(df['product_item'], sort=True)
df = df.drop(columns=['Insurer', 'Coverage'])
df['NCD'] = pd.to_numeric(df['NCD'], errors='coerce')
df['NCD'], _ = pd.factorize(df['NCD'], sort=True)
df['NCD'] = df['NCD'].replace(-1, np.nan)
df['Make'] = df['Make'].astype(str).str.strip().str.replace('\ufeff', '', regex=True).str.upper()
invalid_brands = ['---------------', '--------------------', '_______________________', '----------', 'nan']
df['Make'] = df['Make'].replace(invalid_brands, np.nan)
df['Make'] = df['Make'].replace({'MITSUBISHI FUSO': 'MAYBACH'})
df['Make'], Make_index = pd.factorize(df['Make'], sort=True)
df['Make'] = df['Make'].replace(-1, np.nan)
df['Car.year'] = df['Car.year'].astype(str).str.strip()
df['Car.year'] = df['Car.year'].replace({
    '1980 或之前': '1980',
    'VOXY': np.nan,
    'MODEL S': np.nan,
})
df['Car.year'] = pd.to_numeric(df['Car.year'], errors='coerce')
df['Car.year'] = df['Date'].dt.year - df['Car.year']
df.loc[df['Car.year'] < 0, 'Car.year'] = np.nan
df.sort_values(by='Date', inplace=True, ignore_index=True)
df = df.drop(columns='Date')
print(len(df))
print(df['Occupation'].max(), df['Occupation'].min(), df['Occupation'].nunique())
print(df['NCD'].max(), df['NCD'].min(), df['NCD'].nunique())
print(df['product_item'].max(), df['product_item'].min(), df['product_item'].nunique())
print(df.columns)
#df.to_parquet('AllData.parquet', index=False)
# %%
missing_percentage = df.isna().sum().sum() / df.size * 100
print(f"Percentage of rows with missing values: {missing_percentage:.2f}%")
df_drop = df.dropna(ignore_index=True)
#df_drop.to_parquet('DropNaData.parquet', index=False)
# %%
# df_sdv = df.copy()
# metadata = Metadata.load_from_json(filepath='Metadata.json')
# synthesizer = CTGANSynthesizer(metadata, verbose=True)
# synthesizer.fit(df_sdv)
# synthesizer.save(filepath='synthesizer.pkl')
