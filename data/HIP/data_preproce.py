# %%
import pandas as pd
import numpy as np
from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer
# %%
Data_feature = ['gender', 'distribution_channel', 'C_C', 'C_H', 'age', 'premium', 'cost_claims_year', 'seniority_insured',
           'seniority_policy', 'exposure_time', 'n_medical_services', 'type_product', 'type_policy_dg', 'reimbursement']
feature = ['Gndr', 'Chan', 'Clim', 'Habt', 'Age', 'Prem', 'Clms', 'SnrU', 'SnrP', 'Expo', 'MedS', 'type_product', 'type_policy_dg', 'reimbursement']
df = pd.read_excel('Dataset of health insurance portfolio.xlsx', usecols=Data_feature).rename(columns=dict(zip(Data_feature, feature)))
df['product_item'] = (df['type_policy_dg'] + df['type_product'] + df['reimbursement'] + df['Chan'])
df['product_item'], _ = pd.factorize(df['product_item'], sort=True)
df = df.drop(columns=['type_policy_dg', 'type_product', 'reimbursement', 'Chan'])
df['Gndr'], _ = pd.factorize(df['Gndr'], sort=True)
df['Habt'], _ = pd.factorize(df['Habt'], sort=True)
df['Habt'] = df['Habt'].replace(-1, np.nan)
df['Clim'], _ = pd.factorize(df['Clim'], sort=True)
df['Clim'] = df['Clim'].replace(-1, np.nan)
print(len(df))
print(df['Gndr'].max(), df['Gndr'].min(), df['Gndr'].nunique())
print(df['Habt'].max(), df['Habt'].min(), df['Habt'].nunique())
print(df['Clim'].max(), df['Clim'].min(), df['Clim'].nunique())
print(df['product_item'].max(), df['product_item'].min(), df['product_item'].nunique())
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