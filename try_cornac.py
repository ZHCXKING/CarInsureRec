#%%
import cornac
import numpy as np
from src.utils.load_data import load_data
from cornac.data import Dataset, FeatureModality
from cornac.eval_methods import RatioSplit
from cornac.models import BiVAECF, ItemKNN
from cornac.metrics import Precision, Recall
#%%
df = load_data('test')
#%%
user_ids = df.index.tolist()  #list[int]类型
item_ids = df['InsCov']  #Series[int]类型
ratings = np.ones(len(df))  #array[np.float]类型
timestamps = df['Date'].apply(lambda x: x.timestamp())  #Series[float]类型
#%%
uirt_data = list(zip(user_ids, item_ids, ratings, timestamps))
#%%
feature_cols = ['Age', 'DrivingExp', 'Occupation', 'NCD', 'Make', 'Car.year', 'Car.price']
features = df[feature_cols].values
user_features = FeatureModality(features=features, ids=user_ids)
#%%
rs = RatioSplit(uirt_data, fmt='UIRT', test_size=0.2, rating_threshold=1.0, verbose=True, user_feature=user_features, exclude_unknowns=False)
bivaecf = BiVAECF(likelihood='pois', cap_priors={'user': True, 'item': False}, verbose=True)
itemknn = ItemKNN()
metrice = [Precision(k=10), Recall(k=10)]
#%%
cornac.Experiment(eval_method=rs, models=[bivaecf, itemknn], metrics=metrice).run()
