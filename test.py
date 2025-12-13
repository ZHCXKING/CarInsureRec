from src.utils import load
from models import BNRecommend, KNNRecommend, LRRecommend, DeepFMRecommend

train, test = load('test', amount=1000, split_num=500)
user_name = ['Age', 'DrivingExp', 'Occupation', 'NCD', 'Make', 'Car.year', 'Car.price']
item_name = 'InsCov'
date_name = 'Date'
sparse_features = ['Occupation', 'NCD', 'Make']
dense_features = ['Age', 'Car.year', 'Car.price', 'DrivingExp']
#model = BNRecommend(user_name, item_name, date_name, sparse_features, dense_features, standard_bool=True, seed=42)
#model = KNNRecommend(user_name, item_name, date_name, sparse_features, dense_features, standard_bool=True, seed=42)
#model = LRRecommend(user_name, item_name, date_name, sparse_features, dense_features, standard_bool=True, seed=42)
model = DeepFMRecommend(user_name, item_name, date_name, sparse_features, dense_features, standard_bool=True, seed=42)
model.fit(train)
# y_prob = model.recommend(test, k=10)
# print(y_prob)
# print(type(y_prob))
# print(y_prob.max(), y_prob.min(), y_prob.nunique())