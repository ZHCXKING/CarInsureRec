from src.utils import load
from models import LRRecommend, BNRecommend, KNNRecommend
train, test = load('original', amount=10000, split_num=5000, fillna=True, seed=42)
user_name = ['Age', 'DrivingExp', 'Occupation', 'NCD', 'Make', 'Car.year', 'Car.price']
item_name = 'InsCov'
date_name = 'Date'

model = LRRecommend(user_name, item_name, date_name, seed=42)
model.fit(train)
y_prob = model.recommend(test, k=10)
print(y_prob)
print(type(y_prob))
print(y_prob.max(), y_prob.min(), y_prob.nunique())

model = BNRecommend(user_name, item_name, date_name, seed=42)
model.fit(train)
y_prob = model.recommend(test, k=10)
print(y_prob)
print(type(y_prob))
print(y_prob.max(), y_prob.min(), y_prob.nunique())

model = KNNRecommend(user_name, item_name, date_name, seed=42)
model.fit(train)
y_prob = model.recommend(test, k=10)
print(y_prob)
print(type(y_prob))
print(y_prob.max(), y_prob.min(), y_prob.nunique())