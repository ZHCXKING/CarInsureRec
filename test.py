from src.utils import load
from models import BNRecommend, KNNRecommend, LRRecommend, DeepFMRecommend, WideDeepRecommend, RFRecommend, XGBRecommend, LGBMRecommend, CatBRecommend
from src.Evaluation import mrr_k, recall_k, ndcg_k

train, test = load('test', amount=1000, split_num=500, fillna=True)
user_name = ['Age', 'DrivingExp', 'Occupation', 'NCD', 'Make', 'Car.year', 'Car.price']
item_name = 'InsCov'
date_name = 'Date'
sparse_features = ['Occupation', 'NCD', 'Make']
dense_features = ['Age', 'Car.year', 'Car.price', 'DrivingExp']
#model = BNRecommend(user_name, item_name, date_name, sparse_features, dense_features, standard_bool=True, seed=42)
#model = KNNRecommend(user_name, item_name, date_name, sparse_features, dense_features, standard_bool=True, seed=42)
#model = LRRecommend(user_name, item_name, date_name, sparse_features, dense_features, standard_bool=True, seed=42)
#model = DeepFMRecommend(user_name, item_name, date_name, sparse_features, dense_features, standard_bool=True, seed=42, epochs=200)
#model = WideDeepRecommend(user_name, item_name, date_name, sparse_features, dense_features, standard_bool=True, seed=42, epochs=200)
#model = RFRecommend(user_name, item_name, date_name, sparse_features, dense_features, standard_bool=True, seed=42)
model = XGBRecommend(user_name, item_name, date_name, sparse_features, dense_features, standard_bool=True, seed=42)
#model = LGBMRecommend(user_name, item_name, date_name, sparse_features, dense_features, standard_bool=True, seed=42)
#model = CatBRecommend(user_name, item_name, date_name, sparse_features, dense_features, standard_bool=True, seed=42)
model.fit(train)
#y = model.get_proba(test)
#y_item, y_proba = model.get_topk_proba(test, k=5)
y_true = test[item_name] #Series
y_item = model.recommend(test, k=5) #DataFrame
print(y_true)
print(y_item)
#score = mrr_k(y_true, y_item, k=5)
#score = recall_k(y_true, y_item, k=5)
score = ndcg_k(y_true, y_item, k=5)
print(score)
#print(y_proba)
# print(type(y_prob))
# print(y_prob.max(), y_prob.min(), y_prob.nunique())