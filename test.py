from src.utils import load
from models import BNRecommend, KNNRecommend, LRRecommend, DeepFMRecommend, WideDeepRecommend, RFRecommend, XGBRecommend, LGBMRecommend, CatBRecommend, MLPRecommend
from src.evaluation import mrr_k, recall_k, ndcg_k

train, test = load('dropna', amount=None, split_num=1000)
user_name = ['Age', 'DrivingExp', 'Occupation', 'NCD', 'Make', 'Car.year', 'Car.price']
item_name = 'InsCov'
date_name = None
sparse_features = ['Occupation', 'NCD', 'Make']
dense_features = ['Age', 'Car.year', 'Car.price', 'DrivingExp']
#model = BNRecommend(user_name, item_name, date_name, sparse_features, dense_features, standard_bool=True, seed=42)
#model = KNNRecommend(user_name, item_name, date_name, sparse_features, dense_features, standard_bool=True, seed=42)
#model = LRRecommend(user_name, item_name, date_name, sparse_features, dense_features, standard_bool=True, seed=42)
model = DeepFMRecommend(user_name, item_name, date_name, sparse_features, dense_features, standard_bool=True, seed=42, epochs=200)
#model = WideDeepRecommend(user_name, item_name, date_name, sparse_features, dense_features, standard_bool=True, seed=42, epochs=200)
#model = RFRecommend(user_name, item_name, date_name, sparse_features, dense_features, standard_bool=True, seed=42)
#model = XGBRecommend(user_name, item_name, date_name, sparse_features, dense_features, standard_bool=True, seed=42)
#model = LGBMRecommend(user_name, item_name, date_name, sparse_features, dense_features, standard_bool=True, seed=42)
#model = CatBRecommend(user_name, item_name, date_name, sparse_features, dense_features, standard_bool=True, seed=42)
#model = MLPRecommend(user_name, item_name, date_name, sparse_features, dense_features, standard_bool=True, seed=42, verbose=True)
model.fit(train)
#y = model.get_proba(test)
#y_item, y_proba = model.get_topk_proba(test, k=5)

score = model.score_test(test, method='mrr', k=5)
print(score)
#print(y_proba)
# print(type(y_prob))
# print(y_prob.max(), y_prob.min(), y_prob.nunique())