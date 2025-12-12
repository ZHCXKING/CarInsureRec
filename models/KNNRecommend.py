#%%
import pandas as pd
from models import BaseRecommender
from sklearn.neighbors import KNeighborsClassifier
#https://scikit-learn.org.cn/view/695.html
#%%
class KNNRecommend(BaseRecommender):
    def __init__(self, user_name:list, item_name:str, date_name:str|None=None, seed:int=42, **kwargs):
        if (user_name is None) or (item_name is None):
            raise ValueError('user_name and item_name are required')
        super().__init__('KNN', user_name, item_name, date_name, seed)
        self.kwargs = kwargs
        self.model = KNeighborsClassifier(**kwargs)
    #%%
    def fit(self, train_data: pd.DataFrame):
        X = train_data[self.user_name]
        y = train_data[self.item_name]
        self.unique_item = y.unique().tolist()
        self.model.fit(X, y)
        self.is_trained = True
    #%%
    def recommend(self, test_data:pd.DataFrame, k:int=5):
        if not self.is_trained:
            raise ValueError('model is not trained')
        X = test_data[self.user_name]
        y = self.model.predict_proba(X)
        result = pd.DataFrame(y, index=test_data.index, columns=self.unique_item)
        topk_item = result.apply(lambda row: row.nlargest(k).index.tolist(), axis=1)
        topk_item = pd.DataFrame(topk_item.tolist(), columns=[f'top{i + 1}' for i in range(k)])
        return topk_item
#%%
if __name__ == '__main__':
    from src.utils import load
    from models import BNRecommend
    train, test = load('test', amount=1000, split_num=500)
    user_name = ['Age', 'DrivingExp', 'Occupation', 'NCD', 'Make', 'Car.year', 'Car.price']
    item_name = 'InsCov'
    date_name = 'Date'
    sparse_features = ['Occupation', 'NCD', 'Make']
    dense_features = ['Age', 'Car.year', 'Car.price', 'DrivingExp']
    model = BNRecommend(user_name, item_name, date_name, seed=42)
    model.fit(train)
    y_prob = model.recommend(test, k=10)
    # print(y_prob)
    # print(type(y_prob))
    # print(y_prob.max(), y_prob.min(), y_prob.nunique())