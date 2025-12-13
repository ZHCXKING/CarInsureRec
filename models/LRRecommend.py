#%%
import pandas as pd
from models import BaseRecommender
from sklearn.linear_model import LogisticRegression
#https://scikit-learn.org.cn/view/378.html
#%%
class LRRecommend(BaseRecommender):
    def __init__(self, user_name: list, item_name: str, date_name: str | None = None,
                 sparse_features: list | None = None, dense_features: list | None = None, standard_bool: bool = False,
                 seed: int = 42, **kwargs):
        if (user_name is None) or (item_name is None):
            raise ValueError('user_name and item_name are required')
        super().__init__('LR', user_name, item_name, date_name, sparse_features, dense_features, standard_bool, seed)
        default_params = {
            'C': 10,
            'penalty': 'l2',
            'solver': 'lbfgs',
            'class_weight': 'balanced',
            'max_iter': 1000,
            'random_state': self.seed
        }
        model_params = ['C', 'penalty', 'solver', 'class_weight', 'max_iter', 'random_state']
        self.kwargs.update(default_params)
        self.kwargs.update(kwargs)
        model_params = {key: self.kwargs[key] for key in model_params}
        self.model = LogisticRegression(**model_params)
    #%%
    def fit(self, train_data: pd.DataFrame):
        X = train_data[self.user_name]
        if self.standard_bool:
            X = self._Standardize(X, fit_bool=True)
        y = train_data[self.item_name]
        self.unique_item = y.unique().tolist()
        self.model.fit(X, y)
        self.is_trained = True
    #%%
    def recommend(self, test_data: pd.DataFrame, k:int=5):
        if not self.is_trained:
            raise ValueError('model is not trained')
        X = test_data[self.user_name]
        if self.standard_bool:
            X = self._Standardize(X, fit_bool=False)
        y = self.model.predict_proba(X)
        result = pd.DataFrame(y, index=test_data.index, columns=self.unique_item)
        topk_item = result.apply(lambda row: row.nlargest(k).index.tolist(), axis=1)
        topk_item = pd.DataFrame(topk_item.tolist(), columns=[f'top{i+1}' for i in range(k)])
        return topk_item
#%%
if __name__ == '__main__':
    from src.utils import load
    from models import LRRecommend
    train, test = load('test', amount=10000, split_num=5000)
    user_name = ['Age', 'DrivingExp', 'Occupation', 'NCD', 'Make', 'Car.year', 'Car.price']
    item_name = 'InsCov'
    date_name = 'Date'
    sparse_features = ['Occupation', 'NCD', 'Make']
    dense_features = ['Age', 'Car.year', 'Car.price', 'DrivingExp']
    model = LRRecommend(user_name, item_name, date_name, sparse_features, dense_features, standard_bool=True, seed=42)
    model.fit(train)
    y_prob = model.recommend(test, k=10)
    print(y_prob)
    # print(type(y_prob))
    # print(y_prob.max(), y_prob.min(), y_prob.nunique())