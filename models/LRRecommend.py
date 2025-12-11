#%%
import pandas as pd
from models import BaseRecommender
#%%
class LRRecommend(BaseRecommender):
    def __init__(self, item_name):
        super().__init__(model_name='LR', item_name=item_name)
    #%%
    def fit(self, train_data: pd.DataFrame):
        pass