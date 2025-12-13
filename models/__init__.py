# %%
from .base import BaseRecommender
from .BNRecommend import BNRecommend
from .KNNRecommend import KNNRecommend
from .LRRecommend import LRRecommend
from .DeepFMRecommend import DeepFMRecommend
from .WideDeepRecommend import WideDeepRecommend

# %%
__all__ = [
    'BaseRecommender',
    'LRRecommend',
    'BNRecommend',
    'KNNRecommend',
    'DeepFMRecommend',
    'WideDeepRecommend'
]
