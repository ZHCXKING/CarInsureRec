# %%
from .BNRecommend import BNRecommend
from .KNNRecommend import KNNRecommend
from .LRRecommend import LRRecommend
from .DeepFMRecommend import DeepFMRecommend
from .WideDeepRecommend import WideDeepRecommend
from .RFRecommend import RFRecommend
from .XGBRecommend import XGBRecommend
from .LGBMRecommend import LGBMRecommend
from .CatBRecommend import CatBRecommend
from .MLPRecommend import MLPRecommend

# %%
__all__ = [
    'LRRecommend',
    'BNRecommend',
    'KNNRecommend',
    'DeepFMRecommend',
    'WideDeepRecommend',
    'RFRecommend',
    'XGBRecommend',
    'LGBMRecommend',
    'CatBRecommend',
    'MLPRecommend'
]
