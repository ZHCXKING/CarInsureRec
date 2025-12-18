# %%
from .MachineModel import LRRecommend, BNRecommend, KNNRecommend
from .TreeModel import RFRecommend, XGBRecommend, LGBMRecommend, CatBRecommend
from .NetworkModel import DeepFMRecommend, WideDeepRecommend
from .CoMICERecommend import CoMICERecommend

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
    'CoMICERecommend'
]
