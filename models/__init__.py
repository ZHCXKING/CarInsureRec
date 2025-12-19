# %%
from .MachineModel import LRRecommend, BNRecommend, KNNRecommend
from .TreeModel import RFRecommend, XGBRecommend, LGBMRecommend, CatBRecommend
from .NetworkModel import DeepFMRecommend, WideDeepRecommend, DCNv2Recommend
from .CoMICERecommend import CoMICERecommend

# %%
__all__ = [
    'LRRecommend',
    'BNRecommend',
    'KNNRecommend',
    'DeepFMRecommend',
    'WideDeepRecommend',
    'DCNv2Recommend',
    'RFRecommend',
    'XGBRecommend',
    'LGBMRecommend',
    'CatBRecommend',
    'CoMICERecommend'
]
