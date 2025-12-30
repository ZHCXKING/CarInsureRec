# %%
from src.utils import load, filling, round
from src.models import *
from src.evaluation import *
#%%
def add_missing_values(df: pd.DataFrame, missing_rate: float, feature_cols: list = None, seed: int = 42):
    """
    向 DataFrame 的指定列中随机添加缺失值 (NaN)。
    :param df: 原始 DataFrame
    :param missing_rate: 缺失比例 (0.0 到 1.0)，例如 0.2 表示 20% 的数据设为 NaN
    :param feature_cols: 需要添加缺失值的列名列表。如果为 None，则对所有列操作（不推荐，通常要避开标签列）
    :param seed: 随机种子，保证可复现性
    :return: 含有缺失值的 DataFrame 副本
    """
    df_copy = df.copy()
    # 如果没有指定列，默认对所有列操作
    if feature_cols is None:
        feature_cols = df_copy.columns.tolist()
    # 获取特征数据矩阵
    data = df_copy[feature_cols].values.astype(float)
    n_rows, n_cols = data.shape
    # 设置随机种子
    rng = np.random.default_rng(seed)
    # 生成一个与数据形状相同的随机矩阵 (0~1之间)
    random_matrix = rng.random((n_rows, n_cols))
    # 生成掩码：小于 missing_rate 的位置设为 True (即需要变为 NaN 的位置)
    mask = random_matrix < missing_rate
    # 将掩码对应的位置设为 NaN (注意：原列必须支持浮点或对象类型，int列会自动转为float)
    # 也就是 int 类型在赋值 NaN 后会变成 float 类型
    data[mask] = np.nan
    # 将修改后的数据赋值回 DataFrame
    df_copy[feature_cols] = data
    return df_copy
#%%
k = 3
seed = 42
train, valid, test, info = load('HIP', amount=10000, train_ratio=0.7, val_ratio=0.1, is_dropna=True) #original, dropna
item_name = info['item_name']
sparse_features = info['sparse_features']
dense_features = info['dense_features']
# _, imputer = filling(train, method='MICE_NB', seed=42)
# train_filled = imputer.transform(train)
# valid_filled = imputer.transform(valid)
# test_filled = imputer.transform(test)
# train_filled = round(train_filled, sparse_features)
# valid_filled = round(valid_filled, sparse_features)
# test_filled = round(test_filled, sparse_features)
score = []
# model = DCNRecommend(item_name, sparse_features, dense_features, seed=seed, k=k)
# model.fit(train_filled.copy())
# score.append(model.score_test(test_filled.copy(), methods=['auc']))
# model = CoMICERecommend(item_name, sparse_features, dense_features, seed=seed, k=k, num_views=3, mice_method='MICE_NB')
# model.fit(train.copy())
# score.append(model.score_test(test.copy(), methods=['auc']))
# print(score)
for missing_rate in [0.1, 0.2, 0.3, 0.4, 0.5]:
    train_miss = add_missing_values(train, missing_rate, feature_cols=sparse_features+dense_features, seed=42)
    valid_miss = add_missing_values(valid, missing_rate, feature_cols=sparse_features+dense_features, seed=42)
    test_miss = add_missing_values(test, missing_rate, feature_cols=sparse_features+dense_features, seed=42)
    _, imputer = filling(train_miss, method='MICE_NB', seed=42)
    train_filled = imputer.transform(train_miss.copy())
    valid_filled = imputer.transform(valid_miss.copy())
    test_filled = imputer.transform(test_miss.copy())
    train_filled = round(train_filled, sparse_features)
    valid_filled = round(valid_filled, sparse_features)
    test_filled = round(test_filled, sparse_features)
    model = DCNv2Recommend(item_name, sparse_features, dense_features, seed=seed, k=k)
    model.fit(train_filled)
    score.append(model.score_test(test_filled, methods=['auc']))
    model = CoMICERecommend(item_name, sparse_features, dense_features, seed=42, k=k, num_views=3, mice_method='MICE_NB', backbone='DCNv2')
    model.fit(train_miss)
    score.append(model.score_test(test_miss, methods=['auc']))
    # model = XGBRecommend(item_name, sparse_features, dense_features, seed=seed, k=k)
    # model.fit(train_miss)
    # score.append(model.score_test(test_miss, methods=['auc']))
print(score)