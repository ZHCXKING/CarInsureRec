# %%
import pandas as pd
import numpy as np
# %%
def mrr_k(item: pd.Series, topk_item: pd.DataFrame, k: int):
    reciprocal_ranks = []
    for index, row in topk_item.iterrows():
        true_item = item[index]
        rr = 0.0
        for rank in range(k):
            if row.iloc[rank] == true_item:
                rr = 1.0 / (rank + 1)
                break
        reciprocal_ranks.append(rr)
    mrr = np.mean(reciprocal_ranks)
    return mrr
# %%
def recall_k(item: pd.Series, topk_item: pd.DataFrame, k: int):
    recalls = []
    for index, row in topk_item.iterrows():
        true_item = item[index]
        if true_item in row.values[:k]:
            recalls.append(1)
        else:
            recalls.append(0)
    return np.mean(recalls)
# %%
def ndcg_k(item: pd.Series, topk_item: pd.DataFrame, k: int):
    ndcg_scores = []
    for index, row in topk_item.iterrows():
        true_item = item[index]
        ndcg = 0.0
        for rank in range(k):
            if row.iloc[rank] == true_item:
                ndcg = 1.0 / np.log2(rank + 2)
                break
        ndcg_scores.append(ndcg)
    return np.mean(ndcg_scores)
