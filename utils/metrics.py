import numpy as np
from scipy.stats import rankdata

def cal_ranks(scores, labels, filters):
    # scores: (batch_size, n_ent), labels: (batch_size, n_ent) binary
    scores = scores - np.min(scores, axis=1, keepdims=True) + 1e-8
    full_rank = rankdata(-scores, method='average', axis=1)
    filter_scores = scores * filters
    filter_rank = rankdata(-filter_scores, method='min', axis=1)
    ranks = (full_rank - filter_rank + 1) * labels
    ranks = ranks[np.nonzero(ranks)]
    return list(ranks)

def cal_performance(ranks):
    if len(ranks) == 0:
        return 0, 0, 0, 0
    ranks = np.array(ranks)
    mrr = (1. / ranks).sum() / len(ranks)
    h_1 = np.mean(ranks <= 1)
    h_3 = np.mean(ranks <= 3)
    h_10 = np.mean(ranks <= 10)
    return mrr, h_1, h_3, h_10