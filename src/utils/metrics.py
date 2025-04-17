import torch
import numpy as np

from scipy.stats import pearsonr, combine_pvalues


def pearson_r(x, y):
    """
    As mentioned above ¯\_(ツ)_/¯
    """
    return torch.tensor(pearsonr(x, y)[0])

def masked_pearson_r(x, y, mask):
    """
    Pearson r that accounts for multiple modalities being defined in x/y
    """
    return torch.tensor(pearsonr(x, y)[0])

def pearson_p(x, y):
    """
    As mentioned above ¯\_(ツ)_/¯
    """
    return torch.tensor(pearsonr(x, y)[1])

def masked_pearson_p(x, y, mask):
    """
    P-value that accounts for multiple modalities being defined in x/y
    """
    return torch.tensor(pearsonr(x, y)[1])

def batched_within_pearson_r(x, y, batch_idxs, batch_threshold=5):
    """
    As mentioned above ¯\_(ツ)_/¯
    """
    avg_r = 0.0
    bad_batches = 0
    curr_idx = 0
    for batch_idx in batch_idxs:
        if batch_idx - curr_idx < batch_threshold:
            bad_batches += 1
        else:
            batch_r = pearsonr(x[curr_idx:batch_idx], y[curr_idx:batch_idx])[0]
            if np.isnan(batch_r):
                avg_r += 0.0
            else:
                avg_r += batch_r
        curr_idx = batch_idx

    avg_r = avg_r / (len(batch_idxs) - bad_batches)
    return torch.tensor(avg_r)

def batched_within_pearson_p(x, y, batch_idxs, batch_threshold=5):
    """
    As mentioned above ¯\_(ツ)_/¯
    """
    all_p = []
    curr_idx = 0
    for batch_idx in batch_idxs:
        if batch_idx - curr_idx >= batch_threshold:
            batch_p = pearsonr(x[curr_idx:batch_idx], y[curr_idx:batch_idx])[1]
            if np.isnan(batch_p):
                all_p.append(0.0)
            else:
                all_p.append(batch_p)
        curr_idx = batch_idx

    return torch.tensor(combine_pvalues(all_p)[1])

def batched_between_pearson_r(x, y, batch_idxs, batch_threshold=5):
    """
    As mentioned above ¯\_(ツ)_/¯
    """
    all_avg_x = []
    all_avg_y = []
    curr_idx = 0
    for batch_idx in batch_idxs:
        if batch_idx - curr_idx >= batch_threshold:
            all_avg_x.append(x[curr_idx:batch_idx].mean())
            all_avg_y.append(y[curr_idx:batch_idx].mean())
        curr_idx = batch_idx

    return torch.tensor(pearsonr(all_avg_x, all_avg_y)[0])

def batched_between_pearson_p(x, y, batch_idxs, batch_threshold=5):
    """
    As mentioned above ¯\_(ツ)_/¯
    """
    all_avg_x = []
    all_avg_y = []
    curr_idx = 0
    for batch_idx in batch_idxs:
        if batch_idx - curr_idx >= batch_threshold:
            all_avg_x.append(x[curr_idx:batch_idx].mean())
            all_avg_y.append(y[curr_idx:batch_idx].mean())
        curr_idx = batch_idx

    return torch.tensor(pearsonr(all_avg_x, all_avg_y)[1])