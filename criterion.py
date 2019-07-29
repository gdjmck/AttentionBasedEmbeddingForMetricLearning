import torch
import torch.nn.functional as F

def L_metric(feat1, feat2, same_class=True):
    if same_class:
        return F.mse_loss(feat1, feat2)
    else:
        return max(0, 1-F.mse_loss(feat1, feat2))

def L_divergence(feats):
    n = len(feats)
    loss = 0
    for i in range(n):
        for j in range(i+1, n):
            loss += max(0, 1-F.mse_loss(feats[i], feats[j]))
    return loss