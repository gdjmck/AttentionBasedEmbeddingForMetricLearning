import torch
import torch.nn.functional as F

def L_metric(feat1, feat2, same_class=True):
    if same_class:
        return torch.dist(feat1, feat2).pow(2)
    else:
        return torch.clamp(1-torch.dist(feat1, feat2).pow(2), min=0)

def L_divergence(feats):
    n = feats.shape[0]
    loss = 0
    for i in range(n):
        for j in range(i+1, n):
            loss += torch.clamp(1-torch.sum((feats[i, ...] - feats[j, ...]).pow(2)), min=0)
    return loss

def loss_func(tensor):
        assert tensor.shape[0] % 2 == 0
        batch_split = int(tensor.shape[0] / 2) # idx < batch_split are positive pairs, negative pairs otherwise
        loss_homo, loss_heter, loss_div = 0, 0, 0
        for i in range(0, batch_split, 2):
                loss_div += L_divergence(tensor[i, ...])
                loss_div += L_divergence(tensor[i+1, ...])
                loss_homo += L_metric(tensor[i, ...], tensor[i+1, ...])
        for i in range(batch_split, batch_split*2, 2):
                loss_div += L_divergence(tensor[i, ...])
                loss_div += L_divergence(tensor[i+1, ...])
                loss_heter += L_metric(tensor[i, ...], tensor[i+1, ...], False)
        return loss_div, loss_homo, loss_heter