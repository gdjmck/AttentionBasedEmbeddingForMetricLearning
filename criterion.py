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

def criterion(anchors, positives, negatives):
        loss_homo = L_metric(anchors, positives)
        loss_heter = L_metric(anchors, negatives, False)
        loss_div = 0
        print('\tAnchor:', anchors.shape)
        for i in range(anchors.shape[0]):
                loss_div += L_divergence(anchors[i, ...]) + L_divergence(positives[i, ...]) + L_divergence(negatives[i, ...])
        return loss_div, loss_homo, loss_heter
        
def cluster_centroid_loss(cluster_a, cluster_b, margin=1):
        '''
                cluster_a and cluster_b are two batch of data drawn from two different class label
                we want the distance of all samples from one class are nearer to the centroid of its class than the other class by a margin
                Larger the size of cluster the better
        '''
        centroid_a = torch.mean(cluster_a, 0, keepdim=True)
        centroid_b = torch.mean(cluster_b, 0, keepdim=True)

        loss_a = torch.clamp(
                (cluster_a - centroid_a) **2 - (cluster_a - centroid_b) ** 2 + margin,
                min=0.
        )
        loss_b = torch.clamp(
                (cluster_b - centroid_b) ** 2 - (cluster_b - centroid_a) **2 + margin,
                min=0.
        )
        return loss_a + loss_b