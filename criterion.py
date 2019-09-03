import torch
import torch.nn.functional as F

def L_metric(feat1, feat2, same_class=True):
    '''
        feat1 same size as feat2
        feat size: (batch_size, atts, feat_size)
    '''
    d = torch.sum((feat1 - feat2).pow(2).view((-1, feat1.size(-1))), 1)
    if same_class:
        return d.sum()
    else:
        return torch.clamp(1-d, min=0).sum()

def L_divergence(feats):
    n = feats.shape[0]
    loss = 0
    for i in range(n):
        for j in range(i+1, n):
            loss += torch.clamp(1-torch.sum((feats[i, :] - feats[j, :]).pow(2)), min=0)
    return loss

def loss_func(tensor, batch_k):
        batch_size = tensor.size(0)
        assert batch_size % batch_k == 0
        assert batch_k > 1
        loss_homo, loss_heter, loss_div = 0, 0, 0
        for i in range(batch_size):
                loss_div += L_divergence(tensor[i, ...])

        for group_index in range(batch_size // batch_k):
                for i in range(batch_k):
                        anchor = tensor[i+group_index*batch_k: 1+i+group_index*batch_k, ...]
                        for j in range(i+1, (group_index+1)*batch_k):
                                loss_homo += L_metric(anchor, tensor[j:j+1, ...])
                        for j in range((group_index+1)*batch_k, batch_size):
                                loss_heter += L_metric(anchor, tensor[j:j+1, ...])
        return loss_div/(tensor.size(1)-1), loss_homo/(batch_k-1), loss_heter/(batch_size-batch_k)   

def criterion(anchors, positives, negatives):
        loss_homo = L_metric(anchors, positives)
        loss_heter = L_metric(anchors, negatives, False)
        loss_div = 0
        for i in range(anchors.shape[0]):
                loss_div += (L_divergence(anchors[i, ...]) + L_divergence(positives[i, ...]) + L_divergence(negatives[i, ...])) / 3
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
