import torch
import torch.nn as nn
import torch.nn.functional as F

eps = 1e-4
m_c = 1 # threshold for different class metric

def L2_squared(x1, x2):
    '''
        both x1, x2 of shape: (b, n)
    '''
    return (x1-x2).pow(2).sum(1)

def L_metric(feat1, feat2, same_class=True):
    '''
        feat1 same size as feat2
        feat size: (batch_size, atts, feat_size)
    '''
    b, a, f = feat1.size()
    feat1 = feat1.view((b*a, f))
    feat2 = feat2.view((b*a, f))
    
    d = L2_squared(feat1, feat2)
    if same_class:
        return d.sum()
    else:
        # 因为在测试集上不同class的metric的L2距离也小于1
        d_heter = torch.clamp(m_c-d, min=0)
        feat1_length = torch.clamp(feat1.pow(2).sum(1) - 1, min=0)
        feat2_length = torch.clamp(feat2.pow(2).sum(1) - 1, min=0)

        return ((1 + feat1_length * feat2_length) * d_heter).sum()

def L_divergence(feats):
    '''
        feats: metric of same image with different attention
        feats size: (atts, n)
    '''
    n = feats.shape[0]
    loss = 0
    cnt = 0
    for i in range(n):
        for j in range(i+1, n):
            l_div = torch.clamp(1-L2_squared(feats[i:i+1, :], feats[j:j+1, :]), min=0)
            loss += l_div
            cnt += 1
    # no average op refer to the paper
    return loss

def loss_func(tensor, batch_k):
        batch_size = tensor.size(0)
        assert batch_size % batch_k == 0
        assert batch_k > 1
        loss_homo, loss_heter, loss_div = tensor.new(1).zero_(), tensor.new(1).zero_(), tensor.new(1).zero_()
        # divergence loss
        for i in range(batch_size):
                loss_div += L_divergence(tensor[i, ...])

        cnt_homo, cnt_heter = 0, 0
        for group_index in range(batch_size // batch_k):
                for i in range(batch_k):
                        anchor = tensor[i+group_index*batch_k: 1+i+group_index*batch_k, ...]
                        for j in range(i+1, batch_k):
                                index = j+group_index*batch_k
                                loss_homo += L_metric(anchor, tensor[index: 1+index, ...])
                                cnt_homo += 1
                        for j in range((group_index+1)*batch_k, batch_size):
                                loss_heter += L_metric(anchor, tensor[j:j+1, ...], same_class=False)
                                cnt_heter += 1
        return loss_div, loss_homo/cnt_homo, loss_heter/cnt_heter

def criterion(anchors, positives, negatives):
        loss_homo = L_metric(anchors, positives)
        loss_heter = L_metric(anchors, negatives, False)
        loss_div = 0
        for i in range(anchors.shape[0]):
                loss_div += (L_divergence(anchors[i, ...]) + L_divergence(positives[i, ...]) + L_divergence(negatives[i, ...])) / 3
        return loss_div / anchors.shape[0], loss_homo, loss_heter
        
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

class CenterLoss(nn.Module):
        def __init__(self, num_classes=196, feat_dim=512, beta=0.05, use_gpu=True):
                super(CenterLoss, self).__init__()
                self.num_classes = num_classes
                self.feat_dim = feat_dim
                self.beta = beta
                self.use_gpu = use_gpu
                self.criterion = nn.MSELoss()

                if use_gpu:
                        self.centers = nn.Parameter(torch.zeros(self.num_classes, self.feat_dim).cuda())
                else:
                        self.centers = nn.Parameter(torch.zeros(self.num_classes, self.feat_dim))

        def forward(self, x, labels):
                loss = self.criterion(x, self.centers[labels])
                self.centers[labels].add_(self.beta * (x.detach() - self.centers[labels]))

                return loss

def regularization(model, layer_names):
    assert type(layer_names) == list
    loss = 0.
    cnt = 0
    for layer in layer_names:
        for n, p in getattr(model, layer).named_parameters():
            if 'bias' in n:
                continue
            loss += p.data.norm()**2 / p.data.numel()
            cnt += 1
    return loss / cnt


if __name__ == '__main__':
        a = torch.randn(2, 3)
        b = torch.randn(2, 3)
        print('a&b:', a.size())
        print(L2_squared(a, b).size())
        print(a, torch.exp(a))