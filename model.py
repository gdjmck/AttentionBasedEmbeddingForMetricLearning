import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
import GoogLeNet
import os

class MetricLearner(GoogLeNet.GoogLeNet):
    def __init__(self, att_heads=8, pretrain=None, batch_k=5, normalize=False):
        super(MetricLearner, self).__init__()
        if pretrain:
            if os.path.exists(pretrain):
                self.load_state_dict(torch.load(pretrain))
                print('Loaded pretrained GoogLeNet.')
            else:
                print('Downloading pretrained GoogLeNet.')
                state_dict = torch.utils.model_zoo.load_url('https://download.pytorch.org/models/googlenet-1378be20.pth')
                self.load_state_dict(state_dict)
        assert 512 % att_heads == 0
        self.att_heads = att_heads
        self.out_dim = int(512 / self.att_heads)
        self.att_depth = 480
        self.att = nn.ModuleList([nn.Conv2d(in_channels=832, out_channels=self.att_depth, kernel_size=1) for i in range(att_heads)])
        self.batch_norm = nn.BatchNorm2d(self.att_depth)
        self.last_fc = nn.Linear(1024, self.out_dim)

        self.sampled = DistanceWeightedSampling(batch_k=batch_k, normalize=normalize)

    def feat_spatial(self, x):
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)

        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28
        x = self.maxpool3(x)

        return x

    def feat_global(self, x):
        # N x 480 x 14 x 14
        x = self.a4_to_e4(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7

        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        #x = self.dropout(x)
        # N x 1024
        x = self.last_fc(x)
        # N x (512/M)
        x = F.normalize(x)
        return x

    def a4_to_e4(self, x):
        # N x 480 x 14 x 14
        a4 = self.inception4a(x)
        # N x 512 x 14 x 14
        b4 = self.inception4b(a4)
        # N x 512 x 14 x 14
        c4 = self.inception4c(b4)
        # N x 512 x 14 x 14
        d4 = self.inception4d(c4)
        # N x 528 x 14 x 14
        e4 = self.inception4e(d4)
        # N x 832 x 14 x 14      
        return e4

    def forward(self, x, ret_att=False, sampling=False):
        # N x 3 x 224 x 224
        sp = self.feat_spatial(x)
        # output of pool3
        att_input = self.a4_to_e4(sp)
        atts = [torch.sigmoid(self.batch_norm(self.att[i](att_input))) for i in range(self.att_heads)] # (N, att_heads, depth, H, W)
        # Normalize attention map
        '''
        for i in range(len(atts)):
            N, D, H, W = atts[i].size()
            att = atts[i].view(-1, H*W)
            att_max, _ = att.max(dim=1, keepdim=True)
            att_min, _ = att.min(dim=1, keepdim=True)
            atts[i] = ((att - att_min) / (att_max - att_min)).view(N, D, H, W)
        '''

        embedding = torch.cat([self.feat_global(atts[i]*sp).unsqueeze(1) for i in range(self.att_heads)], 1)
        #print('embedding in forward:', embedding.shape)
        embedding = torch.flatten(embedding, 1)
        if sampling:
            return self.sampled(embedding) if not ret_att else (self.sampled(embedding), atts)
        else:
            return (embedding, atts) if ret_att else embedding

def l2_norm(x):
    if len(x.shape):
        x = x.reshape((x.shape[0],-1))
    return F.normalize(x, p=2, dim=1)


def get_distance(x):
    _x = x.detach()
    sim = torch.matmul(_x, _x.t())
    sim = torch.clamp(sim, max=1.0)
    #print('\n\n', np.count_nonzero(sim.cpu().numpy() > 0.9) / (sim.shape[0] * sim.shape[1]), sim.shape)
    dist = 2 - 2*sim
    dist += torch.eye(dist.shape[0]).to(dist.device)   # maybe dist += torch.eye(dist.shape[0]).to(dist.device)*1e-8
    dist = dist.sqrt()
    return dist

class   DistanceWeightedSampling(nn.Module):

    def __init__(self, batch_k, cutoff=0.5, nonzero_loss_cutoff=1.4, normalize =False,  **kwargs):
        super(DistanceWeightedSampling,self).__init__()
        self.batch_k = batch_k
        self.cutoff = cutoff
        self.nonzero_loss_cutoff = nonzero_loss_cutoff
        self.normalize = normalize

    def forward(self, x):
        k = self.batch_k
        n, d = x.shape
        x_in = x
        x = F.normalize(x)
        #print('Raw x:', x[0, :])
        distance = get_distance(x) # n x n
        #print('Raw distance:', distance[0, ...])
        distance = distance.clamp(min=self.cutoff) # 将inner product > 0.875 的压缩到0.875，即不让两个vector太过像
        log_weights = ((2.0 - float(d)) * distance.log() - (float(d-3)/2)*torch.log(torch.clamp(1.0 - 0.25*(distance*distance), min=1e-8)))

        if self.normalize:
            log_weights = (log_weights - log_weights.min()) / (log_weights.max() - log_weights.min() + 1e-8)

        weights = torch.exp(log_weights - torch.max(log_weights))

        if x.device != weights.device:
            weights = weights.to(x.device)

        mask = torch.ones_like(weights)
        for i in range(0,n,k):
            mask[i:i+k, i:i+k] = 0

        mask_uniform_probs = mask.double() *(1.0/(n-k))

        weights = weights*mask*((distance < self.nonzero_loss_cutoff).float()) + 1e-8
        weights_sum = torch.sum(weights, dim=1, keepdim=True)
        weights = weights / weights_sum
        #print('\t1st line of w:', weights[0, ...])

        a_indices = []
        p_indices = []
        n_indices = []

        np_weights = weights.cpu().numpy()
        # np_weights = np.nan_to_num(np_weights, 1e-8)
        for i in range(n):
            block_idx = i // k

            if weights_sum[i] != 0:
                n_indices +=  np.random.choice(n, k-1, p=np_weights[i]).tolist()
            else:
                n_indices +=  np.random.choice(n, k-1, p=mask_uniform_probs[i]).tolist()
            for j in range(block_idx*k, (block_idx + 1)*k):
                if j != i:
                    a_indices.append(i)
                    p_indices.append(j)

        return  a_indices, x_in[a_indices], x_in[p_indices], x_in[n_indices], x_in




if __name__ == '__main__':
    import numpy as np
    import sys
    model = MetricLearner()
    for n, p in model.named_parameters():
        print(n, p.data.mean(), p.data.var())
    print('\n=========================================\n')
    attr = 'inception5a'
    for n, p in getattr(model, attr).named_parameters():
        print(n, p.size())
    sys.exit(0)
    from thop import profile
    input = torch.Tensor(np.zeros((1, 3, 224, 224)))
    flops, params = profile(model, inputs=(input, ))
    print(flops, params)
