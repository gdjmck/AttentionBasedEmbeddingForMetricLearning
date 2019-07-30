import torch
import torch.nn as nn
import torchvision.models as models
import GoogLeNet

class MetricLearner(GoogLeNet.GoogLeNet):
    def __init__(self, att_heads=8):
        super(MetricLearner, self).__init__()
        assert 512 % att_heads == 0
        self.att_heads = att_heads
        self.out_dim = int(512 / self.att_heads)
        self.att = nn.ModuleList([nn.Conv2d(in_channels=2896, out_channels=480, kernel_size=1) for i in range(att_heads)])
        for layer in self.att:
            nn.init.xavier_uniform_(layer.weight)
        self.last_fc = nn.Linear(1024, self.out_dim)

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
        x = self.inception4a(x)
        # N x 512 x 14 x 14
        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 528 x 14 x 14
        if self.training and self.aux_logits:
            aux2 = self.aux2(x)

        x = self.inception4e(x)
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
        x = self.dropout(x)
        # N x 1024
        x = self.last_fc(x)
        # N x (512/M)
        return x

    def att_prep(self, x):
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
        return torch.cat([a4, b4, c4, d4, e4], 1)

    def forward(self, x):
        # N x 3 x 224 x 224
        sp = self.feat_spatial(x)
        att_input = self.att_prep(sp)
        atts = [att_func(att_input) for att_func in self.att]
        return torch.cat([self.feat_global(att*sp).unsqueeze(1) for att in atts], 1)
        '''
        embeddings = torch.cat([self.feat_global(att*sp) for att in atts], 1)
        assert embeddings.shape[1] == 512
        return embeddings
        '''

if __name__ == '__main__':
    import numpy as np
    model = MetricLearner()
    input = torch.Tensor(np.zeros((1, 3, 224, 224)))
    output = model(input)
    print(output.detach().shape)