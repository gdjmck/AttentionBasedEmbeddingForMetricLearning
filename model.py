import torch
import torch.nn as nn
import torchvision.models as models
import GoogLeNet

class MetricLearner(GoogLeNet.GoogLeNet):
    def __init__(self, branches=4):
        super(MetricLearner, self).__init__()
        self.att = nn.Conv2d(in_channels=480, out_channels=480)
        nn.init.xavier_uniform_(self.att.weight)

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

    def forward(self, x):
        sp = self.feat_spatial(x)
        a4 = self.inception4a(sp)
        b4 = self.inception4b(a4)
        c4 = self.inception4c(b4)
        d4 = self.inception4d(c4)


if __name__ == '__main__':
    model = MetricLearner()