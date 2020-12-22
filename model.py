import pretrainedmodels
import torch.nn as nn
from torch.nn import functional as F


class resnet34(nn.Module):
    def __init__(self, pretrained):
        super(resnet34, self).__init__()
        if pretrained:
            self.model = pretrainedmodels.__dict__["resnet34"](pretrained="imagenet")
        else:
            self.model = pretrainedmodels.__dict__["resnet34"](pretrained=None)

        self.out = nn.Linear(512, 10)

    def forward(self, x):  # Takes a batch
        bs, channels, height, width = x.shape
        x = self.model.features(x)  # function of pretrainedmodels

        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        out = self.out(x)

        return out
