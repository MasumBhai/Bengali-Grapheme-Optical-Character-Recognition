import pretrainedmodels
import torch.nn as nn
from torch.nn import functional as F


class ResNet34(nn.Module):
    def __init__(self, pretrained):
        super(ResNet34, self).__init__()

        if pretrained is True:
            self.model = pretrainedmodels.__dict__["resnet34"](pretrained="imagenet")
        else:
            self.model = pretrainedmodels.__dict__["resnet34"](pretrained=None)

        self.layer0 = nn.Linear(512, 168)  # 168 graphemes present in my dataset
        self.layer1 = nn.Linear(512, 11)  # 11 vowel present in my dataset
        self.layer2 = nn.Linear(512, 7)  # 7 consonants present in my dataset

    def forward(self, x): # also, need to modify the forward function from resnet class
        bs, _, _, _ = x.shape  # batchSize, channel, height & weights
        x = self.model.features(x)  # this is i'm pulling from resnet structure
        x = F.adaptive_max_pool2d(x, 1).reshape(bs, -1)
        layer0 = self.layer0(x)
        layer1 = self.layer1(x)
        layer2 = self.layer2(x)

        return layer0, layer1, layer2
