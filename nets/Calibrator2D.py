# Code for "Group Contextualization for Video Recognition"
# CVPR2022
# Yanbin Hao, Hao Zhang, Chong-Wah Ngo, Xiangnan He
# haoyanbin@hotmail.com, zhanghaoinf@gmail.com

import torch
import torch.nn as nn

class GC_L33D(nn.Module):
    def __init__(self, inplanes, planes, num_segments=8):
        super(GC_L33D, self).__init__()
        self.num_segments = num_segments
        self.conv = nn.Conv3d(inplanes, planes, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0), bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.sigmoid = nn.Sigmoid()

        nn.init.xavier_normal_(self.conv.weight)
        #
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)
    #
    def forward(self, x):
        bn, c, h, w = x.size()
        x = x.view(-1, self.num_segments, c, h, w)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        y = self.conv(x)
        y = self.bn1(y)
        y = self.sigmoid(y)
        x = x*y
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(-1, c, h, w)

        return x


class GC_T13D(nn.Module):
    def __init__(self, inplanes, planes, num_segments=8):
        super(GC_T13D, self).__init__()
        self.num_segments = num_segments
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.sigmoid = nn.Sigmoid()

        nn.init.xavier_normal_(self.conv.weight)
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)
    #
    def forward(self, x):
        bn, c, h, w = x.size()
        y = self.avg_pool(x).view(-1, self.num_segments, c)
        y = y.permute(0, 2, 1).contiguous()
        y = self.conv(y)
        y = self.bn1(y)
        y = self.sigmoid(y)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(-1, self.num_segments, c, 1, 1)
        y = y.view(-1, c, 1, 1)
        x = x*y.expand_as(x)

        return x

class GC_S23DD(nn.Module):
    def __init__(self, inplanes, planes, num_segments=8):
        super(GC_S23DD, self).__init__()
        #
        #
        self.num_segments = num_segments
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.sigmoid = nn.Sigmoid()

        nn.init.xavier_normal_(self.conv.weight)
        #
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)
    #
    def forward(self, x):
        bn, c, h, w = x.size()
        x = x.view(-1, self.num_segments, c, h, w)
        y = x.mean(dim=1).squeeze(1)
        y = self.conv(y)
        y = self.bn1(y)
        y = self.sigmoid(y).view(-1, 1, c, h, w)
        x = x*y.expand_as(x)

        return x.view(-1, c, h, w)

    #
    def forward(self, x):
        bn, c, h, w = x.size()
        x = x.view(-1, self.num_segments, c, h, w)
        y = x.mean(dim=1).squeeze(1)
        y = self.conv(y)
        y = self.sigmoid(y).view(-1, 1, c, h, w)
        x = x*y.expand_as(x)

        return x.view(-1, c, h, w)


class GC_CLLD(nn.Module):
    def __init__(self, inplanes, planes, num_segments=8):
        super(GC_CLLD, self).__init__()
        self.num_segments = num_segments
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Linear(inplanes, planes, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.sigmoid = nn.Sigmoid()

        nn.init.normal_(self.conv.weight, 0, 0.001)
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)

    def forward(self, x):
        bn, c, h, w = x.size()
        batch_size = bn//self.num_segments
        x = x.view(batch_size, self.num_segments, c, h, w)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        #
        y = self.avg_pool(x).view(batch_size, c)
        y = self.conv(y)
        y = self.bn1(y)
        y = self.sigmoid(y).view(batch_size, c, 1, 1, 1)
        x = x*y.expand_as(x)
        #
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(bn, c, h, w)

        return x
