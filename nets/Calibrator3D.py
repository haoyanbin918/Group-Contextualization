# Code for "Group Contextualization for Video Recognition"
# CVPR2022
# Yanbin Hao, Hao Zhang, Chong-Wah Ngo, Xiangnan He
# haoyanbin@hotmail.com, zhanghaoinf@gmail.com

import torch
import torch.nn as nn


class GC_L33Dnb(nn.Module):
    def __init__(self, inplanes, planes):
        super(GC_L33Dnb, self).__init__()
        # self.num_segments = num_segments
        self.conv = nn.Conv3d(inplanes, planes, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0), bias=False)
        self.sigmoid = nn.Sigmoid()

        # nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.xavier_normal_(self.conv.weight)
        #
    #
    def forward(self, x):
        y = self.conv(x)
        y = self.sigmoid(y)
        x = x*y

        return x


class GC_T13Dnb(nn.Module):
    def __init__(self, inplanes, planes):
        super(GC_T13Dnb, self).__init__()
        # self.num_segments = num_segments
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

        nn.init.xavier_normal_(self.conv.weight)
    #
    def forward(self, x):
        bn, c, t, h, w = x.size()
        y = x.reshape(bn*c, t, h, w).contiguous()
        y = self.avg_pool(y).view(bn, c, t)
        y = self.conv(y)
        y = self.sigmoid(y)
        y = y.view(bn, c, t, 1, 1)
        x = x*y.expand_as(x)

        return x


class GC_S23DDnb(nn.Module):
    def __init__(self, inplanes, planes):
        super(GC_S23DDnb, self).__init__()
        #
        # self.num_segments = num_segments
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self.sigmoid = nn.Sigmoid()

        # nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.xavier_normal_(self.conv.weight)
        #
    #
    def forward(self, x):
        bn, c, t, h, w = x.size()
        y = x.mean(dim=2).squeeze(2)
        y = self.conv(y)
        y = self.sigmoid(y).view(bn, c, 1, h, w)
        x = x*y.expand_as(x)

        return x


class GC_CLLDnb(nn.Module):
    def __init__(self, inplanes, planes):
        super(GC_CLLDnb, self).__init__()
        # self.num_segments = num_segments
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Linear(inplanes, planes, bias=False)
        self.sigmoid = nn.Sigmoid()

        nn.init.normal_(self.conv.weight, 0, 0.001)

    def forward(self, x):
        bn, c, t, h, w = x.size()
        y = self.avg_pool(x).view(bn, c)
        y = self.conv(y)
        y = self.sigmoid(y).view(bn, c, 1, 1, 1)
        x = x*y.expand_as(x)

        return x
