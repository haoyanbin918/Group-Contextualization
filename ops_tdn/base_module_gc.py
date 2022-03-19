from __future__ import print_function, division, absolute_import
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.utils.model_zoo as model_zoo

from ops.Calibrator2D import GC_L33D, GC_T13D, GC_S23DD, GC_CLLD

__all__ = ['FBResNet', 'fbresnet50', 'fbresnet101']

model_urls = {
        'fbresnet50': 'http://data.lip6.fr/cadene/pretrainedmodels/resnet50-19c8e357.pth',
        'fbresnet101': 'http://data.lip6.fr/cadene/pretrainedmodels/resnet101-5d3b4d8f.pth'
}


class mSEModule(nn.Module):
    def __init__(self, channel, n_segment=8,index=1):
        super(mSEModule, self).__init__()
        self.channel = channel
        self.reduction = 16
        self.n_segment = n_segment
        self.stride = 2**(index-1)
        self.conv1 = nn.Conv2d(in_channels=self.channel,
                out_channels=self.channel//self.reduction,
                kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=self.channel//self.reduction)
        self.conv2 = nn.Conv2d(in_channels=self.channel//self.reduction,
                out_channels=self.channel//self.reduction,
                kernel_size=3, padding=1, groups=self.channel//self.reduction, bias=False)

        self.avg_pool_forward2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.avg_pool_forward4 = nn.AvgPool2d(kernel_size=4, stride=4)
        
        self.sigmoid_forward = nn.Sigmoid()

        self.avg_pool_backward2 = nn.AvgPool2d(kernel_size=2, stride=2)#nn.AdaptiveMaxPool2d(1)
        self.avg_pool_backward4 = nn.AvgPool2d(kernel_size=4, stride=4)

        self.sigmoid_backward = nn.Sigmoid()

        self.pad1_forward = (0, 0, 0, 0, 0, 0, 0, 1)
        self.pad1_backward = (0, 0, 0, 0, 0, 0, 1, 0)

        self.conv3 = nn.Conv2d(in_channels=self.channel//self.reduction,
                 out_channels=self.channel, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=self.channel)

        self.conv3_smallscale2 = nn.Conv2d(in_channels=self.channel//self.reduction,
                                  out_channels=self.channel//self.reduction,padding=1, kernel_size=3, bias=False)
        self.bn3_smallscale2 = nn.BatchNorm2d(num_features=self.channel//self.reduction)
        
        self.conv3_smallscale4 = nn.Conv2d(in_channels = self.channel//self.reduction,
                                  out_channels=self.channel//self.reduction,padding=1, kernel_size=3, bias=False)
        self.bn3_smallscale4 = nn.BatchNorm2d(num_features=self.channel//self.reduction)

    def spatial_pool(self, x):
        nt, channel, height, width = x.size()
        input_x = x
        # [N, C, H * W]
        input_x = input_x.view(nt, channel, height * width)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]
        context_mask = self.conv_mask(x)
        # [N, 1, H * W]
        context_mask = context_mask.view(nt, 1, height * width)
        # [N, 1, H * W]
        context_mask = self.softmax(context_mask)
        context_mask = context_mask.view(nt,1,height,width)
        return context_mask


    def forward(self, x):
        bottleneck = self.conv1(x) # nt, c//r, h, w
        bottleneck = self.bn1(bottleneck) # nt, c//r, h, w
        reshape_bottleneck = bottleneck.view((-1, self.n_segment) + bottleneck.size()[1:]) # n, t, c//r, h, w
        
        t_fea_forward, _ = reshape_bottleneck.split([self.n_segment -1, 1], dim=1) # n, t-1, c//r, h, w
        _, t_fea_backward = reshape_bottleneck.split([1, self.n_segment -1], dim=1) # n, t-1, c//r, h, w
        
        conv_bottleneck = self.conv2(bottleneck) # nt, c//r, h, w
        reshape_conv_bottleneck = conv_bottleneck.view((-1, self.n_segment) + conv_bottleneck.size()[1:]) # n, t, c//r, h, w
        _, tPlusone_fea_forward = reshape_conv_bottleneck.split([1, self.n_segment-1], dim=1) # n, t-1, c//r, h, w
        tPlusone_fea_backward ,_ = reshape_conv_bottleneck.split([self.n_segment-1, 1], dim=1) # n, t-1, c//r, h, w
        diff_fea_forward = tPlusone_fea_forward - t_fea_forward # n, t-1, c//r, h, w
        diff_fea_backward = tPlusone_fea_backward - t_fea_backward# n, t-1, c//r, h, w
        diff_fea_pluszero_forward = F.pad(diff_fea_forward, self.pad1_forward, mode="constant", value=0) # n, t, c//r, h, w
        diff_fea_pluszero_forward = diff_fea_pluszero_forward.view((-1,) + diff_fea_pluszero_forward.size()[2:]) #nt, c//r, h, w
        diff_fea_pluszero_backward = F.pad(diff_fea_backward, self.pad1_backward, mode="constant", value=0) # n, t, c//r, h, w
        diff_fea_pluszero_backward = diff_fea_pluszero_backward.view((-1,) + diff_fea_pluszero_backward.size()[2:]) #nt, c//r, h, w
        y_forward_smallscale2 = self.avg_pool_forward2(diff_fea_pluszero_forward) # nt, c//r, 1, 1
        y_backward_smallscale2 = self.avg_pool_backward2(diff_fea_pluszero_backward) # nt, c//r, 1, 1

        y_forward_smallscale4 = diff_fea_pluszero_forward
        y_backward_smallscale4 = diff_fea_pluszero_backward
        y_forward_smallscale2 = self.bn3_smallscale2(self.conv3_smallscale2(y_forward_smallscale2))
        y_backward_smallscale2 = self.bn3_smallscale2(self.conv3_smallscale2(y_backward_smallscale2))

        y_forward_smallscale4 = self.bn3_smallscale4(self.conv3_smallscale4(y_forward_smallscale4))
        y_backward_smallscale4 = self.bn3_smallscale4(self.conv3_smallscale4(y_backward_smallscale4))
        
        y_forward_smallscale2 = F.interpolate(y_forward_smallscale2, diff_fea_pluszero_forward.size()[2:])
        y_backward_smallscale2 = F.interpolate(y_backward_smallscale2, diff_fea_pluszero_backward.size()[2:])
        
        y_forward = self.bn3(self.conv3(1.0/3.0*diff_fea_pluszero_forward + 1.0/3.0*y_forward_smallscale2 + 1.0/3.0*y_forward_smallscale4))# nt, c, 1, 1
        y_backward = self.bn3(self.conv3(1.0/3.0*diff_fea_pluszero_backward + 1.0/3.0*y_backward_smallscale2 + 1.0/3.0*y_backward_smallscale4)) # nt, c, 1, 1

        y_forward = self.sigmoid_forward(y_forward) - 0.5
        y_backward = self.sigmoid_backward(y_backward) - 0.5

        y = 0.5*y_forward + 0.5*y_backward
        output = x + x*y
        return output

class ShiftModule(nn.Module):
    def __init__(self, input_channels, n_segment=8,n_div=8, mode='shift'):
        super(ShiftModule, self).__init__()
        self.input_channels = input_channels
        self.n_segment = n_segment
        self.fold_div = n_div
        self.fold = self.input_channels // self.fold_div
        self.conv = nn.Conv1d(self.fold_div*self.fold, self.fold_div*self.fold,
                kernel_size=3, padding=1, groups=self.fold_div*self.fold,
                bias=False)

        if mode == 'shift':
            self.conv.weight.requires_grad = True
            self.conv.weight.data.zero_()
            self.conv.weight.data[:self.fold, 0, 2] = 1 # shift left
            self.conv.weight.data[self.fold: 2 * self.fold, 0, 0] = 1 # shift right
            if 2*self.fold < self.input_channels:
                self.conv.weight.data[2 * self.fold:, 0, 1] = 1 # fixed
        elif mode == 'fixed':
            self.conv.weight.requires_grad = True
            self.conv.weight.data.zero_()
            self.conv.weight.data[:, 0, 1] = 1 # fixed
        elif mode == 'norm':
            self.conv.weight.requires_grad = True

    def forward(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w)
        x = x.permute(0, 3, 4, 2, 1) # (n_batch, h, w, c, n_segment)
        x = x.contiguous().view(n_batch*h*w, c, self.n_segment)
        x = self.conv(x) # (n_batch*h*w, c, n_segment)
        x = x.view(n_batch, h, w, c, self.n_segment)
        x = x.permute(0, 4, 3, 1, 2) # (n_batch, n_segment, c, h, w)
        x = x.contiguous().view(nt, c, h, w)
        return x

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, num_segments, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, num_segments, inplanes, planes, stride=1, downsample=None, use_ef=False, cdiv=4, loop_id=0):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=True)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

        self.use_ef = use_ef

        if self.use_ef:
            print('=> Using Partial Channel Calibrator with cdiv: {}'.format(cdiv))
            self.loop_id = loop_id
            self.eft_c = planes // cdiv
            self.eft1 = GC_L33D(self.eft_c, self.eft_c, num_segments)
            self.eft2 = GC_T13D(self.eft_c, self.eft_c, num_segments)
            self.eft3 = GC_S23DD(self.eft_c, self.eft_c, num_segments)
            self.eft4 = GC_CLLD(self.eft_c, self.eft_c, num_segments)
            # self.eft = (self.eft_c, self.eft_c, num_segments)
            self.start_c1 = loop_id*self.eft_c
            self.end_c1 = self.start_c1 + self.eft_c
            loop_id2 = (loop_id+1)%cdiv
            self.start_c2 = loop_id2*self.eft_c
            self.end_c2 = self.start_c2 + self.eft_c
            loop_id3 = (loop_id+2)%cdiv
            self.start_c3 = loop_id3*self.eft_c
            self.end_c3 = self.start_c3 + self.eft_c
            loop_id4 = (loop_id+3)%cdiv
            self.start_c4 = loop_id4*self.eft_c
            self.end_c4 = self.start_c4 + self.eft_c
            print('loop_ids: [{}:({}-{}), {}:({}-{}), {}:({}-{}), {}:({}-{})]'.format(loop_id, self.start_c1, self.end_c1, \
                loop_id2, self.start_c2, self.end_c2, loop_id3, self.start_c3, self.end_c3, loop_id4, self.start_c4, self.end_c4))

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_ef:
            new_out = torch.zeros_like(out)
            BN, C_size, H_size, W_size = new_out.size()
            # new_out = out
            new_out[:, self.start_c1:self.end_c1, :, :] = self.eft1(out[:, self.start_c1:self.end_c1, :, :])
            new_out[:, self.start_c2:self.end_c2, :, :] = self.eft2(out[:, self.start_c2:self.end_c2, :, :])
            new_out[:, self.start_c3:self.end_c3, :, :] = self.eft3(out[:, self.start_c3:self.end_c3, :, :])
            new_out[:, self.start_c4:self.end_c4, :, :] = self.eft4(out[:, self.start_c4:self.end_c4, :, :])
            # new_out = torch.zeros_like(out)
            # new_out[:, :self.eft_c, :, :] = self.eft(out[:, :self.eft_c, :, :])
            if self.end_c4 > self.start_c1:
                if self.start_c1 > 0:
                    new_out[:, :self.start_c1:, :, :] = out[:, :self.start_c1:, :, :]
                if self.end_c4 < C_size:
                    new_out[:, self.end_c4:, :, :] = out[:, self.end_c4:, :, :]
            elif self.end_c4 < self.start_c1:
                new_out[:, self.end_c4:self.start_c1:, :, :] = out[:, self.end_c4:self.start_c1:, :, :]

            out = new_out

        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class BottleneckShift(nn.Module):
    expansion = 4

    def __init__(self, num_segments, inplanes, planes, stride=1, downsample=None, use_ef=False, cdiv=4, loop_id=0):
        super(BottleneckShift, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.num_segments = num_segments
        self.mse = mSEModule(planes, n_segment=self.num_segments,index=1) 
        self.shift = ShiftModule(planes, n_segment=self.num_segments, n_div=8, mode='shift')

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=True)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

        self.use_ef = use_ef

        if self.use_ef:
            print('=> Using Partial Channel Calibrator with cdiv: {}'.format(cdiv))
            self.loop_id = loop_id
            self.eft_c = planes // cdiv
            self.eft1 = GC_L33D(self.eft_c, self.eft_c, num_segments)
            self.eft2 = GC_T13D(self.eft_c, self.eft_c, num_segments)
            self.eft3 = GC_S23DD(self.eft_c, self.eft_c, num_segments)
            self.eft4 = GC_CLLD(self.eft_c, self.eft_c, num_segments)
            # self.eft = (self.eft_c, self.eft_c, num_segments)
            self.start_c1 = loop_id*self.eft_c
            self.end_c1 = self.start_c1 + self.eft_c
            loop_id2 = (loop_id+1)%cdiv
            self.start_c2 = loop_id2*self.eft_c
            self.end_c2 = self.start_c2 + self.eft_c
            loop_id3 = (loop_id+2)%cdiv
            self.start_c3 = loop_id3*self.eft_c
            self.end_c3 = self.start_c3 + self.eft_c
            loop_id4 = (loop_id+3)%cdiv
            self.start_c4 = loop_id4*self.eft_c
            self.end_c4 = self.start_c4 + self.eft_c
            print('loop_ids: [{}:({}-{}), {}:({}-{}), {}:({}-{}), {}:({}-{})]'.format(loop_id, self.start_c1, self.end_c1, \
                loop_id2, self.start_c2, self.end_c2, loop_id3, self.start_c3, self.end_c3, loop_id4, self.start_c4, self.end_c4))

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)

        if self.use_ef:
            new_out = torch.zeros_like(out)
            BN, C_size, H_size, W_size = new_out.size()
            # new_out = out
            new_out[:, self.start_c1:self.end_c1, :, :] = self.eft1(out[:, self.start_c1:self.end_c1, :, :])
            new_out[:, self.start_c2:self.end_c2, :, :] = self.eft2(out[:, self.start_c2:self.end_c2, :, :])
            new_out[:, self.start_c3:self.end_c3, :, :] = self.eft3(out[:, self.start_c3:self.end_c3, :, :])
            new_out[:, self.start_c4:self.end_c4, :, :] = self.eft4(out[:, self.start_c4:self.end_c4, :, :])
            # new_out = torch.zeros_like(out)
            # new_out[:, :self.eft_c, :, :] = self.eft(out[:, :self.eft_c, :, :])
            if self.end_c4 > self.start_c1:
                if self.start_c1 > 0:
                    new_out[:, :self.start_c1:, :, :] = out[:, :self.start_c1:, :, :]
                if self.end_c4 < C_size:
                    new_out[:, self.end_c4:, :, :] = out[:, self.end_c4:, :, :]
            elif self.end_c4 < self.start_c1:
                new_out[:, self.end_c4:self.start_c1:, :, :] = out[:, self.end_c4:self.start_c1:, :, :]

            out = new_out

        out = self.relu(out)

        out = self.mse(out)
        out = self.shift(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class FBResNet(nn.Module):

    def __init__(self, num_segments, block, layers, num_classes=1000, use_ef=False, cdiv=4, loop=False):
        self.inplanes = 64

        self.input_space = None
        self.input_size = (224, 224, 3)
        self.mean = None
        self.std = None
        self.num_segments = num_segments
        self.loop_id = 0
        super(FBResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.loop = loop
        self.layer1 = self._make_layer(self.num_segments,Bottleneck, 64, layers[0], use_ef=use_ef, cdiv=cdiv)
        self.layer2 = self._make_layer(self.num_segments,block, 128, layers[1], stride=2, use_ef=use_ef, cdiv=cdiv)
        self.layer3 = self._make_layer(self.num_segments,block, 256, layers[2], stride=2, use_ef=use_ef, cdiv=cdiv)
        self.layer4 = self._make_layer(self.num_segments,block, 512, layers[3], stride=2, use_ef=use_ef, cdiv=cdiv)
        self.last_linear = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _make_layer(self, num_segments ,block, planes, blocks, stride=1, use_ef=False, cdiv=4):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(num_segments, self.inplanes, planes, stride, downsample, use_ef, cdiv, loop_id=self.loop_id))
        self.inplanes = planes * block.expansion

        if self.loop:
            self.loop_id = (self.loop_id+1)%cdiv

        n_round = 1
        if blocks >= 23:
            n_round = 2
            print('=> Using n_round {} to insert Element Filter -T'.format(n_round))

        for i in range(1, blocks):
            if i % n_round == 0:
                layers.append(block(num_segments, self.inplanes, planes, use_ef=use_ef, cdiv=cdiv, loop_id=self.loop_id))
                if self.loop:
                    self.loop_id = (self.loop_id+1)%cdiv
            else:
                layers.append(block(num_segments, self.inplanes, planes, use_ef=False))

        return nn.Sequential(*layers)


    def features(self, input):
        x = self.conv1(input)
        self.conv1_input = x.clone()
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, features):
        adaptiveAvgPoolWidth = features.shape[2]
        x = F.avg_pool2d(features, kernel_size=adaptiveAvgPoolWidth)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x


    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

def fbresnet50(num_segments=8,pretrained=False,num_classes=1000, use_ef=True, cdiv=4, loop=False):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FBResNet(num_segments,BottleneckShift, [3, 4, 6, 3], num_classes=num_classes, use_ef=use_ef, cdiv=cdiv, loop=loop)
    if pretrained:
         model.load_state_dict(model_zoo.load_url(model_urls['fbresnet50']),strict=False)
    return model


def fbresnet101(num_segments=8,pretrained=False,num_classes=1000, use_ef=True, cdiv=4, loop=False):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FBResNet(num_segments,BottleneckShift, [3, 4, 23, 3], num_classes=num_classes, use_ef=use_ef, cdiv=cdiv, loop=loop)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['fbresnet101']),strict=False)
    return model


if __name__ == "__main__":
    import torch
    inputs = torch.rand(8, 3, 224, 224) #[btz, channel, T, H, W]
    # inputs = torch.rand(1, 64, 4, 112, 112) #[btz, channel, T, H, W]
    net = fbresnet50(num_classes=174)
    net.eval()
    output=net(inputs)
    print(output.size())
    from thop import profile
    flops, params = profile(net, inputs=(inputs, ), custom_ops={net: net})
    print(flops)
    print(params)

