import torch
import torch.nn as nn

class EF_E33D(nn.Module):
    def __init__(self, inplanes, planes, cdiv=1, num_segments=8):
        super(EF_E33D, self).__init__()
        self.num_segments = num_segments
        self.conv = nn.Conv3d(inplanes, planes//cdiv, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0), bias=False)
        self.bn1 = nn.BatchNorm3d(planes//cdiv)
        self.deconv = nn.Conv3d(planes//cdiv, planes, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0), bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.xavier_normal_(self.deconv.weight)
        #
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)
        nn.init.constant_(self.bn2.weight, 1)
        nn.init.constant_(self.bn2.bias, 0)
    #
    def forward(self, x):
        bn, c, h, w = x.size()
        x = x.view(-1, self.num_segments, c, h, w)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        y = self.conv(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.deconv(y)
        y = self.bn2(y)
        y = self.sigmoid(y)
        x = x*y
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(-1, c, h, w)

        return x

class EF_E31D(nn.Module):
    def __init__(self, inplanes, planes, cdiv=1, num_segments=8):
        super(EF_E31D, self).__init__()
        self.num_segments = num_segments
        self.conv = nn.Conv3d(inplanes, planes//cdiv, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0), bias=False)
        self.bn1 = nn.BatchNorm3d(planes//cdiv)
        self.deconv = nn.Conv3d(planes//cdiv, planes, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.xavier_normal_(self.deconv.weight)
        #
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)
        nn.init.constant_(self.bn2.weight, 1)
        nn.init.constant_(self.bn2.bias, 0)
    #
    def forward(self, x):
        bn, c, h, w = x.size()
        x = x.view(-1, self.num_segments, c, h, w)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        y = self.conv(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.deconv(y)
        y = self.bn2(y)
        y = self.sigmoid(y)
        x = x*y
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(-1, c, h, w)

        return x

class EF_T33D(nn.Module):
    def __init__(self, inplanes, planes, cdiv=1, num_segments=8):
        super(EF_T33D, self).__init__()
        self.num_segments = num_segments
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(inplanes, planes//cdiv, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes//cdiv)
        self.deconv = nn.Conv1d(planes//cdiv, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.xavier_normal_(self.deconv.weight)
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)
        nn.init.constant_(self.bn2.weight, 1)
        nn.init.constant_(self.bn2.bias, 0)
    #
    def forward(self, x):
        bn, c, h, w = x.size()
        y = self.avg_pool(x).view(-1, self.num_segments, c)
        y = y.permute(0, 2, 1).contiguous()
        y = self.conv(y)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.deconv(y)
        y = self.bn2(y)
        y = self.sigmoid(y)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(-1, self.num_segments, c, 1, 1)
        y = y.view(-1, c, 1, 1)
        x = x*y.expand_as(x)

        return x

class EF_T31D(nn.Module):
    def __init__(self, inplanes, planes, cdiv=1, num_segments=8):
        super(EF_T31D, self).__init__()
        self.num_segments = num_segments
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(inplanes, planes//cdiv, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes//cdiv)
        self.deconv = nn.Conv1d(planes//cdiv, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.xavier_normal_(self.deconv.weight)
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)
        nn.init.constant_(self.bn2.weight, 1)
        nn.init.constant_(self.bn2.bias, 0)
    #
    def forward(self, x):
        bn, c, h, w = x.size()
        y = self.avg_pool(x).view(-1, self.num_segments, c)
        y = y.permute(0, 2, 1).contiguous()
        y = self.conv(y)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.deconv(y)
        y = self.bn2(y)
        y = self.sigmoid(y)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(-1, self.num_segments, c, 1, 1)
        y = y.view(-1, c, 1, 1)
        x = x*y.expand_as(x)

        return x

class EF_S33D(nn.Module):
    def __init__(self, inplanes, planes, cdiv=2, num_segments=8):
        super(EF_S33D, self).__init__()
        #
        # outpadding, if h/w is odd, use (1, 1), else even, use (0, 0)
        #
        self.num_segments = num_segments
        self.conv = nn.Conv2d(inplanes, planes//cdiv, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes//cdiv)
        self.deconv = nn.Conv2d(planes//cdiv, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.xavier_normal_(self.deconv.weight)
        #
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)
        nn.init.constant_(self.bn2.weight, 1)
        nn.init.constant_(self.bn2.bias, 0)
    #
    def forward(self, x):
        bn, c, h, w = x.size()
        x = x.view(-1, self.num_segments, c, h, w)
        y = x.mean(dim=1).squeeze(1)
        y = self.conv(y)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.deconv(y)
        y = self.bn2(y)
        y = self.sigmoid(y).view(-1, 1, c, h, w)
        x = x*y.expand_as(x)

        return x.view(-1, c, h, w)

class EF_S31D(nn.Module):
    def __init__(self, inplanes, planes, cdiv=2, num_segments=8):
        super(EF_S31D, self).__init__()
        #
        # outpadding, if h/w is odd, use (1, 1), else even, use (0, 0)
        #
        self.num_segments = num_segments
        self.conv = nn.Conv2d(inplanes, planes//cdiv, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes//cdiv)
        self.deconv = nn.Conv2d(planes//cdiv, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.xavier_normal_(self.deconv.weight)
        #
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)
        nn.init.constant_(self.bn2.weight, 1)
        nn.init.constant_(self.bn2.bias, 0)
    #
    def forward(self, x):
        bn, c, h, w = x.size()
        x = x.view(-1, self.num_segments, c, h, w)
        y = x.mean(dim=1).squeeze(1)
        y = self.conv(y)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.deconv(y)
        y = self.bn2(y)
        y = self.sigmoid(y).view(-1, 1, c, h, w)
        x = x*y.expand_as(x)

        return x.view(-1, c, h, w)

class EF_CLLD(nn.Module):
    def __init__(self, inplanes, planes, cdiv=1, num_segments=8):
        super(EF_CLLD, self).__init__()
        self.num_segments = num_segments
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Linear(inplanes, planes//cdiv, bias=False)
        self.bn1 = nn.BatchNorm1d(planes//cdiv)
        self.deconv = nn.Linear(planes//cdiv, planes, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        nn.init.normal_(self.conv.weight, 0, 0.001)
        nn.init.normal_(self.deconv.weight, 0, 0.001)
        #
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)
        nn.init.constant_(self.bn2.weight, 1)
        nn.init.constant_(self.bn2.bias, 0)

    def forward(self, x):
        bn, c, h, w = x.size()
        batch_size = bn//self.num_segments
        x = x.view(batch_size, self.num_segments, c, h, w)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        #
        y = self.avg_pool(x).view(batch_size, c)
        y = self.conv(y)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.deconv(y)
        y = self.bn2(y)
        y = self.sigmoid(y).view(batch_size, c, 1, 1, 1)
        x = x*y.expand_as(x)
        #
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(bn, c, h, w)

        return x

if __name__ == "__main__":
    inputs = torch.rand(24, 256, 56, 56) #[btz, channel, T, H, W]
    # inputs = torch.rand(1, 64, 4, 112, 112) #[btz, channel, T, H, W]
    net = EF_S(256, 256, 16, 8)
    output = net(inputs)
    print(output.size())
