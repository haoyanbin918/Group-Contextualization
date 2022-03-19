import torch.nn as nn
from ops.transforms import *
from torch.nn.init import normal_, constant_
from ops.basic_ops import ConsensusModule

import sys
from importlib import import_module
sys.path.append('..')
# import C2D

class VideoNet(nn.Module):
    def __init__(self, num_class, num_segments, modality,
                backbone='resnet50', net=None, consensus_type='avg',
                dropout=0.5, partial_bn=True, print_spec=True, pretrain='imagenet', ef_lr5=False,
                fc_lr5=False, alpha=4, beta=2, non_local=False, element_filter=False, stage='S2B', cdiv=2, loop=False, target_transforms=None):
        super(VideoNet, self).__init__()
        self.num_segments = num_segments
        self.modality = modality
        self.backbone = backbone
        self.net = net
        self.dropout = dropout
        self.pretrain = pretrain
        self.consensus_type = consensus_type
        self.init_crop_size = 256

        self.alpha = alpha
        self.beta = beta
        self.stage = stage
        self.loop = loop

        self.ef_lr5 = ef_lr5
        self.fc_lr5 = fc_lr5
        self.non_local = non_local
        self.element_filter = element_filter
        self.cdiv = cdiv
        self.target_transforms = target_transforms

        self._prepare_base_model(backbone)
        self._prepare_fc(num_class)
        self.consensus = ConsensusModule(consensus_type)
        self.softmax = nn.Softmax()
        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def _prepare_base_model(self, backbone):
        # assert not (self.non_local and self.element_filter)
        # assert self.pretrain == 'imagenet'
        
        if 'resnet' in backbone:
            # if self.non_local:
            #     print('=> base model: STC_NLN, with backbone: {}'.format(backbone))
            #     import nets.GST_NLN as GST_NLN
            #     self.base_model = getattr(GST_NLN, backbone)(alpha=self.alpha, beta=self.beta)
            if self.net == 'GST':
                print('=> base model: GST, with backbone: {}'.format(backbone))
                import nets.GST as GST
                self.base_model = getattr(GST, backbone)(alpha=self.alpha, beta=self.beta)
                
            elif self.element_filter:
                if self.net == 'M4':
                    assert self.cdiv >= 4
                    base_model_name = 'nets.GC_GST' 
                    GST_GC = import_module(base_model_name)
                    print('=> base model: {}, with backbone: {}, loop: {}'.format(base_model_name, backbone, self.loop))
                    self.base_model = getattr(GST_GC, backbone)(alpha=self.alpha, beta=self.beta, cdiv=self.cdiv, loop=self.loop)
                else:
                    base_model_name = 'nets.ECal_GST' + '_S'
                    print('=> base model: {}, with backbone: {}, net: {}, loop: {}'.format(base_model_name, backbone, self.net, self.loop))
                    GST_GC = import_module(base_model_name)
                    block = 'GC_%s'%self.net
                    self.base_model = getattr(GST_GC, backbone)(EF=block, alpha=self.alpha, beta=self.beta, cdiv=self.cdiv, loop=self.loop)
            else:
                raise ValueError('Wrong networks')
            #
            self.base_model.last_layer_name = 'fc'
            self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]
        #
        else:
            raise ValueError('Unknown backbone: {}'.format(backbone))

    def _prepare_fc(self, num_class):
        self.feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
        if self.dropout == 0:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(self.feature_dim, num_class))
            self.new_fc = None
        else:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
            self.new_fc = nn.Linear(self.feature_dim, num_class)

        std = 0.001
        if self.new_fc is None:
            normal_(getattr(self.base_model, self.base_model.last_layer_name).weight, 0, std)
            constant_(getattr(self.base_model, self.base_model.last_layer_name).bias, 0)
        else:
            if hasattr(self.new_fc, 'weight'):
                normal_(self.new_fc.weight, 0, std)
                constant_(self.new_fc.bias, 0)

    #
    def train(self, mode=True):
        # Override the default train() to freeze the BN parameters
        super(VideoNet, self).train(mode)
        count = 0
        if self._enable_pbn and mode:
            print("Freezing BatchNorm3D except the first one.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm3d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()
                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

    #
    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        lr5_weight = []
        lr10_bias = []
        bn = []
        custom_ops = []

        ef_weight = []
        ef_bias = []

        conv_cnt = 0
        bn_cnt = 0
        for m, p in self.named_modules():
            if 'eft' in m or 'efs' in m or 'efc' in m:
                if isinstance(p, torch.nn.Conv1d) or isinstance(p, torch.nn.Conv2d) or isinstance(p, torch.nn.Conv3d) or isinstance(p, torch.nn.ConvTranspose3d):
                    if self.ef_lr5:
                        ps = list(p.parameters())
                        ef_weight.append(ps[0])
                        if len(ps)==2:
                            ef_bias.append(ps[1])
                    else:
                        ps = list(p.parameters())
                        normal_weight.append(ps[0])
                        if len(ps)==2:
                            normal_bias.append(ps[1])
                elif isinstance(p, torch.nn.Linear):
                    if self.ef_lr5:
                        ps = list(p.parameters())
                        ef_weight.append(ps[0])
                        if len(ps)==2:
                            ef_bias.append(ps[1])
                    else:
                        ps = list(p.parameters())
                        normal_weight.append(ps[0])
                        if len(ps)==2:
                            normal_bias.append(ps[1])
                elif isinstance(p, torch.nn.BatchNorm3d) or isinstance(p, torch.nn.BatchNorm2d) or isinstance(p, torch.nn.BatchNorm1d):
                    bn.extend(list(p.parameters()))
                elif len(p._modules) == 0:
                    if len(list(p.parameters())) > 0:
                        raise ValueError("New atomic module type: {} in eft blocks. Need to give it a learning policy".format(type(m)))

            else:
                if isinstance(p, torch.nn.Conv2d) or isinstance(p, torch.nn.Conv3d):
                    ps = list(p.parameters())
                    conv_cnt += 1
                    if conv_cnt == 1:
                        first_conv_weight.append(ps[0])
                        if len(ps) == 2:
                            first_conv_bias.append(ps[1])
                    else:
                        normal_weight.append(ps[0])
                        if len(ps) == 2:
                            normal_bias.append(ps[1])

                elif isinstance(p, torch.nn.ConvTranspose3d):
                    ps = list(p.parameters())
                    if self.fc_lr5:
                        lr5_weight.append(ps[0])
                        if len(ps) == 2:
                            lr5_weight.append(ps[1])
                    else:
                        normal_weight.append(ps[0])
                        if len(ps) == 2:
                            normal_weight.append(ps[1])

                elif isinstance(p, torch.nn.Linear):
                    ps = list(p.parameters())
                    if self.fc_lr5:
                        lr5_weight.append(ps[0])
                    else:
                        normal_weight.append(ps[0])
                    if len(ps) == 2:
                        if self.fc_lr5:
                            lr10_bias.append(ps[1])
                        else:
                            normal_bias.append(ps[1])
                elif isinstance(p, torch.nn.BatchNorm3d) or isinstance(p, torch.nn.BatchNorm2d):
                    bn_cnt += 1
                    # later BN's are frozen
                    if not self._enable_pbn or bn_cnt == 1:
                        bn.extend(list(p.parameters()))
                elif len(p._modules) == 0:
                    if len(list(p.parameters())) > 0:
                        raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        return [
            {'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
            {'params': custom_ops, 'lr_mult': 1, 'decay_mult': 1,
             'name': "custom_ops"},
            # for ef
            {'params': ef_weight, 'lr_mult': 2, 'decay_mult': 1,
             'name': "ef_weight"},
            {'params': ef_bias, 'lr_mult': 4, 'decay_mult': 0,
             'name': "ef_bias"},
            # for fc
            {'params': lr5_weight, 'lr_mult': 5, 'decay_mult': 1,
             'name': "lr5_weight"},
            {'params': lr10_bias, 'lr_mult': 10, 'decay_mult': 0,
             'name': "lr10_bias"},
        ]

    def forward(self, input):
        # input size [batch_size, num_segments, 3, h, w]
        input = input.view((-1, self.num_segments, 3) + input.size()[-2:])
        input = input.transpose(1, 2).contiguous()
        base_out = self.base_model(input)
        if self.dropout > 0:
            base_out = self.new_fc(base_out)
        base_out = base_out.view((-1,self.num_segments)+base_out.size()[1:])
        #
        if self.consensus_type == 'avg':
            output = self.consensus(base_out)
            return output.squeeze(1)
        else:
            output = base_out.mean(dim=1)
            return output

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * self.init_crop_size // self.input_size

    def get_augmentation(self, flip=True):
        if self.modality == 'RGB':
            if flip:
                return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                                       GroupRandomHorizontalFlip2(self.target_transforms)])
            else:
                print('#' * 20, 'NO FLIP!!!')
                return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66])])
        elif self.modality == 'Flow':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=True)])
        elif self.modality == 'RGBDiff':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])