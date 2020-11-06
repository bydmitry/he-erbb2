import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision import models
import pretrainedmodels

from base import BaseModel

def instantiate_network(conf):
    network_module = getattr(sys.modules[__name__], conf['architecture']['type'])
    model = network_module(**conf['architecture']['args'])

    return model

# ------------------------- FinProgNet -------------------------------------
class FCViewer(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class PretrainedModel(BaseModel):
    def __init__(self, backbone, drop, ncls, pretrained=True):
        super().__init__()
        #if pretrained:
        #    model = pretrainedmodels.__dict__[backbone](num_classes=1000, pretrained='imagenet')
        #else:
        #    model = pretrainedmodels.__dict__[backbone](num_classes=1000, pretrained=None)
        
        model = pretrainedmodels.__dict__[backbone](num_classes=1000, pretrained=None)
        self.encoder = list(model.children())[:-2]

        self.encoder.append(nn.AdaptiveAvgPool2d(1))
        self.encoder = nn.Sequential(*self.encoder)

        if drop > 0:
            self.fc = nn.Sequential(FCViewer(),
                                    nn.Dropout(drop),
                                    nn.Linear(model.last_linear.in_features, ncls))
        else:
            self.fc = nn.Sequential(
                FCViewer(),
                nn.Linear(model.last_linear.in_features, ncls)
            )

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)
        return x

class aux_branch(nn.Module):
    def __init__(self, i_size, o_size, h_size=32, activation=None, drop=0.5):
        super(aux_branch, self).__init__()

        if h_size > 0:
            self.branch = nn.Sequential(
                nn.Dropout(p=drop),
                nn.Linear(i_size, h_size),
                nn.ReLU(),
                nn.Linear(h_size, o_size)
            )
        else:
            self.branch = nn.Sequential(
                nn.Dropout(p=drop),
                nn.Linear(i_size, o_size)
            )

        if activation:
            self.branch.add_module('a', getattr(nn, activation)())

    def forward(self, feats):
        return self.branch(feats)

class FinProgNet(BaseModel):
    def __init__(self, backbone_net, aux_input=None, aux_outputs=None, pretrained=True):
        super(FinProgNet, self).__init__()

        backbone = PretrainedModel(backbone_net, 1, 1, pretrained)
        self.encoder = backbone.encoder

        # configure auxiliary inputs:
        self.aux_mlp = None
        if aux_input is not None:
            self.aux_mlp = aux_branch(**aux_input['net_args'])
            self.joint_feat_size = backbone.fc[-1].in_features + aux_input['net_args'].get('o_size',0)
        else:
            self.joint_feat_size = backbone.fc[-1].in_features

        # configure outputs:
        classifiers = dict()
        for dd in aux_outputs:
            classifiers[dd['name']] = aux_branch(
                i_size = self.joint_feat_size, **dd['args'] )

        self.heads = nn.ModuleDict(classifiers)

    def forward(self, x):
        img = x['rgb']

        o = self.encoder(img)
        img_feats = o.view(o.size(0), -1)

        if self.aux_mlp:
            aux = x['aux_inputs']
            aux_feats = self.aux_mlp(aux)
            img_feats = torch.cat((img_feats, aux_feats), 1)

        outputs = dict()
        for name, subnet in self.heads.items():
            outputs[name] = subnet(img_feats)

        return outputs

# ---------------------- Attention ResNet ----------------------------------
class ProjectorBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ProjectorBlock, self).__init__()
        self.op = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=1, padding=0, bias=False)
    def forward(self, inputs):
        return self.op(inputs)

class LinearAttentionBlock(nn.Module):
    def __init__(self, in_features, normalize_attn=True):
        super(LinearAttentionBlock, self).__init__()
        self.normalize_attn = normalize_attn
        self.op = nn.Conv2d(in_channels=in_features, out_channels=1, kernel_size=1, padding=0, bias=False)
    def forward(self, l, g):
        N, C, W, H = l.size()
        c = self.op(l+g) # batch_sizex1xWxH
        if self.normalize_attn:
            a = F.softmax(c.view(N,1,-1), dim=2).view(N,1,W,H)
        else:
            a = torch.sigmoid(c)
        g = torch.mul(a.expand_as(l), l)
        if self.normalize_attn:
            g = g.view(N,C,-1).sum(dim=2) # batch_sizexC
        else:
            g = F.adaptive_avg_pool2d(g, (1,1)).view(N,C)
        return c.view(N,1,W,H), g

class AttResNet(BaseModel):
    def __init__(self, backbone_net, aux_input=None, aux_outputs=None, attention=True, normalize_attn=True, pretrained=True):
        super(AttResNet, self).__init__()
        self.attention = attention

        if backbone_net.endswith('_wsl'):
            _model = torch.hub.load('facebookresearch/WSL-Images', backbone_net)
        else:
            _model = models.__dict__[backbone_net](pretrained)

        self.encoder1 = nn.Sequential(*list(_model.children())[:-4])
        self.encoder2 = nn.Sequential(*list(_model.children())[-4])
        self.encoder3 = nn.Sequential(*list(_model.children())[-3])

        _gfsize = int(_model.fc.in_features)

        self.global_descriptor = nn.Sequential(
            nn.Conv2d(
                in_channels  = _gfsize,
                out_channels = _gfsize,
                kernel_size  = (3, 3),
                stride       = (1, 1),
                padding      = 0,
                bias         = False
            ),
            nn.AdaptiveMaxPool2d(1)
        )

        # Projectors & Compatibility functions
        if self.attention:
            self.projector = ProjectorBlock(int(_gfsize/2), _gfsize)
            self.attn1 = LinearAttentionBlock(in_features=_gfsize, normalize_attn=normalize_attn)
            self.attn2 = LinearAttentionBlock(in_features=_gfsize, normalize_attn=normalize_attn)

        # final classification layer
        if self.attention:
            self.classify = nn.Linear(in_features=_gfsize*2, out_features=1, bias=True)
        else:
            self.classify = nn.Linear(in_features=_gfsize, out_features=1, bias=True)

        # configure outputs:
        feat_size = _gfsize*2 if self.attention else _gfsize
        classifiers = dict()
        for dd in aux_outputs:
            classifiers[dd['name']] = aux_branch(i_size=feat_size, **dd['args'])

        self.heads = nn.ModuleDict(classifiers)

    def forward(self, x):
        img = x['rgb']

        feats1 = self.encoder1(img)
        feats2 = self.encoder2(feats1)
        feats3 = self.encoder3(feats2)

        g = self.global_descriptor(feats3)

        if self.attention:
            # project features _gfsize/2 -> _gfsize
            feats2 = self.projector(feats2)

            a1, g1 = self.attn1(feats2, g)
            a2, g2 = self.attn2(feats3, g)

            g = torch.cat((g1,g2), dim=1)
        else:
            a1, a2 = None, None

        outputs = dict()
        for name, subnet in self.heads.items():
            outputs[name] = subnet(torch.squeeze(g))

        if self.attention:
            outputs['a1'] = a1
            outputs['a2'] = a2

        return outputs


class AttShuffleNet(BaseModel):
    def __init__(self, backbone_net, aux_input=None, aux_outputs=None, attention=True, normalize_attn=True, pretrained=True):
        super(AttShuffleNet, self).__init__()
        self.attention = attention

        _model = models.__dict__[backbone_net](pretrained)

        self.encoder1 = nn.Sequential(*list(_model.children())[:-3])
        self.encoder2 = nn.Sequential(*list(_model.children())[-3])
        self.encoder3 = nn.Sequential(*list(_model.children())[-2])

        _gfsize = int(_model.fc.in_features)
        _projection_size = int(list(_model.stage4.children())[0].branch1[0].in_channels * 2)

        self.global_descriptor = nn.Sequential(
            nn.Conv2d(
                in_channels  = _gfsize,
                out_channels = _gfsize,
                kernel_size  = (3, 3),
                stride       = (1, 1),
                padding      = 0,
                bias         = False
            ),
            nn.AdaptiveMaxPool2d(1)
        )

        # Projectors & Compatibility functions
        if self.attention:
            self.projector = ProjectorBlock(_projection_size, _gfsize)
            self.attn1 = LinearAttentionBlock(in_features=_gfsize, normalize_attn=normalize_attn)
            self.attn2 = LinearAttentionBlock(in_features=_gfsize, normalize_attn=normalize_attn)

        # final classification layer
        if self.attention:
            self.classify = nn.Linear(in_features=_gfsize*2, out_features=1, bias=True)
        else:
            self.classify = nn.Linear(in_features=_gfsize, out_features=1, bias=True)

        # configure outputs:
        feat_size = _gfsize*2 if self.attention else _gfsize
        classifiers = dict()
        for dd in aux_outputs:
            classifiers[dd['name']] = aux_branch(i_size=feat_size, **dd['args'])

        self.heads = nn.ModuleDict(classifiers)

    def forward(self, x):
        img = x['rgb']

        feats1 = self.encoder1(img)
        feats2 = self.encoder2(feats1)
        feats3 = self.encoder3(feats2)

        g = self.global_descriptor(feats3)

        if self.attention:
            # project features
            feats2 = self.projector(feats2)

            a1, g1 = self.attn1(feats2, g)
            a2, g2 = self.attn2(feats3, g)

            g = torch.cat((g1,g2), dim=1)
        else:
            a1, a2 = None, None

        outputs = dict()
        for name, subnet in self.heads.items():
            outputs[name] = subnet(torch.squeeze(g))

        if self.attention:
            outputs['a1'] = a1
            outputs['a2'] = a2

        return outputs
