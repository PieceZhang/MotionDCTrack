# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from siamban.models.backbone.resnet_atrous import resnet18, resnet34, resnet50

BACKBONES = {
              'resnet18': resnet18,
              'resnet34': resnet34,
              'resnet50': resnet50,
            }


def get_backbone(name, **kwargs):
    if name in BACKBONES.keys():
        return BACKBONES[name](**kwargs)
    else:
        raise KeyError('Unknown backbone: {}'.format(name))
