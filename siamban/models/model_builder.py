# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

from siamban.core.config import cfg
from siamban.models.loss import select_cross_entropy_loss, select_iou_loss
from siamban.models.backbone import get_backbone
from siamban.models.head import get_ban_head, get_aux_head, get_motion_tokenizer
from siamban.models.neck import get_neck


class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)

        # build adjust layer
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)

        # build ban head
        if cfg.BAN.BAN:
            self.head = get_ban_head(cfg.BAN.TYPE,
                                     **cfg.BAN.KWARGS)

        if cfg.Motion.Motion:
            self.mt = get_motion_tokenizer(cfg.Motion.TYPE,
                                           **cfg.Motion.KWARGS)

        if cfg.AuxBranch.AuxBranch:
            self.auxattn = get_aux_head(cfg.AuxBranch.TYPE,
                                          **cfg.AuxBranch.KWARGS)
            if cfg.AuxBranch.DualAttn:
                self.mainattn = get_aux_head(cfg.AuxBranch.TYPE,
                                          **cfg.AuxBranch.KWARGS)
            else:
                self.mainattn = None

        self.outlist = {'main': None, 'aux': None}
        self.xflist = {'main': None, 'aux': None, 'mt_main': None, 'mt_aux': None}

    def template(self, z):
        # use main cam frame as template
        zf = self.backbone(z)
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
        self.zf = zf

    def track_add_frame(self, x, mode):
        xf = self.backbone(x)
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)
        self.xflist[mode] = xf

    def track_add_motion(self, mt, mode):
        self.xflist[f"mt_{mode}"] = torch.tensor(mt).cuda()

    def track_calc(self):
        if cfg.AuxBranch.AuxBranch:  # be aware of the order, keep consistent with the order in forward
            mt_main = None
            mt_aux = None
            if cfg.Motion.Motion:
                mt_main = self.mt(self.xflist['mt_main'])
                mt_aux = mt_main

            if not cfg.AuxBranch.DualAttn:
                xf, xauxf, zf = self.xflist['main'], self.xflist['aux'], self.zf
                cls, loc = self.head(zf, xf, mt_main)
                self.outlist['main'] = {'cls': cls, 'loc': loc}
                zf = self.auxattn(xf, xauxf, zf)
                cls, loc = self.head(zf, xauxf, mt_aux)
                self.outlist['aux'] = {'cls': cls, 'loc': loc}
            else:
                xf, xauxf, zf = self.xflist['main'], self.xflist['aux'], self.zf
                zf_withxaux = self.mainattn(xauxf, xf, zf)  # integrate xauxf information into zf
                cls, loc = self.head(zf_withxaux, xf, mt_main)
                self.outlist['main'] = {'cls': cls, 'loc': loc}
                zf_withx = self.auxattn(xf, xauxf, zf)  # integrate xf information into zf
                cls, loc = self.head(zf_withx, xauxf, mt_aux)
                self.outlist['aux'] = {'cls': cls, 'loc': loc}
        else:
            raise NotImplementedError

    def track_output(self, mode):
        return self.outlist[mode]

    def log_softmax(self, cls):
        if cfg.BAN.BAN:
            cls = cls.permute(0, 2, 3, 1).contiguous()
            cls = F.log_softmax(cls, dim=3)
        return cls

    def forward(self, data):
        """ only used in training
        """
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['label_loc'].cuda()
        historical_motion = data['historical_motion'].cuda()

        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
            xf = self.neck(xf)

        # motion tokenizer
        mt_main = None
        mt_aux = None
        if cfg.Motion.Motion:
            mt_main = self.mt(historical_motion)  # (B, L=n*s, 4) -> transpose -> linear -> (B, 4, C=256)
            mt_aux = mt_main

        if cfg.AuxBranch.AuxBranch:
            search_aux = data['search_aux'].cuda()
            xauxf = self.backbone(search_aux)
            if cfg.ADJUST.ADJUST:
                xauxf = self.neck(xauxf)

            # main branch first
            if cfg.AuxBranch.DualAttn:
                zf_withxaux = self.mainattn(xauxf, xf, zf)  # integrate xauxf information into zf
                cls, loc = self.head(zf_withxaux, xf, mt_main)
            else:
                cls, loc = self.head(zf, xf, mt_main)
            # get loss
            cls = self.log_softmax(cls)
            cls_loss = select_cross_entropy_loss(cls, label_cls)
            loc_loss = select_iou_loss(loc, label_loc, label_cls)

            # aux branch
            zf_withx = self.auxattn(xf, xauxf, zf)  # integrate xf information into zf
            cls_aux, loc_aux = self.head(zf_withx, xauxf, mt_aux)
            # get loss
            cls_aux = self.log_softmax(cls_aux)
            label_cls_aux = data['label_cls_aux'].cuda()
            label_loc_aux = data['label_loc_aux'].cuda()
            cls_loss_aux = select_cross_entropy_loss(cls_aux, label_cls_aux)
            loc_loss_aux = select_iou_loss(loc_aux, label_loc_aux, label_cls_aux)
        else:
            raise NotImplementedError

        if cfg.AuxBranch.AuxBranch:
            cls_loss = (cls_loss + cls_loss_aux) / 2
            loc_loss = (loc_loss + loc_loss_aux) / 2

        outputs = {}
        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
            cfg.TRAIN.LOC_WEIGHT * loc_loss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss

        return outputs
