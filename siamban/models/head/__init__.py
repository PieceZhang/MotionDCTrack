from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from siamban.models.head.transhead import TransHead
from siamban.models.head.CCA import CCA2
from siamban.models.head.motion import MambaMT2

BANS = {
        'TransHead': TransHead,
       }


def get_ban_head(name, **kwargs):
    return BANS[name](**kwargs)


AUX = {
        'CCA2': CCA2
       }


def get_aux_head(name, **kwargs):
    return AUX[name](**kwargs)


MOTION = {
        'MambaMT2': MambaMT2
       }


def get_motion_tokenizer(name, **kwargs):
    return MOTION[name](**kwargs)
