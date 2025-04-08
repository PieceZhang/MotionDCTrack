# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import logging
import sys
import os

import cv2
import numpy as np
from torch.utils.data import Dataset
from scipy import interpolate

from siamban.utils.bbox import center2corner, Center, Corner, corner2center
from siamban.datasets.point_target import PointTarget
from siamban.datasets.augmentation import Augmentation, Augmentation2
from siamban.core.config import cfg
from siamban.utils.motion_utils import motion_normalize

logger = logging.getLogger("global")

# setting opencv
pyv = sys.version[0]
if pyv[0] == '3':
    cv2.ocl.setUseOpenCL(False)


def auxcam_num(x):
    if x == '1':
        return '2'
    elif x == '2':
        return '1'
    elif x == '3':
        return '4'
    elif x == '4':
        return '3'
    elif x == '5':
        return '6'
    elif x == '6':
        return '5'


class SubDataset(object):
    def __init__(self, name, root, anno, frame_range, num_use, start_idx):
        cur_path = os.path.dirname(os.path.realpath(__file__))
        self.name = name
        self.root = os.path.join(cur_path, '../../', root)
        self.anno = os.path.join(cur_path, '../../', anno)
        self.frame_range = frame_range
        self.num_use = num_use
        self.start_idx = start_idx
        logger.info("loading " + name)
        with open(self.anno, 'r') as f:
            meta_data = json.load(f)
            meta_data = self._camera_match(meta_data)
            meta_data = self._filter_zero(meta_data)

        for video in list(meta_data.keys()):
            for track in meta_data[video]:
                frames = meta_data[video][track]
                frames = list(map(int,
                              filter(lambda x: x.isdigit(), frames.keys())))
                frames.sort()
                meta_data[video][track]['frames'] = frames
                if len(frames) <= 0:
                    logger.warning("{}/{} has no frames".format(video, track))
                    del meta_data[video][track]

        for video in list(meta_data.keys()):
            if len(meta_data[video]) <= 0:
                logger.warning("{} has no tracks".format(video))
                del meta_data[video]

        self.labels = meta_data
        self.num = len(self.labels)
        self.num_use = self.num if self.num_use == -1 else self.num_use
        self.videos = list(meta_data.keys())
        logger.info("{} loaded".format(self.name))
        self.path_format = '{}.{}.{}.jpg'
        self.pick = self.shuffle()

        # for video select
        self.lut_video = {}
        for video in self.videos:
            self.lut_video[video] = video[20:30]
        tlist = list(self.lut_video.values())
        tset = set(tlist)
        self.lut_time = {}
        for t in tset:
            self.lut_time[t] = []
        for i, t in enumerate(tlist):
            self.lut_time[t].append(self.videos[i])

    def _filter_zero(self, meta_data):
        meta_data_new = {}
        for video, tracks in meta_data.items():
            new_tracks = {}
            for trk, frames in tracks.items():
                new_frames = {}
                for frm, bbox in frames.items():
                    if not isinstance(bbox, dict):
                        if len(bbox) == 4:
                            x1, y1, x2, y2 = bbox
                            w, h = x2 - x1, y2 - y1
                        else:
                            w, h = bbox
                        if w <= 0 or h <= 0:
                            continue
                    new_frames[frm] = bbox
                if len(new_frames) > 0:
                    new_tracks[trk] = new_frames
            if len(new_tracks) > 0:
                meta_data_new[video] = new_tracks
        return meta_data_new

    def _camera_match(self, meta_data):
        # 匹配来自多个相机同个视频的图像
        for video, tracks in meta_data.items():
            video_ = video[:13] + auxcam_num(video[13]) + video[14:]
            tracks = tracks['00']
            for frm, bbox in tracks.items():
                if bbox == [0.0, 0.0, 0.0, 0.0]:
                    meta_data[video_]['00'][frm] = bbox
        return meta_data

    def log(self):
        logger.info("{} start-index {} select [{}/{}] path_format {}".format(
            self.name, self.start_idx, self.num_use,
            self.num, self.path_format))

    def shuffle(self):
        lists = list(range(self.start_idx, self.start_idx + self.num))
        pick = []
        while len(pick) < self.num_use:
            np.random.shuffle(lists)
            pick += lists
        return pick[:self.num_use]

    def get_image_anno(self, video, track, frame):
        frame = "{:06d}".format(frame)
        image_path = os.path.join(self.root, video,
                                  self.path_format.format(frame, track, 'x'))
        image_anno = self.labels[video][track][frame]
        return image_path, image_anno

    def get_positive_pair(self, index):
        video_name = self.videos[index]
        video = self.labels[video_name]
        wlist = []
        hlist = []
        for item in self.labels.values():
            if 'frames' in item['00'].keys():
                del item['00']['frames']
            for box in item['00'].values():
                w = box[2] - box[0]
                h = box[3] - box[1]
                wlist.append(w)
                hlist.append(h)
        w = np.mean(wlist)
        h = np.mean(hlist)
        track = np.random.choice(list(video.keys()))
        track_info = video[track]
        frames = track_info['frames']
        template_frame = np.random.choice(frames)

        video_name_ = np.random.choice(self.lut_time[self.lut_video[video_name]])
        video_ = self.labels[video_name_]
        track_ = np.random.choice(list(video_.keys()))
        track_info_ = video_[track_]
        frames_ = track_info_['frames']
        search_frame = np.random.choice(frames_)

        video_name_aux = video_name_[:13] + auxcam_num(video_name_[13]) + video_name_[14:]

        return self.get_image_anno(video_name, track, template_frame), \
            self.get_image_anno(video_name_, track_, search_frame), \
            self.get_image_anno(video_name_aux, track_, search_frame)

    def get_neg_pair(self, index=-1):
        if index == -1:
            index = np.random.randint(0, self.num)
        video_name = self.videos[index]
        video = self.labels[video_name]
        track = np.random.choice(list(video.keys()))
        track_info = video[track]
        frames = track_info['frames']
        template_frame = np.random.choice(frames)

        negt = list(self.lut_time.keys())
        # negt.remove(self.lut_video[video_name])  # for neg sample, select main and aux image from different camera with z
        video_name_ = np.random.choice(self.lut_time[np.random.choice(negt)])
        video_ = self.labels[video_name_]
        track_ = np.random.choice(list(video_.keys()))
        track_info_ = video_[track_]
        frames_ = track_info_['frames']
        search_frame = np.random.choice(frames_)

        # video_name_aux = video_name_[:13] + auxcam_num(video_name_[13]) + video_name_[14:]
        video_name_aux = np.random.choice(self.lut_time[np.random.choice(negt)])
        video_aux = self.labels[video_name_aux]
        track_aux = np.random.choice(list(video_aux.keys()))
        track_info_aux = video_aux[track_aux]
        frames_aux = track_info_aux['frames']
        search_frame_aux = np.random.choice(frames_aux)

        return self.get_image_anno(video_name, track, template_frame), \
            self.get_image_anno(video_name_, track_, search_frame), \
            self.get_image_anno(video_name_aux, track_aux, search_frame_aux)

    def get_historical_motion(self, index, neg=False):  # TODO neg
        video_name = self.videos[index]
        video = self.labels[video_name]
        track = np.random.choice(list(video.keys()))
        track_info = video[track]
        frames = track_info['frames']
        template_frame = np.random.choice(frames)

        start_frame_bool = False
        _ct = 0
        n = cfg.Motion.KWARGS.n + 1
        histm = []
        while template_frame >= 0 and n > 0:
            try:
                histm.append(self.get_image_anno(video_name, track, template_frame)[-1])
            except KeyError:  # in case of target disappear for several frames
                histm.append(histm[-1])
                _ct += 1
            template_frame -= 1
            n -= 1
        if n > 0 or _ct > len(histm)*0.5:
            start_frame_bool = True
        while n > 0:  # if n != 0 and template_frame == 0, repeat the last frame
            histm.append(histm[-1])
            template_frame -= 1
            n -= 1
        histm.reverse()  # [former, ..., later, nearest]
        return histm, start_frame_bool

    def get_random_target(self, index=-1):
        if index == -1:
            index = np.random.randint(0, self.num)
        video_name = self.videos[index]
        video = self.labels[video_name]
        track = np.random.choice(list(video.keys()))
        track_info = video[track]
        frames = track_info['frames']
        frame = np.random.choice(frames)
        return self.get_image_anno(video_name, track, frame)

    def __len__(self):
        return self.num


class BANDataset(Dataset):
    def __init__(self,):
        super(BANDataset, self).__init__()

        # desired_size = (cfg.TRAIN.SEARCH_SIZE - cfg.TRAIN.EXEMPLAR_SIZE) / \
        #     cfg.POINT.STRIDE + 1 + cfg.TRAIN.BASE_SIZE
        # if desired_size != cfg.TRAIN.OUTPUT_SIZE:
        #     raise Exception('size not match!')

        # create point target
        self.point_target = PointTarget()

        # create sub dataset
        self.all_dataset = []
        start = 0
        self.num = 0
        for name in cfg.DATASET.NAMES:
            subdata_cfg = getattr(cfg.DATASET, name)
            sub_dataset = SubDataset(
                    name,
                    subdata_cfg.ROOT,
                    subdata_cfg.ANNO,
                    subdata_cfg.FRAME_RANGE,
                    subdata_cfg.NUM_USE,
                    start
                )
            start += sub_dataset.num
            self.num += sub_dataset.num_use

            sub_dataset.log()
            self.all_dataset.append(sub_dataset)

        # data augmentation
        self.template_aug = Augmentation(
                cfg.DATASET.TEMPLATE.SHIFT,
                cfg.DATASET.TEMPLATE.SCALE,
                cfg.DATASET.TEMPLATE.BLUR,
                cfg.DATASET.TEMPLATE.FLIP,
                cfg.DATASET.TEMPLATE.COLOR
            )
        self.search_aug = Augmentation2(
                cfg.DATASET.SEARCH.SHIFT,
                cfg.DATASET.SEARCH.SCALE,
                cfg.DATASET.SEARCH.BLUR,
                cfg.DATASET.SEARCH.FLIP,
                cfg.DATASET.SEARCH.COLOR
            )
        # data augmentation for neg sample
        self.template_aug_neg = Augmentation(
                cfg.DATASET.TEMPLATE.SHIFT * 2,
                cfg.DATASET.TEMPLATE.SCALE * 2,
                cfg.DATASET.TEMPLATE.BLUR * 2,
                cfg.DATASET.TEMPLATE.FLIP * 2,
                cfg.DATASET.TEMPLATE.COLOR * 2
            )
        self.search_aug_neg = Augmentation2(
                cfg.DATASET.SEARCH.SHIFT * 8,  # larger shift, smaller scale for neg sample
                cfg.DATASET.SEARCH.SCALE * 1,
                cfg.DATASET.SEARCH.BLUR * 4,
                cfg.DATASET.SEARCH.FLIP * 6,
                cfg.DATASET.SEARCH.COLOR * 4
            )
        videos_per_epoch = cfg.DATASET.VIDEOS_PER_EPOCH
        self.num = videos_per_epoch if videos_per_epoch > 0 else self.num
        self.num *= cfg.TRAIN.EPOCH
        self.pick = self.shuffle()

    def shuffle(self):
        pick = []
        m = 0
        while m < self.num:
            p = []
            for sub_dataset in self.all_dataset:
                sub_p = sub_dataset.pick
                p += sub_p
            np.random.shuffle(p)
            pick += p
            m = len(pick)
        logger.info("shuffle done!")
        logger.info("dataset length {}".format(self.num))
        return pick[:self.num]

    def _find_dataset(self, index):
        for dataset in self.all_dataset:
            if dataset.start_idx + dataset.num > index:
                return dataset, index - dataset.start_idx

    def _get_bbox(self, image, shape):
        imh, imw = image.shape[:2]
        if len(shape) == 4:
            w, h = shape[2]-shape[0], shape[3]-shape[1]
        else:
            w, h = shape
        context_amount = 0.5
        exemplar_size = cfg.TRAIN.EXEMPLAR_SIZE
        wc_z = w + context_amount * (w+h)
        hc_z = h + context_amount * (w+h)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = exemplar_size / s_z
        w = w*scale_z
        h = h*scale_z
        cx, cy = imw//2, imh//2
        bbox = center2corner(Center(cx, cy, w, h))
        return bbox

    def _histmotion_process(self, histm):
        mu = cfg.Motion.KWARGS.Gauss_mu
        sigma = cfg.Motion.KWARGS.Gauss_sigma
        n = cfg.Motion.KWARGS.n + 1
        n_origin = cfg.Motion.KWARGS.n
        s = cfg.Motion.KWARGS.s

        for i in range(len(histm)):
            histm[i] = corner2center(Corner(histm[i][0], histm[i][1], histm[i][2], histm[i][3]))

        histm = np.array(histm)
        f0 = interpolate.interp1d(np.linspace(0, n-1, n), histm[:, 0], kind='cubic')
        f1 = interpolate.interp1d(np.linspace(0, n-1, n), histm[:, 1], kind='cubic')
        f2 = interpolate.interp1d(np.linspace(0, n-1, n), histm[:, 2], kind='cubic')
        f3 = interpolate.interp1d(np.linspace(0, n-1, n), histm[:, 3], kind='cubic')

        cx = f0(np.linspace(0, n-1, n*s))
        deltax = np.diff(cx)[-n_origin*s:]
        deltax += np.random.normal(mu, sigma/10, deltax.shape)  # smaller sigma

        cy = f1(np.linspace(0, n-1, n*s))
        deltay = np.diff(cy)[-n_origin*s:]
        deltay += np.random.normal(mu, sigma/10, deltay.shape)

        w = f2(np.linspace(0, n-1, n*s))[-n_origin*s:]
        w += np.random.normal(mu, sigma, w.shape)

        h = f3(np.linspace(0, n-1, n*s))[-n_origin*s:]
        h += np.random.normal(mu, sigma, h.shape)

        histm = np.stack([deltax, deltay, w, h], axis=1).astype(np.float32)
        # histm += np.random.normal(mu, sigma, histm.shape)

        return histm

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        index = self.pick[index]
        dataset, index = self._find_dataset(index)

        gray = cfg.DATASET.GRAY and cfg.DATASET.GRAY > np.random.random()
        neg = cfg.DATASET.NEG and cfg.DATASET.NEG > np.random.random()

        # get one dataset
        if neg:
            template, search, search_aux = dataset.get_neg_pair()
        else:
            template, search, search_aux = dataset.get_positive_pair(index)

        # get image
        template_image = cv2.imread(template[0].replace(template[0][-9:], '/000000.00.x.jpg'))
        search_image = cv2.imread(search[0].replace(search[0][-9:], '/000000.00.x.jpg'))
        search_aux_image = cv2.imread(search_aux[0].replace(search_aux[0][-9:], '/000000.00.x.jpg'))

        # get bounding box
        template_box = self._get_bbox(template_image, template[1])
        search_box = self._get_bbox(search_image, search[1])
        search_aux_box = self._get_bbox(search_aux_image, search_aux[1])

        # get historical motion for MT
        historical_motion = None
        if cfg.Motion.Motion:
            historical_motion, start_frame = dataset.get_historical_motion(index, neg)
            historical_motion = self._histmotion_process(historical_motion)

        # augmentation
        if neg:
            template, _ = self.template_aug_neg(template_image,
                                                template_box,
                                                cfg.TRAIN.EXEMPLAR_SIZE,
                                                gray=gray,
                                                neg=neg)
            imagelist, bboxlist = self.search_aug_neg([search_image, search_aux_image],
                                                  [search_box, search_aux_box],
                                                  cfg.TRAIN.SEARCH_SIZE,
                                                  gray=gray,
                                                  neg=neg)
        else:
            template, _ = self.template_aug(template_image,
                                            template_box,
                                            cfg.TRAIN.EXEMPLAR_SIZE,
                                            gray=gray)
            imagelist, bboxlist = self.search_aug([search_image, search_aux_image],
                                                  [search_box, search_aux_box],
                                                  cfg.TRAIN.SEARCH_SIZE,
                                                  gray=gray,
                                                  histm=historical_motion,
                                                  start_frame=start_frame)  # random param is consistent for search and search_aux

        search = imagelist[0]
        search_aux = imagelist[1]
        bbox = bboxlist[0]
        bbox_aux = bboxlist[1]

        # debug
        # cv2.imshow('template_image', template_image)
        # cv2.imshow('search_image', search_image)
        # cv2.imshow('search_aux_image', search_aux_image)
        # cv2.imshow('template', template.astype('uint8'))
        # cv2.imshow('search', search.astype('uint8'))
        # cv2.imshow('search_aux', search_aux.astype('uint8'))
        # print(neg)
        # cv2.waitKey(0)

        # get labels
        cls, delta = self.point_target(bbox, cfg.TRAIN.OUTPUT_SIZE, neg)
        cls_aux, delta_aux = self.point_target(bbox_aux, cfg.TRAIN.OUTPUT_SIZE, neg)

        template = template.transpose((2, 0, 1)).astype(np.float32)
        search = search.transpose((2, 0, 1)).astype(np.float32)
        search_aux = search_aux.transpose((2, 0, 1)).astype(np.float32)
        if historical_motion is not None:
            historical_motion = motion_normalize(historical_motion)
        return {
                'template': template,
                'search': search,
                'search_aux': search_aux,
                'label_cls': cls,
                'label_loc': delta,
                'label_cls_aux': cls_aux,
                'label_loc_aux': delta_aux,
                'bbox': np.array(bbox),
                'bbox_aux': np.array(bbox_aux),
                'historical_motion': historical_motion
                }
