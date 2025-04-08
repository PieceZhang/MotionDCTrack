from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse

import cv2
import torch
import numpy as np
from glob import glob

from siamban.core.config import cfg
from siamban.models.model_builder import ModelBuilder
from siamban.tracker.tracker_builder import build_tracker
from siamban.utils.model_load import load_pretrain

import time

from scipy.io import savemat
from utils_mvs import cam_init
from config_mvsparams import CAMPARAM_INIT
from utils_3Dtranslation import XYZEstimation
from my_tracker import MVSTracker

torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='tracking demo')


""" Macro define """
parser.add_argument('--config', default='../experiments/motiondctrack/config.yaml', type=str, help='config file')
parser.add_argument('--snapshot', default='../experiments/motiondctrack/checkpoint.pth', type=str, help='model name')
parser.add_argument('--videodir', default='../demo_video', type=str)
parser.add_argument('--timestamp', default='2023-02-07_23-31-42', type=str, help='time stamp')

args = parser.parse_args()

TIMESTAMP = args.timestamp
VIDEODIR = args.videodir

SYSTEM_DEFINE = 6
param = CAMPARAM_INIT(SYSTEM_DEFINE)
USE_CAM_OR_VIDEO = 2
[camera1, camera2] = cam_init(param, USE_CAM_OR_VIDEO, VIDEODIR, TIMESTAMP)


def get_frames():
    if USE_CAM_OR_VIDEO == 1:
        # cam warmup
        for i in range(10):
            camera1.read()
            camera2.read()
    while True:
        ret, c1fsource = camera1.read()
        ret, c2fsource = camera2.read()
        if ret:
            if SYSTEM_DEFINE == 1:
                c1fsource = cv2.flip(c1fsource, -1)
                c2fsource = cv2.flip(c2fsource, -1)
            c1fsource = cv2.resize(c1fsource, (param['WINDOW_WIDTH'], param['WINDOW_HEIGHT']), interpolation=cv2.INTER_CUBIC)
            c2fsource = cv2.resize(c2fsource, (param['WINDOW_WIDTH'], param['WINDOW_HEIGHT']), interpolation=cv2.INTER_CUBIC)
            c1favg = (np.mean(c1fsource[..., 0]), np.mean(c1fsource[..., 1]), np.mean(c1fsource[..., 2]))  # 使用像素均值填充边缘
            c1fremap = cv2.remap(c1fsource, param['StereoMap'][1][0], param['StereoMap'][1][1],
                                 cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, borderValue=c1favg)  # 立体校正
            c2favg = (np.mean(c2fsource[..., 0]), np.mean(c2fsource[..., 1]), np.mean(c2fsource[..., 2]))  # 使用像素均值填充边缘
            c2fremap = cv2.remap(c2fsource, param['StereoMap'][2][0], param['StereoMap'][2][1],
                                 cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, borderValue=c2favg)  # 立体校正
            yield c1fremap, c2fremap, c1fsource, c2fsource, int(camera1.get(cv2.CAP_PROP_POS_FRAMES))
        else:
            break


def main():
    # load config
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # create model
    model = ModelBuilder()

    # load model
    # model.load_state_dict(torch.load(args.snapshot, map_location=lambda storage, loc: storage.cpu()))
    # model.eval().to(device)
    model = load_pretrain(model, args.snapshot).cuda().eval()

    # build tracker
    tracker1 = build_tracker(model)
    tracker2 = build_tracker(model)

    # set camnum
    tracker1.set_camnum(0)
    tracker2.set_camnum(1)

    # build MVS tracker
    mvstracker = MVSTracker([tracker1, tracker2])
    del tracker1
    del tracker2

    first_frame = True
    video1_name = 'camera1'
    video2_name = 'camera2'

    bboxlog = [[], []]
    xyzlog = [[], [], []]

    ifnewtemplate = False

    estimator = XYZEstimation(param)
    stereosystem = 1

    for camera1_frame, camera2_frame, _c1, _c2, fcount in get_frames():
        if first_frame:
            if ifnewtemplate is False and USE_CAM_OR_VIDEO == 2:
                try:
                    my_first_frame = np.load("my_first_frame_{}.npz".format(TIMESTAMP))
                    camera1_frame = my_first_frame['camera1_frame']
                    camera2_frame = my_first_frame['camera2_frame']
                    init_rect1 = my_first_frame['init_rect1']
                    init_rect2 = my_first_frame['init_rect2']
                    mvstracker.init([camera1_frame, camera2_frame], [init_rect1, init_rect2])
                    del my_first_frame
                    first_frame = False
                    continue
                except FileNotFoundError:
                    pass

            init_rect1 = cv2.selectROI(video1_name, camera1_frame, False, False)
            init_rect2 = cv2.selectROI(video2_name, camera2_frame, False, False)
            if init_rect1[2] == init_rect1[3] == init_rect2[2] == init_rect2[3] == 0:
                stereosystem = 0
            else:
                mvstracker.init([camera1_frame, camera2_frame], [init_rect1, init_rect2])
                stereosystem = 1
            if USE_CAM_OR_VIDEO == 2:
                np.savez("my_first_frame_{}.npz".format(TIMESTAMP), camera1_frame=camera1_frame, camera2_frame=camera2_frame,
                         init_rect1=init_rect1, init_rect2=init_rect2)
            first_frame = False
        else:

            key = cv2.waitKey(1)
            if key == ord("q"):
                break
            elif key == ord("s"):
                init_rect1 = cv2.selectROI(video1_name, camera1_frame, False, False)
                init_rect2 = cv2.selectROI(video2_name, camera2_frame, False, False)
                if init_rect1[2] == init_rect1[3] == init_rect2[2] == init_rect2[3] == 0:
                    stereosystem = 0
                else:
                    mvstracker.init([camera1_frame, camera2_frame], [init_rect1, init_rect2])
                    stereosystem = 1

            if stereosystem == 1:
                # measure the time of each frame
                start = time.time()
                outputs = mvstracker.track([camera1_frame, camera2_frame])
                end = time.time()
                print("Time: {:.2f}ms".format((end - start) * 1000))
            elif stereosystem == 0:  # no target
                X1, Y1, Z, X2, Y2 = 0, 0, 0, 0, 0

            if stereosystem == 1:
                bbox1 = list(map(int, outputs[0]['bbox']))
                cv2.rectangle(camera1_frame, (bbox1[0], bbox1[1]),
                              (bbox1[0] + bbox1[2], bbox1[1] + bbox1[3]),
                              (0, 0, 255), 2)

                bbox2 = list(map(int, outputs[1]['bbox']))
                cv2.rectangle(camera2_frame, (bbox2[0], bbox2[1]),
                              (bbox2[0] + bbox2[2], bbox2[1] + bbox2[3]),
                              (0, 0, 255), 2)

                X1, Y1, Z, X2, Y2 = estimator.estimate(bbox1, bbox2)
                bboxlog[0].append(bbox1)
                bboxlog[1].append(bbox2)

            cv2.imshow(video1_name, camera1_frame)
            cv2.imshow(video2_name, camera2_frame)

            xyzlog[0].append(X1)
            xyzlog[1].append(Y1)
            xyzlog[2].append(Z)

            print("Current frame: {}".format(fcount))

    print("[INFO] Save XYZ: xyzlog_{}.mat".format(TIMESTAMP))
    xyzlog = np.array(xyzlog).transpose()
    savemat("xyzlog_{}.mat".format(TIMESTAMP), {'xyz': xyzlog})

    print("[INFO] Save bbox: bboxlog_{}.npy".format(TIMESTAMP))
    np.save("bboxlog_{}.npy".format(TIMESTAMP), bboxlog)


if __name__ == '__main__':
    main()
