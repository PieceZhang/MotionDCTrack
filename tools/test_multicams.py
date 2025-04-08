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
from utils_mvs import cam_init_6cams
from config_mvsparams import CAMPARAM_INIT
from utils_3Dtranslation import XYZEstimation
from my_tracker import MVSTracker

torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='tracking demo')

""" Macro define """
parser.add_argument('--config', default='../experiments/motiondctrack/config.yaml', type=str, help='config file')
parser.add_argument('--snapshot', default='../experiments/motiondctrack/checkpoint.pth', type=str, help='model name')
parser.add_argument('--videodir', default='', type=str)
parser.add_argument('--timestamp', default='', type=str, help='time stamp')

args = parser.parse_args()

TIMESTAMP = args.timestamp
VIDEODIR = args.videodir

param = CAMPARAM_INIT(SYSTEM_DEFINE=6)
param2 = CAMPARAM_INIT(SYSTEM_DEFINE=7)
param3 = CAMPARAM_INIT(SYSTEM_DEFINE=8)
USE_CAM_OR_VIDEO = 2
[camera1, camera2, camera3, camera4, camera5, camera6] = cam_init_6cams(param, USE_CAM_OR_VIDEO, TIMESTAMP, VIDEODIR)


def get_frames():
    while True:
        ret, c1fsource = camera1.read()
        ret, c2fsource = camera2.read()
        ret, c3fsource = camera3.read()
        ret, c4fsource = camera4.read()
        ret, c5fsource = camera5.read()
        ret, c6fsource = camera6.read()
        if ret:
            c1fsource = cv2.resize(c1fsource, (param['WINDOW_WIDTH'], param['WINDOW_HEIGHT']), interpolation=cv2.INTER_CUBIC)
            c2fsource = cv2.resize(c2fsource, (param['WINDOW_WIDTH'], param['WINDOW_HEIGHT']), interpolation=cv2.INTER_CUBIC)
            c1favg = (np.mean(c1fsource[..., 0]), np.mean(c1fsource[..., 1]), np.mean(c1fsource[..., 2]))
            c1fremap = cv2.remap(c1fsource, param['StereoMap'][1][0], param['StereoMap'][1][1],
                                 cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, borderValue=c1favg)
            c2favg = (np.mean(c2fsource[..., 0]), np.mean(c2fsource[..., 1]), np.mean(c2fsource[..., 2]))
            c2fremap = cv2.remap(c2fsource, param['StereoMap'][2][0], param['StereoMap'][2][1],
                                 cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, borderValue=c2favg)

            c3fsource = cv2.resize(c3fsource, (param2['WINDOW_WIDTH'], param2['WINDOW_HEIGHT']), interpolation=cv2.INTER_CUBIC)
            c4fsource = cv2.resize(c4fsource, (param2['WINDOW_WIDTH'], param2['WINDOW_HEIGHT']), interpolation=cv2.INTER_CUBIC)
            c3favg = (np.mean(c3fsource[..., 0]), np.mean(c3fsource[..., 1]), np.mean(c3fsource[..., 2]))
            c3fremap = cv2.remap(c3fsource, param2['StereoMap'][1][0], param2['StereoMap'][1][1],
                                 cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, borderValue=c3favg)
            c4favg = (np.mean(c4fsource[..., 0]), np.mean(c4fsource[..., 1]), np.mean(c4fsource[..., 2]))
            c4fremap = cv2.remap(c4fsource, param2['StereoMap'][2][0], param2['StereoMap'][2][1],
                                 cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, borderValue=c4favg)

            c5fsource = cv2.resize(c5fsource, (param3['WINDOW_WIDTH'], param3['WINDOW_HEIGHT']), interpolation=cv2.INTER_CUBIC)
            c6fsource = cv2.resize(c6fsource, (param3['WINDOW_WIDTH'], param3['WINDOW_HEIGHT']), interpolation=cv2.INTER_CUBIC)
            c5favg = (np.mean(c5fsource[..., 0]), np.mean(c5fsource[..., 1]), np.mean(c5fsource[..., 2]))
            c5fremap = cv2.remap(c5fsource, param3['StereoMap'][1][0], param3['StereoMap'][1][1],
                                 cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, borderValue=c5favg)
            c6favg = (np.mean(c6fsource[..., 0]), np.mean(c6fsource[..., 1]), np.mean(c6fsource[..., 2]))
            c6fremap = cv2.remap(c6fsource, param3['StereoMap'][2][0], param3['StereoMap'][2][1],
                                 cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, borderValue=c6favg)

            yield c1fremap, c2fremap, c3fremap, c4fremap, c5fremap, c6fremap, int(camera1.get(cv2.CAP_PROP_POS_FRAMES))
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
    tracker3 = build_tracker(model)
    tracker4 = build_tracker(model)
    tracker5 = build_tracker(model)
    tracker6 = build_tracker(model)

    # set camnum
    tracker1.set_camnum(0)
    tracker2.set_camnum(1)
    tracker3.set_camnum(2)
    tracker4.set_camnum(3)
    tracker5.set_camnum(4)
    tracker6.set_camnum(5)

    # build MVS tracker
    mvstracker = MVSTracker([tracker1, tracker2])
    mvstracker2 = MVSTracker([tracker3, tracker4])
    mvstracker3 = MVSTracker([tracker5, tracker6])
    del tracker1
    del tracker2
    del tracker3
    del tracker4
    del tracker5
    del tracker6

    first_frame = True
    video1_name = 'camera1'
    video2_name = 'camera2'
    video3_name = 'camera3'
    video4_name = 'camera4'
    video5_name = 'camera5'
    video6_name = 'camera6'

    bboxlog = [[], [], []]  # bbox1, bbox2, systemNo
    xyzlog = [[], [], [], []]  # X, Y, Z, systemNo

    ifnewtemplate = False

    estimator1 = XYZEstimation(param)
    estimator2 = XYZEstimation(param2)
    estimator3 = XYZEstimation(param3)
    stereosystem = 1
    ifstill = False

    for camera1_frame, camera2_frame, camera3_frame, camera4_frame, camera5_frame, camera6_frame, fcount in get_frames():
        if first_frame:
            init_rect1 = cv2.selectROI(video1_name, camera1_frame, False, False)
            init_rect2 = cv2.selectROI(video2_name, camera2_frame, False, False)
            if init_rect1[2] == init_rect1[3] == init_rect2[2] == init_rect2[3] == 0:
                stereosystem = 0
            else:
                mvstracker.init([camera1_frame, camera2_frame], [init_rect1, init_rect2])
            first_frame = False
        else:
            key = cv2.waitKey(1)
            if key == ord("q"):
                break
            elif key == ord("w"):
                ifstill = True if not ifstill else False
            elif key == ord("1"):  # switch to another cam
                stereosystem = 1
                init_rect1 = cv2.selectROI(video1_name, camera1_frame, False, False)
                init_rect2 = cv2.selectROI(video2_name, camera2_frame, False, False)
                if init_rect1[2] == init_rect1[3] == init_rect2[2] == init_rect2[3] == 0:
                    stereosystem = 0
                else:
                    if ifstill:
                        bbox1, bbox2 = list(init_rect1), list(init_rect2)
                    mvstracker.init([camera1_frame, camera2_frame], [init_rect1, init_rect2])
            elif key == ord("2"):
                stereosystem = 2
                init_rect3 = cv2.selectROI(video3_name, camera3_frame, False, False)
                init_rect4 = cv2.selectROI(video4_name, camera4_frame, False, False)
                if init_rect3[2] == init_rect3[3] == init_rect4[2] == init_rect4[3] == 0:
                    stereosystem = 0
                else:
                    if ifstill:
                        bbox3, bbox4 = list(init_rect3), list(init_rect4)
                    mvstracker2.init([camera3_frame, camera4_frame], [init_rect3, init_rect4])
            elif key == ord("3"):
                stereosystem = 3
                init_rect5 = cv2.selectROI(video5_name, camera5_frame, False, False)
                init_rect6 = cv2.selectROI(video6_name, camera6_frame, False, False)
                if init_rect5[2] == init_rect5[3] == init_rect6[2] == init_rect6[3] == 0:
                    stereosystem = 0
                else:
                    if ifstill:
                        bbox5, bbox6 = list(init_rect5), list(init_rect6)
                    mvstracker3.init([camera5_frame, camera6_frame], [init_rect5, init_rect6])
            elif key == ord("0"):
                ifstill = False
                stereosystem = 0

            if stereosystem == 1:
                if not ifstill:
                    outputs = mvstracker.track([camera1_frame, camera2_frame])
                    bbox1 = list(map(int, outputs[0]['bbox']))
                    bbox2 = list(map(int, outputs[1]['bbox']))
                cv2.rectangle(camera1_frame, (bbox1[0], bbox1[1]),
                              (bbox1[0] + bbox1[2], bbox1[1] + bbox1[3]),
                              (0, 255, 0), 3)
                cv2.rectangle(camera2_frame, (bbox2[0], bbox2[1]),
                              (bbox2[0] + bbox2[2], bbox2[1] + bbox2[3]),
                              (0, 255, 0), 3)
                bboxlog[0].append(bbox1)
                bboxlog[1].append(bbox2)
                bboxlog[2].append(stereosystem)
                X1, Y1, Z, X2, Y2 = estimator1.estimate(bbox1, bbox2)
            elif stereosystem == 2:
                if not ifstill:
                    outputs = mvstracker2.track([camera3_frame, camera4_frame])
                    bbox3 = list(map(int, outputs[0]['bbox']))
                    bbox4 = list(map(int, outputs[1]['bbox']))
                cv2.rectangle(camera3_frame, (bbox3[0], bbox3[1]),
                              (bbox3[0] + bbox3[2], bbox3[1] + bbox3[3]),
                              (0, 255, 0), 3)
                cv2.rectangle(camera4_frame, (bbox4[0], bbox4[1]),
                              (bbox4[0] + bbox4[2], bbox4[1] + bbox4[3]),
                              (0, 255, 0), 3)
                bboxlog[0].append(bbox3)
                bboxlog[1].append(bbox4)
                bboxlog[2].append(stereosystem)
                X1, Y1, Z, X2, Y2 = estimator2.estimate(bbox3, bbox4)
            elif stereosystem == 3:
                if not ifstill:
                    outputs = mvstracker3.track([camera5_frame, camera6_frame])
                    bbox5 = list(map(int, outputs[0]['bbox']))
                    bbox6 = list(map(int, outputs[1]['bbox']))
                cv2.rectangle(camera5_frame, (bbox5[0], bbox5[1]),
                              (bbox5[0] + bbox5[2], bbox5[1] + bbox5[3]),
                              (0, 255, 0), 3)
                cv2.rectangle(camera6_frame, (bbox6[0], bbox6[1]),
                              (bbox6[0] + bbox6[2], bbox6[1] + bbox6[3]),
                              (0, 255, 0), 3)
                bboxlog[0].append(bbox5)
                bboxlog[1].append(bbox6)
                bboxlog[2].append(stereosystem)
                X1, Y1, Z, X2, Y2 = estimator3.estimate(bbox5, bbox6)
            elif stereosystem == 0:  # no target
                bboxlog[0].append([0, 0, 0, 0])
                bboxlog[1].append([0, 0, 0, 0])
                bboxlog[2].append(stereosystem)
                X1, Y1, Z, X2, Y2 = 0, 0, 0, 0, 0

            xyzlog[0].append(X1)
            xyzlog[1].append(Y1)
            xyzlog[2].append(Z)
            xyzlog[3].append(stereosystem)

            cv2.imshow(video1_name, camera1_frame)
            cv2.imshow(video2_name, camera2_frame)
            cv2.imshow(video3_name, camera3_frame)
            cv2.imshow(video4_name, camera4_frame)
            cv2.imshow(video5_name, camera5_frame)
            cv2.imshow(video6_name, camera6_frame)

            print("Current frame: {},\t system: {}".format(fcount, stereosystem))


if __name__ == '__main__':
    main()
