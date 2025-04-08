# coding:utf-8
import cv2
import numpy as np
from config_mvsparams import CAMPARAM_INIT
import sys


def get_subwindow(im, bbox, output_sz=None, context=0.5):
    """
    args:
        im: bgr based image
        bbox: (x,y,w,h)
        pos: center position
        avg_chans: channel average
    """
    avg_chans = np.mean(im, axis=(0, 1))
    pos = np.array([bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2])

    size = np.array(bbox[2:])
    w_z = size[0] + context * np.sum(size)
    h_z = size[1] + context * np.sum(size)
    s_z = np.sqrt(w_z * h_z)
    # s_z = s_z * (model_sz / 127)
    original_sz = round(s_z)

    if isinstance(pos, float):
        pos = [pos, pos]
    sz = original_sz
    im_sz = im.shape
    c = (original_sz + 1) / 2
    # context_xmin = round(pos[0] - c) # py2 and py3 round
    context_xmin = np.floor(pos[0] - c + 0.5)
    context_xmax = context_xmin + sz - 1
    # context_ymin = round(pos[1] - c)
    context_ymin = np.floor(pos[1] - c + 0.5)
    context_ymax = context_ymin + sz - 1
    left_pad = int(max(0., -context_xmin))
    top_pad = int(max(0., -context_ymin))
    right_pad = int(max(0., context_xmax - im_sz[1] + 1))
    bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

    context_xmin = context_xmin + left_pad
    context_xmax = context_xmax + left_pad
    context_ymin = context_ymin + top_pad
    context_ymax = context_ymax + top_pad

    r, c, k = im.shape
    if any([top_pad, bottom_pad, left_pad, right_pad]):
        size = (r + top_pad + bottom_pad, c + left_pad + right_pad, k)
        te_im = np.zeros(size, np.uint8)
        te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
        if top_pad:
            te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
        if bottom_pad:
            te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
        if left_pad:
            te_im[:, 0:left_pad, :] = avg_chans
        if right_pad:
            te_im[:, c + left_pad:, :] = avg_chans
        im_patch = te_im[int(context_ymin):int(context_ymax + 1),
                   int(context_xmin):int(context_xmax + 1), :]
    else:
        im_patch = im[int(context_ymin):int(context_ymax + 1),
                   int(context_xmin):int(context_xmax + 1), :]

    if output_sz is not None and not np.array_equal(output_sz, original_sz):
        im_patch = cv2.resize(im_patch, (output_sz, output_sz))
    return im_patch


def cam_init(param, USE_CAM_OR_VIDEO, VIDEODIR, TIMESTAMP=None):
    """ Create camera or video stream"""
    if USE_CAM_OR_VIDEO == 1:  # use cam
        print("[INFO_cam_init] Use camera.")
        camera1 = cv2.VideoCapture(1)
        camera2 = cv2.VideoCapture(0)
    elif USE_CAM_OR_VIDEO == 2:  # use video
        assert TIMESTAMP is not None, "[ERROR_cam_init] TIMESTAMP is undefined!"
        print("[INFO_cam_init] Use video {}.".format(TIMESTAMP))
        CAM1_VIDEO = VIDEODIR + '/cam1video_{}.avi'.format(TIMESTAMP)
        CAM2_VIDEO = VIDEODIR + '/cam2video_{}.avi'.format(TIMESTAMP)
        camera1 = cv2.VideoCapture(CAM1_VIDEO)
        camera2 = cv2.VideoCapture(CAM2_VIDEO)
    else:
        raise ValueError
    if sys.platform == 'win32':
        camera1.set(cv2.CAP_PROP_FRAME_WIDTH, param['FRAME_WIDTH'])
        camera1.set(cv2.CAP_PROP_FRAME_HEIGHT, param['FRAME_HEIGHT'])
        camera2.set(cv2.CAP_PROP_FRAME_WIDTH, param['FRAME_WIDTH'])
        camera2.set(cv2.CAP_PROP_FRAME_HEIGHT, param['FRAME_HEIGHT'])
        cv2.namedWindow("camera1")
        cv2.namedWindow("camera2")
        cv2.moveWindow("camera1", 0, 0)
        cv2.moveWindow("camera2", param['WINDOW_WIDTH'], 0)  # 并排显示
    """==========================="""
    return [camera1, camera2]


def cam_init_6cams(param, USE_CAM_OR_VIDEO, TIMESTAMP=None, folder=None):
    """ Create camera or video stream"""
    if USE_CAM_OR_VIDEO == 2:  # use video
        assert TIMESTAMP is not None, "[ERROR_cam_init] TIMESTAMP is undefined!"
        print("[INFO_cam_init] Use video {}.".format(TIMESTAMP))
        CAM1_VIDEO = r'{}/cam1video_{}.avi'.format(folder, TIMESTAMP)
        CAM2_VIDEO = r'{}/cam2video_{}.avi'.format(folder, TIMESTAMP)
        CAM3_VIDEO = r'{}/cam3video_{}.avi'.format(folder, TIMESTAMP)
        CAM4_VIDEO = r'{}/cam4video_{}.avi'.format(folder, TIMESTAMP)
        CAM5_VIDEO = r'{}/cam5video_{}.avi'.format(folder, TIMESTAMP)
        CAM6_VIDEO = r'{}/cam6video_{}.avi'.format(folder, TIMESTAMP)
        camera1 = cv2.VideoCapture(CAM1_VIDEO)
        camera2 = cv2.VideoCapture(CAM2_VIDEO)
        camera3 = cv2.VideoCapture(CAM3_VIDEO)
        camera4 = cv2.VideoCapture(CAM4_VIDEO)
        camera5 = cv2.VideoCapture(CAM5_VIDEO)
        camera6 = cv2.VideoCapture(CAM6_VIDEO)
    else:
        raise ValueError
    if sys.platform == 'win32':
        camera1.set(cv2.CAP_PROP_FRAME_WIDTH, param['FRAME_WIDTH'])
        camera1.set(cv2.CAP_PROP_FRAME_HEIGHT, param['FRAME_HEIGHT'])
        camera2.set(cv2.CAP_PROP_FRAME_WIDTH, param['FRAME_WIDTH'])
        camera2.set(cv2.CAP_PROP_FRAME_HEIGHT, param['FRAME_HEIGHT'])
        camera3.set(cv2.CAP_PROP_FRAME_WIDTH, param['FRAME_WIDTH'])
        camera3.set(cv2.CAP_PROP_FRAME_HEIGHT, param['FRAME_HEIGHT'])
        camera4.set(cv2.CAP_PROP_FRAME_WIDTH, param['FRAME_WIDTH'])
        camera4.set(cv2.CAP_PROP_FRAME_HEIGHT, param['FRAME_HEIGHT'])
        camera5.set(cv2.CAP_PROP_FRAME_WIDTH, param['FRAME_WIDTH'])
        camera5.set(cv2.CAP_PROP_FRAME_HEIGHT, param['FRAME_HEIGHT'])
        camera6.set(cv2.CAP_PROP_FRAME_WIDTH, param['FRAME_WIDTH'])
        camera6.set(cv2.CAP_PROP_FRAME_HEIGHT, param['FRAME_HEIGHT'])
        cv2.namedWindow("camera1")
        cv2.namedWindow("camera2")
        cv2.namedWindow("camera3")
        cv2.namedWindow("camera4")
        cv2.namedWindow("camera5")
        cv2.namedWindow("camera6")
        cv2.moveWindow("camera1", 0, 0)
        cv2.moveWindow("camera2", param['WINDOW_WIDTH'], 0)
        cv2.moveWindow("camera3", 0, param['WINDOW_HEIGHT'])
        cv2.moveWindow("camera4", param['WINDOW_WIDTH'], param['WINDOW_HEIGHT'])
        cv2.moveWindow("camera5", 0, param['WINDOW_HEIGHT'] * 2)
        cv2.moveWindow("camera6", param['WINDOW_WIDTH'], param['WINDOW_HEIGHT'] * 2)
    """==========================="""
    return [camera1, camera2, camera3, camera4, camera5, camera6]

