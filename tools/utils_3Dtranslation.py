from numpy import cos, sin
import numpy as np
import cv2


def rotateMat(aX, aY, aZ):
    aX = aX * np.pi / 180  # 弧度
    aY = aY * np.pi / 180
    aZ = aZ * np.pi / 180
    RotMat = np.array([[1, 0, 0], [0, cos(aX), -sin(aX)], [0, sin(aX), cos(aX)]]).dot(
        np.array([[cos(aY), 0, sin(aY)], [0, 1, 0], [-sin(aY), 0, cos(aY)]])).dot(
        np.array([[cos(aZ), -sin(aZ), 0], [sin(aZ), cos(aZ), 0], [0, 0, 1]]))
    return RotMat


def translateMat(tX, tY, tZ):
    TransMat = np.array([[tX], [tY], [tZ]])
    return TransMat


class XYZEstimation(object):
    def __init__(self, param):
        Rmat = rotateMat(0, 0, 0)
        Tmat = translateMat(0, 0, 0)
        ExMat = np.concatenate([Rmat, Tmat], axis=1)
        self.param = param
        self.ExMat = np.concatenate([ExMat, np.array([0, 0, 0, 1])[None, :]], axis=0)
        self.invIntrinsicMatrix1 = np.linalg.inv(self.param['IntrinsicMatrix'][1])

    def estimate(self, box1, box2):
        """
        estimate XYZ from bbox
        :param box1: bbox from cam1
        :param box2: bbox from cam2
        :return: X, Y, Z, x2, y2 (using coor in origin image)
        """
        # limit
        box1[0] = max(0, box1[0])
        box1[1] = max(0, box1[1])
        box2[0] = max(0, box2[0])
        box2[1] = max(0, box2[1])

        # XY estimation (Center XY)
        x1, y1 = box1[0] + box1[2] // 2, box1[1] + box1[3] // 2
        x2, y2 = box2[0] + box2[2] // 2, box2[1] + box2[3] // 2

        # Z estimation
        D = np.abs(x1 - x2)  # Disparity (pixel)

        # remap to origin image coor
        # https://stackoverflow.com/questions/34265650/can-i-get-the-point-position-using-remap-in-opencv
        x1, y1 = self.param['StereoMap'][1][0][y1, x1, 0], self.param['StereoMap'][1][0][y1, x1, 1]
        x2, y2 = self.param['StereoMap'][2][0][y2, x2, 0], self.param['StereoMap'][2][0][y2, x2, 1]

        # reprojectImageTo3D
        # https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#:~:text=reprojectImageTo3D()
        xyz = self.param['Q'] @ np.array([x1, y1, D, 1]).transpose()
        X, Y, Z = xyz[0] / xyz[3], xyz[1] / xyz[3], xyz[2] / xyz[3]

        return X, Y, Z, x2, y2
