import cv2
import numpy as np


def CAMPARAM_INIT(SYSTEM_DEFINE, FRAME_WIDTH=None, FRAME_HEIGHT=None, WINDOW_WIDTH=None, WINDOW_HEIGHT=None):
    if FRAME_WIDTH is None or FRAME_HEIGHT is None or WINDOW_WIDTH is None or WINDOW_HEIGHT is None:
        FRAME_WIDTH = int(1280 * 1)  # 1280
        FRAME_HEIGHT = int(720 * 1)  # 720
        WINDOW_WIDTH = int(FRAME_WIDTH * 0.5)
        WINDOW_HEIGHT = int(FRAME_HEIGHT * 0.5)

    """ Init MVS params """
    if SYSTEM_DEFINE == 6:
        # new camera (front)
        # calibrated using 640*480 images
        FRAME_WIDTH = int(640)
        FRAME_HEIGHT = int(480)
        WINDOW_WIDTH = int(FRAME_WIDTH)
        WINDOW_HEIGHT = int(FRAME_HEIGHT)
        print("[INFO_CAMPARAM_INIT] SYSTEM_DEFINE: ", SYSTEM_DEFINE)
        # Intrinsic Matrix
        inmat1 = np.array([
            [282.357970545391, 0, 315.308920962467],
            [0, 282.374112968515, 233.874661234589],
            [0, 0, 1]
        ])
        inmat2 = np.array([
            [281.440209171335, 0, 315.031006572725],
            [0, 281.516634558236, 249.706004605719],
            [0, 0, 1]
        ])
        IntrinsicMatrix = {1: inmat1, 2: inmat2}
        F1 = 1.6  # focal length (mm)
        F2 = 1.6  # focal length (mm)

        # Distortion Matrix (先立体矫正，再畸变矫正)
        d1 = np.array([-0.0545924585767390, 0.0103644997612646, 0, 0, 0])  # k1, k2, p1, p2, k3
        d2 = np.array([-0.0550842535366952, 0.0107756429402093, 0, 0, 0])  # k1, k2, p1, p2, k3
        DistortionMatrix = {1: d1, 2: d2}

        # Stereo Rectify
        R = np.array([
            [0.998534382106568, 0.0537265409007298, 0.00652277194882078],
            [-0.0537090446482382, 0.998552621223743, -0.00282863397502064],
            [-0.00666530374609184, 0.00247415642862339, 0.999974725818577]
        ])  # MATLAB标定得到的R (Rotation of camera 2 relative to camera 1, specified as a 3-by-3 matrix.)
        R = np.transpose(R)  # 转置的R
        T = np.array([-9.26200092926269, -0.0672672960331219, -0.132012743740196])
    elif SYSTEM_DEFINE == 7:
        # new camera
        # calibrated using 640*480 images
        FRAME_WIDTH = int(640)
        FRAME_HEIGHT = int(480)
        WINDOW_WIDTH = int(FRAME_WIDTH)
        WINDOW_HEIGHT = int(FRAME_HEIGHT)
        print("[INFO_CAMPARAM_INIT] SYSTEM_DEFINE: ", SYSTEM_DEFINE)
        # Intrinsic Matrix
        inmat1 = np.array([
            [287.670387177877, 0, 315.725240873607],
            [0, 287.641510156020, 241.430184602218],
            [0, 0, 1]
        ])
        inmat2 = np.array([
            [290.822740297873, 0, 321.149591723798],
            [0, 291.111009549891, 232.967121699588],
            [0, 0, 1]
        ])
        IntrinsicMatrix = {1: inmat1, 2: inmat2}
        F1 = 1.6  # focal length (mm)
        F2 = 1.6  # focal length (mm)

        # Distortion Matrix (先立体矫正，再畸变矫正)
        d1 = np.array([-0.0615728963807713, 0.0179516920867288, 0, 0, 0])  # k1, k2, p1, p2, k3
        d2 = np.array([-0.0616875713188978, 0.0167161516380168, 0, 0, 0])  # k1, k2, p1, p2, k3
        DistortionMatrix = {1: d1, 2: d2}

        # Stereo Rectify
        R = np.array([
            [0.997098089616646, 0.0759128613987804, 0.00571289393250387],
            [-0.0759595780284329, 0.997075003985588, 0.00846043337435355],
            [-0.00505392803439962, -0.00886983096733104, 0.999947890597322]
        ])  # MATLAB标定得到的R (Rotation of camera 2 relative to camera 1, specified as a 3-by-3 matrix.)
        R = np.transpose(R)  # 转置的R
        T = np.array([-8.63761228284382, -0.298861998033701, 0.192194923328153])
    elif SYSTEM_DEFINE == 8:
        # new camera
        # calibrated using 640*480 images
        FRAME_WIDTH = int(640)
        FRAME_HEIGHT = int(480)
        WINDOW_WIDTH = int(FRAME_WIDTH)
        WINDOW_HEIGHT = int(FRAME_HEIGHT)
        print("[INFO_CAMPARAM_INIT] SYSTEM_DEFINE: ", SYSTEM_DEFINE)
        # Intrinsic Matrix
        inmat1 = np.array([
            [284.353216239963, 0, 312.237322838262],
            [0, 284.414192878758, 246.358931259506],
            [0, 0, 1]
        ])
        inmat2 = np.array([
            [284.330343969323, 0, 316.795702491927],
            [0, 284.230736938637, 234.909786555181],
            [0, 0, 1]
        ])
        IntrinsicMatrix = {1: inmat1, 2: inmat2}
        F1 = 1.6  # focal length (mm)
        F2 = 1.6  # focal length (mm)

        # Distortion Matrix (先立体矫正，再畸变矫正)
        d1 = np.array([-0.0544072568368386, 0.0112347620767092, 0, 0, 0])  # k1, k2, p1, p2, k3
        d2 = np.array([-0.0535899290716405, 0.00832102271132433, 0, 0, 0])  # k1, k2, p1, p2, k3
        DistortionMatrix = {1: d1, 2: d2}

        # Stereo Rectify
        R = np.array([
            [0.997871184718268, 0.0646599167619950, 0.00849669778753439],
            [-0.0647237994779895, 0.997875244967633, 0.00747163047268445],
            [-0.00799552938171167, -0.00800566331538019, 0.999935988383650]
        ])  # MATLAB标定得到的R (Rotation of camera 2 relative to camera 1, specified as a 3-by-3 matrix.)
        R = np.transpose(R)  # 转置的R
        T = np.array([-9.05988395043143, -0.619850513312531, -0.447542542490896])
    else:
        raise ValueError('[ERROR_CAMPARAM_INIT] System undefined!')
    """==========================="""

    # Distortion Map  # cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
    mapx1, mapy1 = cv2.initUndistortRectifyMap(IntrinsicMatrix[1], DistortionMatrix[1], None, IntrinsicMatrix[1], (WINDOW_WIDTH, WINDOW_HEIGHT), 5)
    mapx2, mapy2 = cv2.initUndistortRectifyMap(IntrinsicMatrix[2], DistortionMatrix[2], None, IntrinsicMatrix[2], (WINDOW_WIDTH, WINDOW_HEIGHT), 5)
    DistortionMap = {1: [mapx1, mapy1], 2: [mapx2, mapy2]}
    # Calibration Map
    RL, RR, PL, PR, Q, roiL, roiR = cv2.stereoRectify(IntrinsicMatrix[1], DistortionMatrix[1], IntrinsicMatrix[2], DistortionMatrix[2],
                                                      (WINDOW_WIDTH, WINDOW_HEIGHT), R, T, 1, (0, 0))
    camera1map = cv2.initUndistortRectifyMap(IntrinsicMatrix[1], DistortionMatrix[1], RL, PL, (WINDOW_WIDTH, WINDOW_HEIGHT), cv2.CV_16SC2)
    camera2map = cv2.initUndistortRectifyMap(IntrinsicMatrix[2], DistortionMatrix[2], RR, PR, (WINDOW_WIDTH, WINDOW_HEIGHT), cv2.CV_16SC2)
    StereoMap = {1: camera1map, 2: camera2map}

    print("[INFO_CAMPARAM_INIT] Use frame size: [{}, {}] [{}, {}]".format(FRAME_WIDTH, FRAME_HEIGHT, WINDOW_WIDTH, WINDOW_HEIGHT))

    return dict(SYSTEM_DEFINE=SYSTEM_DEFINE,
                IntrinsicMatrix=IntrinsicMatrix,
                DistortionMatrix=DistortionMatrix,
                DistortionMap=DistortionMap,
                StereoMap=StereoMap,
                R=R,
                T=T,
                Q=Q,
                FRAME_WIDTH=FRAME_WIDTH,
                FRAME_HEIGHT=FRAME_HEIGHT,
                WINDOW_WIDTH=WINDOW_WIDTH,
                WINDOW_HEIGHT=WINDOW_HEIGHT,
                F=[F1, F2])
