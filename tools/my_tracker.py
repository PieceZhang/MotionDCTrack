import cv2
import torch
import numpy as np


# noinspection PyTypeChecker
class MVSTracker(object):
    def __init__(self, trackerlist: list):
        assert type(trackerlist) is list
        self.trackerlist = trackerlist
        self.outputs = [[] for _ in range(len(trackerlist))]
        self.bboxlog = [[] for _ in range(len(trackerlist))]
        print("[INFO] MVSTracker built: using {} cameras".format(len(trackerlist)))

    def init(self, framelist: list, rectlist: list):
        for camnum, frame in enumerate(framelist):
            self.trackerlist[camnum].init(frame, rectlist[camnum])

    def track(self, framelist: list):
        self.trackerlist[0].update_from_other(self.trackerlist)
        self.trackerlist[1].update_from_other(self.trackerlist)
        self.trackerlist[0].track_stage1(framelist[0])
        self.trackerlist[1].track_stage1(framelist[1])
        self.trackerlist[0].model.track_calc()
        self.outputs[0] = self.trackerlist[0].track_stage2(framelist[0])
        self.outputs[1] = self.trackerlist[1].track_stage2(framelist[1])
        self.bboxlog[0].append(self.outputs[0]['bbox'])
        self.bboxlog[1].append(self.outputs[1]['bbox'])
        return self.outputs
