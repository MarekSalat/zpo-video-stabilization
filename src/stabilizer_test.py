from src.filters import KalmanFilter
from src.stabilzer import OptimalCameraPathStabilizer
from src.trajectory import Trajectory
from src.transformations import fill_mat, get_x, get_y, get_rad_angle, transform


__author__ = 'Marek'

import cv2
import numpy as np


class FrameInfo:
    def __init__(self):
        self.img = None
        self.img_gray = None
        self.features = []
        self.number = 0
        self.trajectory = Trajectory()

    @property
    def width(self):
        return self.img_gray.shape[1]

    @property
    def height(self):
        return self.img_gray.shape[0]

    @property
    def width_height(self):
        return self.img_gray.shape[::-1]

camera = cv2.VideoCapture(r"..\..\data\1.avi")
# camera = cv2.VideoCapture(r"..\..\data\Vietnam_Kim_Long2.avi")
# camera = cv2.VideoCapture(r"..\..\data\hippo.mp4")
# camera = cv2.VideoCapture(r"..\..\data\videoplayback-cut.mp4")
# camera = cv2.VideoCapture(r"C:\Users\Marek\Dropbox\Camera Uploads\Rita\2013-04-24 12.57.54.mp4")
# camera = cv2.VideoCapture(r"C:\Users\Marek\Dropbox\Camera Uploads\Rita\2013-04-24 13.12.37.mp4")
# camera = cv2.VideoCapture(r"C:\Users\Marek\Dropbox\Camera Uploads\2014-04-27 12.13.45.mp4")

frame = None
prev_frame = None

org_trajectories = []
org_transformations = []
frames = []

frame_border = 40
prev_trans = None
frame_number = 0

while True:
    ret, frame_img = camera.read()

    if frame_img is None:
        break

    if frame_number > 350:
        break

    if frame is not None:
        prev_frame = frame

    frame_number += 1
    print frame_number

    frame = FrameInfo()
    frame.number = frame_number
    frame.img = cv2.resize(frame_img, (0, 0), fx=(1336.0/frame_img.shape[1])/2.0, fy=(768.0/frame_img.shape[0])/2.0)
    frame.img_gray = cv2.cvtColor(frame.img, cv2.COLOR_BGR2GRAY)
    frame_img = None
    frame.features = cv2.goodFeaturesToTrack(frame.img_gray, 100, 0.01, 30)
    frames.append(frame)

    if prev_frame is None:
        continue

    # Optical flow
    new_features, status, error = cv2.calcOpticalFlowPyrLK(prev_frame.img, frame.img, prev_frame.features,
                                                           None,
                                                           winSize=(16, 16), maxLevel=8,
                                                           criteria=(
                                                               cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10,
                                                               0.03))

    # Select good points
    good_new = np.array([x for x, s in zip(new_features, status) if s == 1], dtype=np.float32)
    good_old = np.array([x for x, s in zip(prev_frame.features, status) if s == 1], dtype=np.float32)

    trans = cv2.estimateRigidTransform(good_old, good_new, fullAffine=False)

    if trans is None and prev_trans is None:
        print "wuf? trans is None"
        continue

    if trans is None:
        trans = prev_trans

    org_transformations.append(trans)
    prev_trans = trans.copy()

stabilizer_x = OptimalCameraPathStabilizer([get_x(trans) for trans in org_transformations], frame_border)
stabilizer_y = OptimalCameraPathStabilizer([get_y(trans) for trans in org_transformations], frame_border)
stabilizer_angle = OptimalCameraPathStabilizer([get_rad_angle(trans) for trans in org_transformations], 0.06)

new_trans_x, _ = stabilizer_x.stabilize()
new_trans_y, _ = stabilizer_y.stabilize()
new_trans_angle, _ = stabilizer_angle.stabilize()

OptimalCameraPathStabilizer.cleanup()

for _ in xrange(10):
    pressed_q = False
    frame_number = 0
    for t, frame in enumerate(frames):
        if t >= len(org_transformations):
            break

        trans = org_transformations[t].copy()

        fill_mat(trans, new_trans_x[t], new_trans_y[t], new_trans_angle[t])

        out = cv2.warpAffine(frame.img, trans, frame.width_height, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)

        def crop_image(img, crop):
            return img[crop:frame.height-2*crop, crop:frame.width-crop]
        #
        cv2.imshow('out', crop_image(out, frame_border))
        cv2.imshow('org', crop_image(frame.img, frame_border))

        if cv2.waitKey(1000/28) & 0xFF == ord('q'):
            pressed_q = True
            break

    if pressed_q:
        break

# When everything done, release the capture
camera.release()
cv2.destroyAllWindows()
