from datetime import datetime
from src.filters import KalmanFilter
from src.motion_estimator import RansacMotionEstimator
from src.stabilzer import OptimalPathStabilizer, OptimalPathStabilizerXYA
from src.trajectory import Trajectory
from src.transformations import fill_mat, get_x, get_y, get_rad_angle, transform


__author__ = 'Marek'

import cv2
import numpy as np


class FrameInfo:
    def __init__(self):
        # self.img = None
        # self.img_gray = None
        self.features = []
        self.number = 0
        self.trajectory = Trajectory()
        self.shape = ()

    @property
    def width(self):
        return self.shape[1]

    @property
    def height(self):
        return self.shape[0]

    @property
    def width_height(self):
        return self.shape[::-1]


video_path = r"..\..\data\1.avi"
# video_path = r"..\..\data\train.mp4"
# video_path = r"..\..\data\Vietnam_Kim_Long2.avi"
# video_path = r"..\..\data\hippo.mp4"
# video_path = r"..\..\data\videoplayback-cut.mp4"
# video_path = r"C:\Users\Marek\Dropbox\Camera Uploads\Rita\2013-04-24 12.57.54.mp4"
# video_path = r"C:\Users\Marek\Dropbox\Camera Uploads\Rita\2013-04-24 13.12.37.mp4"
# video_path = r"C:\Users\Marek\Dropbox\Camera Uploads\2014-04-27 12.13.45.mp4"

video = cv2.VideoCapture(video_path)

frame = None
prev_frame = None

org_trajectories = []
org_transformations = []
frames = []

prev_trans = None
prev_frame_img = None
frame_number = 0
frame_count = video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)

# frame_width = video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
# frame_height = video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)

frame_width = int(1336.0 / 2)
frame_height = int(768.0 / 2)


def resize(img):
    return cv2.resize(img, (frame_width, frame_height), interpolation=cv2.INTER_LANCZOS4)


lk_params = dict(winSize=(19, 19),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict(maxCorners=100,
                      qualityLevel=0.01,
                      minDistance=8,
                      blockSize=19)

motion_estimator = RansacMotionEstimator(20, 1.5, remember_inlier_indices=False)

crop_rate = 0.9
limits = [int(frame_width * (1 - crop_rate)), int(frame_height * (1 - crop_rate)), 0.05]

print "Status: path finding process started"
while True:
    ret, frame_img = video.read()

    if frame_img is None:
        break

    # if frame_number > 25:
    #     break

    if frame is not None:
        prev_frame = frame

    frame_number += 1
    print "\rStatus: processed %00.2f%% (%04d / %04d)" % (frame_number / frame_count * 100, frame_number, frame_count),

    frame = FrameInfo()
    frame.number = frame_number
    frame_img = resize(frame_img)
    frame_img_gray = cv2.cvtColor(frame_img, cv2.COLOR_BGR2GRAY)
    frame.features = cv2.goodFeaturesToTrack(frame_img_gray, **feature_params)
    frame.shape = frame_img_gray.shape
    frames.append(frame)

    if prev_frame is not None:
        # Optical flow
        new_features, _, _ = cv2.calcOpticalFlowPyrLK(prev_frame_img, frame_img, prev_frame.features, None, **lk_params)
        new_features_for_validation, _, _ = cv2.calcOpticalFlowPyrLK(frame_img, prev_frame_img, new_features, None,
                                                                     **lk_params)

        d = abs(prev_frame.features - new_features_for_validation).reshape(-1, 2).max(-1)
        good_features = d < 1

        # Select good_features points
        good_new = np.array([x for x, s in zip(new_features, good_features) if s], dtype=np.float32)
        good_old = np.array([x for x, s in zip(prev_frame.features, good_features) if s], dtype=np.float32)

        # trans = cv2.estimateRigidTransform(good_old, good_new, fullAffine=False)
        trans, _ = motion_estimator.estimate(good_old, good_new)

        if trans is None and prev_trans is None:
            print "wuf? trans is None and prev trans is None"
            continue

        if trans is None:
            print "wut? trans is None"
            trans = prev_trans

        org_transformations.append(trans)
        prev_trans = trans.copy()
    prev_frame_img = frame_img

print "\nStatus: path finding finished"
print "Status: path optimization started"

stabilizer = OptimalPathStabilizerXYA(
    [get_x(trans) for trans in org_transformations],
    [get_y(trans) for trans in org_transformations],
    [get_rad_angle(trans) for trans in org_transformations]
    , [limits[0]*0.5, limits[1]*0.5, limits[2]])

new_trans = stabilizer.stabilize()

print "Status: path optimization finished"
print "Status: transforming video started"

for _ in xrange(10):
    video.release()
    video = cv2.VideoCapture(video_path)

    frame_number = 0
    pressed_q = False
    output_video = cv2.VideoWriter(video_path+'.stab.avi',
                                   cv2.cv.CV_FOURCC(*'iyuv'), # -1, # int(video.get(cv2.cv.CV_CAP_PROP_FOURCC)),
                                   int(video.get(cv2.cv.CV_CAP_PROP_FPS)),
                                   (frame_width - 2*limits[0], frame_height - 2*limits[1])
    )
    trans = np.zeros((2, 3), dtype=np.float32)

    for t, frame in enumerate(frames):
        if t + 1 >= len(org_transformations):
            break

        print "\rStatus: processed %00.2f%% (%04d / %04d)" % (t / frame_count * 100, t, frame_count),

        fill_mat(trans, new_trans[0][t - 1], new_trans[1][t - 1], new_trans[2][t - 1])

        _, frame_img = video.read()
        frame_img = resize(frame_img)

        out = cv2.warpAffine(frame_img, trans, frame.width_height, flags=cv2.INTER_LANCZOS4,
                             borderMode=cv2.BORDER_REFLECT)

        def crop_image(img):
            return img[limits[1]:frame.height - limits[1], limits[0]:frame.width - limits[0]]

        # output_video.write(crop_image(out))
        cv2.imshow('stab', crop_image(out))
        cv2.imshow('org', crop_image(frame_img))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            pressed_q = True
            break

    video.release()
    if pressed_q:
        break

    video.release()
    output_video.release()

print "Status: transforming video finished"

# for _ in xrange(10):
#     video = cv2.VideoCapture(video_path+'.stab.avi')
#     pressed_q = False
#
#     for t, frame in enumerate(frames):
#         if t + 1 >= len(org_transformations):
#             break
#
#         start_time = datetime.now()
#
#         _, frame_img = video.read()
#
#         if frame_img is None:
#             continue
#
#         cv2.imshow('out', frame_img)
#
#         end_time = datetime.now()
#         elapsed = (end_time - start_time).microseconds / 1e3
#         wait_time = 1000 / video.get(cv2.cv.CV_CAP_PROP_FPS) - elapsed
#
#         if cv2.waitKey(int(wait_time if wait_time > 0 else 1)) & 0xFF == ord('q'):
#             pressed_q = True
#             break
#
#     video.release()
#     if pressed_q:
#         break

print "Status: done"
cv2.destroyAllWindows()
