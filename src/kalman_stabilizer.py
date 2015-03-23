from src.filters import KalmanFilter
from src.trajectory import Trajectory
from src.transformations import fill_mat, get_x, get_y, get_rad_angle, transform


__author__ = 'Marek'

import cv2
import numpy as np
from matplotlib import pyplot as plt

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

trajectory = Trajectory(0, 0, 0)
org_trajectories = []
stb_trajectories = []

crop = 40
crop_rate = crop/20
filter = KalmanFilter(Trajectory(4e-2, 4e-2, 4e-2), Trajectory(crop_rate, crop_rate, crop_rate), Trajectory(1, 1, 1))
surf = cv2.SURF(4000)
prev_trans = None
frame_number = 0

# plt.ion()
# plt.show()

while True:
    # Capture frame_img-by-frame_img
    ret, frame_img = camera.read()

    if frame_img is None:
        break

    if frame is not None:
        prev_frame = frame

    frame_number += 1

    frame = FrameInfo()
    frame.number = frame_number
    frame.img = cv2.resize(frame_img, (0, 0), fx=(1336.0/frame_img.shape[1])/2.0, fy=(768.0/frame_img.shape[0])/2.0)
    frame.img_gray = cv2.cvtColor(frame.img, cv2.COLOR_BGR2GRAY)
    frame_img = None

    if False:
        kp, des = surf.detectAndCompute(frame.img, None)
        frame.features = np.array([[[pt.pt[0], pt.pt[1]]] for pt in kp], dtype=np.float32)
    else:
        frame.features = cv2.goodFeaturesToTrack(frame.img_gray, 100, 0.01, 30)

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

    delta = Trajectory(get_x(trans), get_y(trans), get_rad_angle(trans))
    trajectory += delta

    filter.put(trajectory)
    diff = filter.get() - trajectory
    new_delta = delta + diff

    org_trajectories.append(trajectory)
    stb_trajectories.append(filter.get())

    fill_mat(trans, new_delta.x, new_delta.y, new_delta.angle)

    frame_corners = [(0, 0), (frame.width, 0), (frame.width, frame.height), (0, frame.height)]

    def point_in_frame(point, mat):
        c = np.array([
            [1, 0, point[0], point[1], 0, 0],
            [0, 1, 0, 0, point[0], point[1]]
        ])
        p = np.array([mat[0, 2], mat[1, 2], mat[0, 0], mat[0, 1], mat[1, 0], mat[1, 1]])
        r = c.dot(p)
        cv2.circle(frame.img, (int(r[0]), int(r[1])), 3, [0, 255, 255], -1)
        return 0 <= r[0] < frame.width and 0 <= r[1] < frame.height

    constraint_points = [(crop, crop), (frame.width-crop, crop), (frame.width-crop, frame.height-crop), (crop, frame.height-crop)]

    constraint_ok = True
    for corner in constraint_points:
        if not point_in_frame(corner, trans):
            constraint_ok = False
            # break
    if not constraint_ok:
        print "%3d:> constraint failed" % frame_number

    out = cv2.warpAffine(frame.img, trans, frame.width_height, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    prev_trans = trans.copy()

    # Display the resulting frame_img
    # for t, (new, old) in enumerate(zip(good_new, good_old)):
    #     a, b = new.ravel()
    #     c, d = old.ravel()
    #
    #     color = [0, 255, 255]
    #     cv2.line(out, (a, b), (c, d), color, 2)
    #     cv2.circle(out, (a, b), 3, color, -1)

    def crop_image(img):
        return img[crop:frame.height-2*crop, crop:frame.width-crop]

    cv2.imshow('out', crop_image(out))
    cv2.imshow('org', crop_image(frame.img))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

plt.plot([f.x for f in org_trajectories], [f.y for f in org_trajectories])
plt.plot([f.x for f in stb_trajectories], [f.y for f in stb_trajectories])
plt.show()

# When everything done, release the capture
camera.release()
cv2.destroyAllWindows()
