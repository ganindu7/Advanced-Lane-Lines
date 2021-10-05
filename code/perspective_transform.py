import pickle
import numpy as np 
import cv2
import sys
import time
from matplotlib.pyplot import draw
import matplotlib.image as image
import matplotlib.pyplot as plt


from matplotlib.backends.qt_compat import QtCore, QtWidgets, is_pyqt5
if is_pyqt5():
    from matplotlib.backends.backend_qt5agg import (FigureCanvas, NavigationToolbar2QT as NavToolbar)
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QSlider, QLabel, QFrame, QPushButton
else:
    print("pyQt5 not present please install ")
from matplotlib.figure import Figure

threshold_params = pickle.load(open("../pickle_files/color_and_gradient_thresholds_cam.pickle", "rb"))

mtx = threshold_params['mtx_']
dist = threshold_params['dist_']

P1 = np.array([229, 253])
P2 = np.array([469, 265])
P3 = np.array([521, 282])
P4 = np.array([13, 297])

src_road = np.float32([
        P1,
        P2,
        P3,
        P4
    ])


dst_road = np.float32([
    [150, 100],
    [150 + 285 , 100],
    [150 + 285  , 100 + 176],
    [150, 450]
])
    

M_ppt_road = cv2.getPerspectiveTransform(src_road, dst_road)

threshold_params['M_perspective_transform'] = M_ppt_road

road_img = cv2.imread('../test_images/perspective_trnsform.jpg')
undistorted_road_img = cv2.undistort(road_img, mtx, dist, None, mtx)

warped_road = cv2.warpPerspective(undistorted_road_img, M_ppt_road, (640, 480)) 

pickle.dump(threshold_params, open("../pickle_files/fianal_params_for_lane_lines.pickle", "wb"))
print("params saved")
 


f2, (ax3, ax4) = plt.subplots(1, 2, figsize=(15,6))

ax3.plot(P1[0], P1[1], '.') # top left
ax3.plot(P2[0], P2[1], '.') # top right 
ax3.plot(P3[0], P3[1], '.') # bottom right 
ax3.plot(P4[0], P4[1], '.') # bottom left


ofset = 20

ax3.text(P1[0] + ofset/2, P1[1], "P1", fontsize=14)
ax3.text(P2[0] - ofset, P2[1], "P2", fontsize=14)
ax3.text(P3[0] - ofset, P3[1], "P3", fontsize=14)
ax3.text(P4[0] + ofset, P4[1], "P4", fontsize=14)

ax4.text(dst_road[0][0] , dst_road[0][1], "P1", fontsize=14)
ax4.text(dst_road[1][0] , dst_road[1][1], "P2", fontsize=14)
# ax4.text(dst_road[2][0] , dst_road[2][1], "P3", fontsize=14)
ax4.text(dst_road[3][0] , dst_road[3][1], "P4", fontsize=14)

plt.subplots_adjust(left=0., right=1., top=0.9, bottom=0.)
# f2.tight_layout()
ax3.imshow(undistorted_road_img[:,:,::-1])
ax3.set_title('Undistorted', fontsize=40)
ax4.imshow(warped_road[:,:,::-1]) #, cmap='gray')
ax4.set_title(' warped', fontsize=40)

plt.show()
