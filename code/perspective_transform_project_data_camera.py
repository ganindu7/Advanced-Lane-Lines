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

threshold_params = pickle.load(open("../pickle_files/project_data/hue_filtered_and_gradient_selected.pickle", "rb"))

mtx = threshold_params['mtx_']
dist = threshold_params['dist_']

P1 = np.array([431 , 566])
P2 = np.array([870 , 566])
P3 = np.array([1064, 697])
P4 = np.array([253 , 697])



src_road = np.float32([
        P1,
        P2,
        P3,
        P4
    ])


height = 210
width  =(3.7*height)/3

dst_img_width  = 400
dst_img_height = 700



dst_road = np.float32([
    [dst_img_width/2 - width/2, dst_img_height - height],
    [dst_img_width/2 + width/2, dst_img_height - height],
    [dst_img_width/2 + width/2, dst_img_height],
    [dst_img_width/2 - width/2, dst_img_height]
])
    
M_ppt_road = cv2.getPerspectiveTransform(src_road, dst_road)

threshold_params['M_perspective_transform'] = M_ppt_road
threshold_params['dst_img_width'] = dst_img_width
threshold_params['dst_img_height'] = dst_img_height

road_img = cv2.imread('../test_images/straight_lines2.jpg')

# print("image size " + np.str(np.shape(road_img)))

undistorted_road_img = cv2.undistort(road_img, mtx, dist, None, mtx)

# warped_road = cv2.warpPerspective(undistorted_road_img, M_ppt_road, (1280, 720)) 
warped_road = cv2.warpPerspective(undistorted_road_img, M_ppt_road, (dst_img_width, dst_img_height)) 

pickle.dump(threshold_params, open("../pickle_files/project_data/fianal_params_for_lane_lines_sample.pickle", "wb"))
print("params saved")
print(M_ppt_road)
print("srcpoints \n", src_road)
print("destpoints \n", dst_road)
test = None

hom_src = cv2.convertPointsToHomogeneous(np.array([[431, 566]]).astype(np.float32), test) 
pt = np.array([[431, 566, 1]], dtype=np.float32);
ans = M_ppt_road.dot(pt.T)
print(ans)
print(ans/ans[-1])
# pt_ = pt.reshape(-1, 1, 3)
# print(pt_)
# res = M_ppt_road.dot(pt_.T);
# res = M_ppt_road.dot(hom_src);
# print("hom src point")
np.set_printoptions(precision=0, suppress=True)
# print("src = " , hom_src.T , " shape " , np.shape(hom_src.T))

print("dst point")
# print(res)
# print(np.shape(hom_src) )

 


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
ax4.text(dst_road[2][0] , dst_road[2][1], "P3", fontsize=14)
ax4.text(dst_road[3][0] , dst_road[3][1], "P4", fontsize=14)

plt.subplots_adjust(left=0., right=1., top=0.9, bottom=0.)
# f2.tight_layout()
ax3.imshow(undistorted_road_img[:,:,::-1])
ax3.set_title('Undistorted', fontsize=40)
ax4.imshow(warped_road[:,:,::-1]) #, cmap='gray')
ax4.set_title(' warped', fontsize=40)

plt.show()
