import cv2
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

USE_PROJECT_DATA = True # Otherwise use other data 

settings_source = "../pickle_files/calib_first_pass.pickle" if not USE_PROJECT_DATA else "../pickle_files/project_data/calib_first_pass.pickle"

print("Loading settings from " + settings_source)

data = pickle.load(open("../pickle_files/calib_first_pass.pickle","rb"))
calib_mtx = data['mtx_']
dist_coef = data['dist_']

test_img = cv2.imread('../test_images/test6.jpg')
rgb_testimg = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
undistorted = cv2.undistort(rgb_testimg, calib_mtx, dist_coef, None, calib_mtx)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(13,3))
f.tight_layout()
ax1.imshow(rgb_testimg)
ax1.set_title('Original image', fontsize=40)
ax2.imshow(undistorted)
ax2.set_title('Undistorted image', fontsize=40)
plt.subplots_adjust(left=0., right=1., top=0.9, bottom=0.)


plt.show()
