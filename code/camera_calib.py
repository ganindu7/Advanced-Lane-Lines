
'''
Author: Ganindu Nanayakkara
Email : ganindu@gmail.com
'''

import pickle
import numpy as np 
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
import glob
import re
import os.path

USE_PROJECT_DATA = True # Otherwise use other data 


target_loc_string = '../camera_cal/calibration*.jpg' # for project supplied stuff

images = glob.glob(target_loc_string)

objpoints = [] # 3D object points 
imgpoints = [] # 2D image points 

num_rows = 9 # need swap
num_cols = 6

objp = np.zeros((num_cols*num_rows, 3), np.float32)
objp[:,:2] = np.mgrid[0:num_rows, 0:num_cols].T.reshape(-1,2)

print(np.size(objp))
print('number of images : ', np.size(images))
print('number of object points: ', np.size(objpoints))
print('number of image points: ',np.size(imgpoints))


im = None

fname, fext = os.path.splitext(target_loc_string)
dirname = os.path.dirname(fname)



for cimage in images:

    print('processing ', cimage)
    imgc = cv2.imread(cimage)
    graycimg = cv2.cvtColor(imgc, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(graycimg, (num_rows,num_cols), None)


    if ret == True:
        imgpoints.append(corners)
        objpoints.append(objp)
        

        '''
            redundant but doing for practice
        '''

        matchstring = fname[:-1] + '[\w]' + '*' + fext

        if re.match(matchstring, cimage):

            q = re.search('[\w]*.jpg$|'+dirname+'/[\w]*', cimage)
            tempoutfile_name = q.group() + "_OUT.jpg"
            outfile_name = re.sub(dirname, dirname+"/out", tempoutfile_name)

            im = cv2.drawChessboardCorners(imgc, (num_rows,num_cols), corners, ret)

            if not os.path.exists(os.path.dirname(outfile_name)):
                os.makedirs((os.path.dirname(outfile_name)))

            if not os.path.isfile(outfile_name):

                im = cv2.drawChessboardCorners(imgc, (num_rows,num_cols), corners, ret)
                if cv2.imwrite(outfile_name, im):
                    print('writing' , outfile_name, 'adding ', np.size(corners), ' corners')
  
            
            else:
                print("File exists: ", outfile_name)

            '''
            Note: make sure the pwd is correct 
            
            '''
        # plt.imshow(im)
        # plt.show()

    else:

        failed_image = cimage

        print('failed ', failed_image)



        q_ = re.search('[\w]*.jpg$|'+dirname+'/[\w]*', failed_image)
        tempoutfile_name_ = q_.group() + "_FAILED.jpg"
        outfile_name_ = re.sub(dirname, dirname+"/failed", tempoutfile_name_)


        if not os.path.exists(os.path.dirname(outfile_name_)):
            os.makedirs((os.path.dirname(outfile_name_)))
        cv2.imwrite(outfile_name_, imgc)


'''
once the image and the object points ae sorted into respective containers 
'''


print('number of input images : ', np.size(images))
print('number of sucessful calibration candidates : ', np.shape(objpoints)[0])
print('number of object points: ', np.shape(objpoints))
print('number of image points: ',np.shape(imgpoints))

ret_, mtx_, dist_, rvecs_, tvecs_ = cv2.calibrateCamera(objpoints, imgpoints, graycimg.shape[::-1], None,None)


print('calib_matrix ',  mtx_)
print('dist_coef ',  dist_)

c = {
     "mtx_"        : mtx_ ,
     "dist_"       : dist_
     }
         
pickle.dump(c, open("../pickle_files/calib_first_pass.pickle", "wb")) if not USE_PROJECT_DATA else pickle.dump(c, open("../pickle_files/project_data/calib_first_pass.pickle", "wb")) 
