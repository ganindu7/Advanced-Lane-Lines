import pickle
import numpy as np
import cv2
import sys
import time
from matplotlib.pyplot import draw
import matplotlib.image as image 
from matplotlib.figure import Figure
from matplotlib.backends.qt_compat import QtCore, QtWidgets, is_pyqt5
if is_pyqt5():
    from matplotlib.backends.backend_qt5agg import (FigureCanvas, NavigationToolbar2QT as NavToolbar)
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QSlider, QLabel, QFrame, QPushButton
else:
    print("PyQt5 not present, Please install");


data = pickle.load(open("../pickle_files/calib_first_pass.pickle", "rb"))


class Appwindow(QtWidgets.QMainWindow):
    def __init__(self, dictin) -> None:
        super().__init__()

        self.main = QtWidgets.QWidget()
        self.setCentralWidget(self.main)
        self.grid = QtWidgets.QGridLayout(self.main)
        self.mask_image = None;

        self.static_canvas_left = FigureCanvas(Figure(figsize=(5,3)))
        self.grid.addWidget(self.static_canvas_left, 1, 1)
        self.ax1 = self.static_canvas_left.figure.subplots()

        self.static_canvas_right = FigureCanvas(Figure(figsize=(5,3)))
        self.grid.addWidget(self.static_canvas_right, 1, 2)
        self.ax2 = self.static_canvas_right.figure.subplots()

        '''
        slider config for min gradient angle
        '''
        self.slider_min_theta = QSlider(Qt.Horizontal)
        self.slider_min_theta.setSingleStep(1)
        self.slider_min_theta.setMinimum(0)
        self.slider_min_theta.setMaximum(180)
        self.slider_min_theta.setTickInterval(10)
        self.slider_min_theta.setTickPosition(QSlider.TicksBelow)
        self.slider_min_theta.valueChanged.connect(self.SliderMoved)
        self.grid.addWidget(self.slider_min_theta, 2,1)
       
        '''
        slider config for max gradient angle
        '''
        self.slider_max_theta = QSlider(Qt.Horizontal)
        self.slider_max_theta.setSingleStep(1)
        self.slider_max_theta.setMinimum(0)
        self.slider_max_theta.setMaximum(180)
        self.slider_max_theta.setTickInterval(10)
        self.slider_max_theta.setTickPosition(QSlider.TicksBelow)
        self.slider_max_theta.valueChanged.connect(self.SliderMoved)
        self.grid.addWidget(self.slider_max_theta, 2,2)

        '''
		labels to show value 
        '''

        self.label_theta_min = QLabel(self)
        self.label_theta_min.setAlignment(Qt.AlignBottom | Qt.AlignLeft)
        self.grid.addWidget(self.label_theta_min, 3,2,1,1)


        self.label_theta_max = QLabel(self)
        self.label_theta_max.setAlignment(Qt.AlignBottom | Qt.AlignLeft)
        self.grid.addWidget(self.label_theta_max, 3,2,3,1)


        '''
        note implement button to save changes
        '''
        tx = 'save_gradients'
        self.button = QPushButton(tx, self)
        self.btnWidth = self.button.fontMetrics().boundingRect(tx).width() + 120
        self.button.setMaximumWidth(self.btnWidth)
        self.button.clicked.connect(self.saveButtonPressed)
        self.grid.addWidget(self.button, 3,2,1,1, Qt.AlignRight)



        tx_exit = 'exit'
        self.btnExit = QPushButton(tx_exit, self)
        self.btnExitWidth = self.btnExit.fontMetrics().boundingRect(tx_exit).width() + 120
        self.btnExit.setMaximumWidth(self.btnExitWidth)
        self.btnExit.clicked.connect(self.ExitbuttonPressed)
        # self.grid.addWidget(self.btnExit, 3,1,10,2, Qt.AlignRight)
        self.grid.addWidget(self.btnExit, 3,1,1,1)


        '''
        ****************** Code for the functions *************************** 
        '''

        self.theta_max = ''
        self.theta_min = ''
        self.arct_img = dictin['arct_img']
        self.orig_img = dictin['orig_img']

        self.imglhs = self.ax1.imshow(cv2.cvtColor(self.orig_img, cv2.COLOR_BGR2RGB))
        self.imgrhs = self.ax2.imshow(self.arct_img, cmap='gray')


        # self.mtx  = dictin['mtx_']
        # self.dist = dictin['dist_']


        img = cv2.imread('../examples/binary_combo_example.jpg')

        '''
            implement a gui that has two sliders to select left and right lane-lines 
            but question weather it's really needed because in curvy roads this can't really help as it can be a range 
            the plan is to be able to select lane line gradient ranges 


        '''

    def filter_threshold(self, thresh=(0, np.pi/2)):

    	sobel_bin_filtered = np.zeros_like(self.arct_img)
    	sobel_bin_filtered[(self.arct_img >= thresh[0]) & (self.arct_img <= thresh[1])] = 5

    	binary_output = np.copy(sobel_bin_filtered)
    	return binary_output

        
    def SliderMoved(self):

        theta_min = self.slider_min_theta.value()
        theta_max = self.slider_max_theta.value()


        if theta_max < theta_min:
        	self.slider_max_theta.setValue(theta_min)
        	theta_min = theta_max

        self.theta_min = theta_min
        self.theta_max = theta_max

        self.label_theta_min.setText("theta min = " +  np.str(theta_min))
        self.label_theta_max.setText("theta max = " +  np.str(theta_max))
 
        filtered = self.filter_threshold(thresh=(theta_min*np.pi/180., theta_max*np.pi/180.))
        # self.mask_image = filtered;

        self.imgrhs.set_data(filtered)
        self.ax2.figure.canvas.draw()

    def saveButtonPressed(self):

    	gradients = {

    	'theta_max' : self.theta_max,
    	'theta_min' : self.theta_min

    	}

    	pickle.dump(gradients, open("../pickle_files/gradients_temp.pickle", "wb"))

    def ExitbuttonPressed(self):
    	self.close()
    
if __name__ == "__main__":

    qapp = QtWidgets.QApplication.instance()
    if not qapp:
        qapp = QtWidgets.QApplication(sys.argv)


    test_img = cv2.imread('../test_images/test6.jpg')
    dc  = pickle.load(open("../pickle_files/calib_first_pass.pickle","rb"))

    sobel_kernel = 15

    grayimgraw = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    norm_image = cv2.normalize(grayimgraw, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    blurred_img = cv2.GaussianBlur(norm_image, (3,3), cv2.BORDER_DEFAULT)
    grayimg = blurred_img

    sobel_x = cv2.Sobel(grayimg, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(grayimg, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    '''
    convert image x, y(rows,cols) to Cartesian x, y for ease of visualizing while taking abs 
    by swapping axes
    '''

    abs_sobel_x = np.absolute(sobel_y)
    abs_sobel_y = np.absolute(sobel_x)

    scaled_sobel_x = np.uint8(255*abs_sobel_x/np.max(abs_sobel_x))
    scaled_sobel_y = np.uint8(255*abs_sobel_y/np.max(abs_sobel_y))

    arct_img = np.arctan2(abs_sobel_y, abs_sobel_x)

    dc = {

    'orig_img' : test_img,
    'arct_img' : arct_img

    }

    app = Appwindow(dictin=dc)
    app.show()
    app.activateWindow()
    app.raise_()
    qapp.exec_()

