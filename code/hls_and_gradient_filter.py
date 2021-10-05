import pickle
from PyQt5.QtWidgets import QMainWindow
import numpy as np
import cv2
import sys
import time
from matplotlib.pyplot import draw
import matplotlib.image as image 
from matplotlib.figure import Figure
from pathlib import Path
from matplotlib.backends.qt_compat import QtCore, QtWidgets, is_pyqt5
if is_pyqt5():
    from matplotlib.backends.backend_qt5agg import (FigureCanvas, NavigationToolbar2QT as NavToolbar)
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QSlider, QLabel, QFrame, QPushButton
else:
    print("PyQt5 not present, Please install");



USE_PROJECT_DATA = True # Otherwise use other data 
settings_source = "../pickle_files/calib_first_pass.pickle" if not USE_PROJECT_DATA else "../pickle_files/project_data/calib_first_pass.pickle"
pickle_dest = "../pickle_files/hue_filtered_and_gradient_selected.pickle" if not USE_PROJECT_DATA else "../pickle_files/project_data/hue_filtered_and_gradient_selected.pickle"
print("Loading settings from " + settings_source)

data = pickle.load(open(settings_source , "rb"));

class AppWindowHue(QMainWindow):
    def __init__(self, dictin) -> None:
        super().__init__()

        self.mtx  = dictin['mtx_'];
        self.dist = dictin['dist_'];

        dictin['hue1_low']  = 0;
        dictin['hue1_high'] = 100;

        dictin['hue2_low']  = 0;
        dictin['hue2_high'] = 100;

        dictin['theta_min'] = 0;
        dictin['theta_max'] = 1;

        dictin['saturation_min'] = 1;
        dictin['saturation_max'] = 200;

        self.derivative_view_thresh = 1; #if im actually serious I would crerate sliders fot this
        self.saturation_threshold = 1;
        self.color_view_thershold   = 255;

        self.dict = dictin;
        sobel_kernel_size = 15;

        # img               = cv2.imread('../test_images/out_4.jpg', cv2.IMREAD_COLOR);
        # img               = cv2.imread('../test_images/straight_lines2.jpg', cv2.IMREAD_COLOR);
        img               = cv2.imread('../test_images/challenge_snap.png', cv2.IMREAD_COLOR);  # out-19
        self.gray         = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
        self.norm_image   = cv2.normalize(self.gray, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
        self.blurred_gray = cv2.GaussianBlur(self.norm_image, (3,3), cv2.BORDER_DEFAULT)
        sobel_x           = cv2.Sobel(self.blurred_gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel_size)
        sobel_y           = cv2.Sobel(self.blurred_gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel_size)

        '''
        convert image x, y(rows,cols) to Cartesian x, y for ease of visualizing while taking abs 
        by swapping axes
        '''

        abs_sobel_x = np.absolute(sobel_y)
        abs_sobel_y = np.absolute(sobel_x)

        scaled_sobel_x = (255*abs_sobel_x/np.max(abs_sobel_x))
        scaled_sobel_y = (255*abs_sobel_y/np.max(abs_sobel_y))

        self.arct_img = np.arctan2(scaled_sobel_y, scaled_sobel_x)



        hls             = cv2.cvtColor(img, cv2.COLOR_BGR2HLS);
        self.H_channel  = hls[:,:,0];
        self.S_channel  = hls[:,:,2];

        self._main = QtWidgets.QWidget();
        self.setCentralWidget(self._main);

        self.grid = QtWidgets.QGridLayout(self._main);

        self.static_canvas_left = FigureCanvas(Figure(figsize=(10,6)))
        self.grid.addWidget(self.static_canvas_left, 1, 1);
        self.ax1 = self.static_canvas_left.figure.subplots();

        self.static_canvas_right = FigureCanvas(Figure(figsize=(10,6)))
        self.grid.addWidget(self.static_canvas_right, 1, 2);
        self.ax2 = self.static_canvas_right.figure.subplots();

        self.static_canvas_3 = FigureCanvas(Figure(figsize=(10,6)))
        self.grid.addWidget(self.static_canvas_3, 1, 3);
        self.ax3 = self.static_canvas_3.figure.subplots();

        self.static_canvas_4 = FigureCanvas(Figure(figsize=(10,6)))
        self.grid.addWidget(self.static_canvas_4, 1, 4);
        self.ax4 = self.static_canvas_4.figure.subplots();

        self.static_canvas_5 = FigureCanvas(Figure(figsize=(10,6)))
        self.grid.addWidget(self.static_canvas_5, 1, 5);
        self.ax5 = self.static_canvas_5.figure.subplots();

        self.theta_min = dictin['theta_min'];
        self.theta_max = dictin['theta_max'];

        self.hue1_min = dictin['hue1_low'];
        self.hue1_max = dictin['hue1_high'];

        self.hue2_min = dictin['hue2_low'];
        self.hue2_max = dictin['hue2_high'];

        self.smin = dictin['saturation_min'];
        self.smax = dictin['saturation_max'];

        '''
        If there is a pickle file generated from a previous run we read in those values .
        '''

        file  = Path(pickle_dest);
        if file.is_file():
            print("we have a pickle file from a previous run at " + np.str(file) +", loading it!\n")
            prev_data = pickle.load(open(pickle_dest, "rb"));

            self.theta_min = prev_data['theta_min'];
            self.theta_max = prev_data['theta_max'];

            self.hue1_min = prev_data['hue1_low'];
            self.hue1_max = prev_data['hue1_high'];

            self.hue2_min = prev_data['hue2_low'];
            self.hue2_max = prev_data['hue2_high'];

            self.smin = prev_data['saturation_min'];
            self.smax = prev_data['saturation_max'];

        self.result = None

        self.slider_hue_1_floor = QSlider(Qt.Horizontal)
        self.slider_hue_1_floor.setSingleStep(1)
        self.slider_hue_1_floor.setMinimum(0)
        self.slider_hue_1_floor.setMaximum(255)
        self.slider_hue_1_floor.setTickInterval(10)
        self.slider_hue_1_floor.setValue(self.hue1_min)
        self.slider_hue_1_floor.setTickPosition(QSlider.TicksBelow)

        self.slider_hue_1_floor.valueChanged.connect(self.SliderMoved)
        self.grid.addWidget(self.slider_hue_1_floor, 2, 1)

        self.slider_hue_1_ceil = QSlider(Qt.Horizontal)
        self.slider_hue_1_ceil.setSingleStep(1)
        self.slider_hue_1_ceil.setMinimum(0)
        self.slider_hue_1_ceil.setMaximum(255)
        self.slider_hue_1_ceil.setTickInterval(10)
        self.slider_hue_1_ceil.setValue(self.hue1_max)
        self.slider_hue_1_ceil.setTickPosition(QSlider.TicksBelow)

        self.slider_hue_1_ceil.valueChanged.connect(self.SliderMoved)
        self.grid.addWidget(self.slider_hue_1_ceil, 3, 1)

        self.slider_hue_2_floor = QSlider(Qt.Horizontal)
        self.slider_hue_2_floor.setSingleStep(1)
        self.slider_hue_2_floor.setMinimum(0)
        self.slider_hue_2_floor.setMaximum(255)
        self.slider_hue_2_floor.setTickInterval(10)
        self.slider_hue_2_floor.setValue(self.hue2_min)
        self.slider_hue_2_floor.setTickPosition(QSlider.TicksBelow)

        self.slider_hue_2_floor.valueChanged.connect(self.SliderMoved)
        self.grid.addWidget(self.slider_hue_2_floor, 2, 2)

        self.slider_hue_2_ceil = QSlider(Qt.Horizontal)
        self.slider_hue_2_ceil.setSingleStep(1)
        self.slider_hue_2_ceil.setMinimum(0)
        self.slider_hue_2_ceil.setMaximum(255)
        self.slider_hue_2_ceil.setTickInterval(10)
        self.slider_hue_2_ceil.setValue(self.hue2_max)
        self.slider_hue_2_ceil.setTickPosition(QSlider.TicksBelow)

        self.slider_hue_2_ceil.valueChanged.connect(self.SliderMoved)
        self.grid.addWidget(self.slider_hue_2_ceil, 3, 2)
        
        '''
        Slider for min angle
        '''
        self.slider_min_theta = QSlider(Qt.Horizontal)
        self.slider_min_theta.setSingleStep(1)
        self.slider_min_theta.setMinimum(0)
        self.slider_min_theta.setMaximum(180)
        self.slider_min_theta.setTickInterval(10)
        self.slider_min_theta.setValue(self.theta_min)
        self.slider_min_theta.setTickPosition(QSlider.TicksBelow)

        self.slider_min_theta.valueChanged.connect(self.SliderMoved)
        self.grid.addWidget(self.slider_min_theta, 2,3)
       
        '''
        slider config for max gradient angle
        '''
        self.slider_max_theta = QSlider(Qt.Horizontal)
        self.slider_max_theta.setSingleStep(1)
        self.slider_max_theta.setMinimum(0)
        self.slider_max_theta.setMaximum(180)
        self.slider_max_theta.setTickInterval(10)
        self.slider_max_theta.setValue(self.theta_max)
        self.slider_max_theta.setTickPosition(QSlider.TicksBelow)

        self.slider_max_theta.valueChanged.connect(self.SliderMoved)
        self.grid.addWidget(self.slider_max_theta, 3,3)

        self.slider_S_min = QSlider(Qt.Horizontal)
        self.slider_S_min.setSingleStep(1)
        self.slider_S_min.setMinimum(0)
        self.slider_S_min.setMaximum(255)
        self.slider_S_min.setTickInterval(10)
        self.slider_S_min.setValue(self.smin)
        self.slider_S_min.setTickPosition(QSlider.TicksBelow)

        self.slider_S_min.valueChanged.connect(self.SliderMoved)
        self.grid.addWidget(self.slider_S_min, 2, 4)

        self.slider_S_max = QSlider(Qt.Horizontal)
        self.slider_S_max.setSingleStep(1)
        self.slider_S_max.setMinimum(0)
        self.slider_S_max.setMaximum(255)
        self.slider_S_max.setTickInterval(10)
        self.slider_S_max.setValue(self.smax)
        self.slider_S_max.setTickPosition(QSlider.TicksBelow)

        self.slider_S_max.valueChanged.connect(self.SliderMoved)
        self.grid.addWidget(self.slider_S_max, 3, 4)



        self.label_hue_1_floor = QLabel(self)
        self.label_hue_1_floor.setText("HUE 1 MIN: " + np.str(self.hue1_min));
        self.label_hue_1_floor.setAlignment(Qt.AlignBottom | Qt.AlignLeft)
        self.grid.addWidget(self.label_hue_1_floor, 8, 1)

        self.label_hue_1_ceil = QLabel(self)
        self.label_hue_1_ceil.setText("HUE 1 MAX: " + np.str(self.hue1_max));
        self.grid.addWidget(self.label_hue_1_ceil, 9, 1)


        self.label_hue_2_floor = QLabel(self)
        self.label_hue_2_floor.setText("HUE 2 MIN: " + np.str(self.hue2_min));
        self.label_hue_2_floor.setAlignment(Qt.AlignBottom | Qt.AlignLeft)
        self.grid.addWidget(self.label_hue_2_floor, 8, 2)

        self.label_hue_2_ceil = QLabel(self)
        self.label_hue_2_ceil.setText("HUE 2 MAX: " + np.str(self.hue2_max));
        self.label_hue_2_ceil.setAlignment(Qt.AlignBottom)
        self.grid.addWidget(self.label_hue_2_ceil, 9, 2)


        '''
		labels to show gradient value 
        '''

        self.label_theta_min = QLabel(self)
        self.label_theta_min.setText("THETA LOW: " + np.str(self.theta_min));
        self.label_theta_min.setAlignment(Qt.AlignBottom | Qt.AlignLeft)
        self.grid.addWidget(self.label_theta_min, 4,3,1,1)


        self.label_theta_max = QLabel(self)
        self.label_theta_max.setText("THETA HI     : " + np.str(self.theta_max));
        self.label_theta_max.setAlignment(Qt.AlignBottom | Qt.AlignLeft)
        self.grid.addWidget(self.label_theta_max, 4,3,4,1)


        self.label_sat_min = QLabel(self)
        self.label_sat_min.setText("SAT LOW: " + np.str(self.smin));
        self.label_sat_min.setAlignment(Qt.AlignBottom | Qt.AlignLeft)
        self.grid.addWidget(self.label_sat_min, 4,4,1,1)


        self.label_sat_max = QLabel(self)
        self.label_sat_max.setText("SAT HI     : " + np.str(self.smax));
        self.label_sat_max.setAlignment(Qt.AlignBottom | Qt.AlignLeft)
        self.grid.addWidget(self.label_sat_max, 4,4,4,1)

        '''
        initial processing 
        '''

        h1_binary_t = np.zeros_like(self.H_channel);
        h2_binary_t = np.zeros_like(self.H_channel);
        h1_binary_t[(self.H_channel >= self.hue1_min) & (self.H_channel <= self.hue1_max)] = 1;
        h2_binary_t[(self.H_channel >= self.hue2_min) & (self.H_channel <= self.hue2_max)] = 1;

        self.composite_stacked = np.dstack((np.zeros_like(self.H_channel), h1_binary_t, h2_binary_t)) * 255;
        self.imglhs = self.ax1.imshow(self.composite_stacked, cmap="gray");

        combined_binary = np.zeros_like(self.H_channel);
        combined_binary[(h1_binary_t == 1) | (h2_binary_t == 1)] = 255;
        self.imgrhs = self.ax2.imshow(combined_binary, cmap="gray");


        filtered_gradient = self.filter_threshold(thresh=(self.theta_min*np.pi/180., self.theta_max*np.pi/180.))
        self.img_3 = self.ax3.imshow(filtered_gradient, cmap='gray')
        self.ax3.figure.canvas.draw()

        s_binary_t = np.zeros_like(self.S_channel)
        s_binary_t[(self.S_channel >= self.smin) & (self.S_channel <= self.smax)] = self.saturation_threshold
        # s_stacked = np.dstack((np.zeros_like(self.S_channel), np.zeros_like(self.S_channel), s_binary_t)) * 255;
        # self.img_4 = self.ax4.imshow(s_stacked, cmap='gray')
        self.img_4 = self.ax4.imshow(s_binary_t * 225, cmap='gray')

        combined_hue_sat_grad_binary = np.zeros_like(self.arct_img)
        combined_hue_sat_grad_binary[(combined_binary == 255) & (filtered_gradient == self.derivative_view_thresh) & (s_binary_t == self.saturation_threshold)] = 255
        self.img_5 = self.ax5.imshow(combined_hue_sat_grad_binary, cmap='gray')

        self.result = np.copy(combined_hue_sat_grad_binary);

        tx = 'Save threholds'
        self.button = QPushButton(tx, self);
        self.bwidth = self.button.fontMetrics().boundingRect(tx).width() + 120;
        self.button.setMaximumWidth(self.bwidth);
        self.button.clicked.connect(self.buttonPressed);
        self.grid.addWidget(self.button, 9, 2, 1, 1, Qt.AlignRight);

    
    def filter_threshold(self, thresh=(0, np.pi/2)):
        sobel_bin_filtered = np.zeros_like(self.arct_img)
        sobel_bin_filtered[(self.arct_img >= thresh[0]) & (self.arct_img <= thresh[1])] = self.derivative_view_thresh ;
        binary_output = np.copy(sobel_bin_filtered);
        return binary_output;


    def SliderMoved(self):
        val_h1_min  = self.slider_hue_1_floor.value();
        val_h1_max  = self.slider_hue_1_ceil.value();

        if val_h1_max < val_h1_min:
            self.slider_hue_1_ceil.setValue(val_h1_min);
            val_h1_max = val_h1_min;

        val_h2_min  = self.slider_hue_2_floor.value();
        val_h2_max  = self.slider_hue_2_ceil.value();

        if val_h2_max < val_h2_min:
            self.slider_hue_2_ceil.setValue(val_h2_min);
            val_h2_max = val_h2_min;


        self.hue1_min = val_h1_min;
        self.hue1_max = val_h1_max;
        self.hue2_min = val_h2_min;
        self.hue2_max = val_h2_max;


        self.label_hue_1_floor.setText('HUE 1 MIN: '+np.str(self.hue1_min));
        self.label_hue_1_ceil.setText('HUE 1 MAX: '+np.str(self.hue1_max));
        self.label_hue_2_floor.setText('HUE 2 MIN: '+np.str(self.hue2_min));
        self.label_hue_2_ceil.setText('HUE 2 MAX: '+np.str(self.hue2_max));


        h1_binary_t = np.zeros_like(self.H_channel);
        h2_binary_t = np.zeros_like(self.H_channel);
        h1_binary_t[(self.H_channel >= self.hue1_min) & (self.H_channel <= self.hue1_max)] = 1;
        h2_binary_t[(self.H_channel >= self.hue2_min) & (self.H_channel <= self.hue2_max)] = 1;

        composite_stacked = np.dstack((np.zeros_like(self.H_channel), h1_binary_t, h2_binary_t)) * self.color_view_thershold;
        self.imglhs.set_data(composite_stacked);
        self.ax1.figure.canvas.draw(); # 1


        combined_binary = np.zeros_like(self.H_channel);
        combined_binary[(h1_binary_t == 1) | (h2_binary_t == 1)] = 255; # 255 is fine for a binary image
        self.imgrhs.set_data(combined_binary);
        self.ax2.figure.canvas.draw(); # 2


        theta_min = self.slider_min_theta.value()
        theta_max = self.slider_max_theta.value()

        if theta_max < theta_min:
            self.slider_max_theta.setValue(theta_min);
            theta_min = theta_max;

        self.theta_min = theta_min;
        self.theta_max = theta_max;

        self.label_theta_min.setText("THETA LOW: " + np.str(theta_min))
        self.label_theta_max.setText("THETA HI     : " + np.str(theta_max))

        filtered_gradient = self.filter_threshold(thresh=(theta_min*np.pi/180., theta_max*np.pi/180.))
        self.img_3.set_data(filtered_gradient)
        self.ax3.figure.canvas.draw()

        smin = self.slider_S_min.value()
        smax = self.slider_S_max.value()

        if smax < smin:
            self.slider_S_max.setValue(smin);
            smin = theta_max;

        self.smin = smin;
        self.smax = smax;

        s_binary_t = np.zeros_like(self.S_channel)
        s_binary_t[(self.S_channel >= self.smin) & (self.S_channel <= self.smax)] = 1
        self.img_4.set_data(s_binary_t*225)
        self.ax4.figure.canvas.draw()
        self.label_sat_min.setText("SAT LOW: " + np.str(self.smin));
        self.label_sat_max.setText("SAT HI     : " + np.str(self.smax));

        '''
        now we create the final binary image
        '''

        combined_hue_grad_binary = np.zeros_like(self.arct_img)
        combined_hue_grad_binary[(combined_binary == 255) & (filtered_gradient == self.derivative_view_thresh) & (s_binary_t == self.saturation_threshold)] = 255
        self.img_5.set_data(combined_hue_grad_binary)
        self.ax5.figure.canvas.draw()

        self.result = np.copy(combined_hue_grad_binary);





    def buttonPressed(self):

        self.dict['hue1_low']  = self.hue1_min;
        self.dict['hue1_high'] = self.hue1_max;
        self.dict['hue2_low']  = self.hue2_min;
        self.dict['hue2_high'] = self.hue2_max;
        self.dict['theta_min'] = self.theta_min;
        self.dict['theta_max'] = self.theta_max;
        self.dict['saturation_min'] = self.smin;
        self.dict['saturation_max'] = self.smax;



        print("saving params at " + pickle_dest);

        pickle.dump(self.dict, open(pickle_dest, "wb"));

        cv2.imwrite("../output_images/filtered.jpg", self.result)

        print("params saved");



if __name__ == "__main__":

    qapp = QtWidgets.QApplication.instance()
    if not qapp:
        qapp = QtWidgets.QApplication(sys.argv);


    dc  = pickle.load(open("../pickle_files/calib_first_pass.pickle","rb"))
    
    apphue = AppWindowHue(dictin=data)
    apphue.show()
    apphue.activateWindow()
    apphue.raise_()
    qapp.exec_()
