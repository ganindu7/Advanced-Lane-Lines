import pickle
from PyQt5.QtWidgets import QMainWindow
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


data = pickle.load(open("../pickle_files/calib_first_pass.pickle", "rb"));

class AppWindowHue(QMainWindow):
    def __init__(self, dictin) -> None:
        super().__init__()

        self.mtx  = dictin['mtx_'];
        self.dist = dictin['dist_'];

        dictin['hue1_low']  = 0;
        dictin['hue1_high'] = 0;

        dictin['hue2_low']  = 0;
        dictin['hue2_high'] = 0;

        self.dict = dictin;

        img            = cv2.imread('../test_images/test6.jpg', cv2.IMREAD_COLOR);
        self.gray      = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
        hls            = cv2.cvtColor(img, cv2.COLOR_BGR2HLS);
        self.H_channel = hls[:,:,0];

        self._main = QtWidgets.QWidget();
        self.setCentralWidget(self._main);

        self.grid = QtWidgets.QGridLayout(self._main);

        self.static_canvas_left = FigureCanvas(Figure(figsize=(10,6)))
        self.grid.addWidget(self.static_canvas_left, 1, 1);
        self.ax1 = self.static_canvas_left.figure.subplots();

        self.static_canvas_right = FigureCanvas(Figure(figsize=(10,6)))
        self.grid.addWidget(self.static_canvas_right, 1, 2);
        self.ax2 = self.static_canvas_right.figure.subplots();

        self.hue1_min = 0;
        self.hue1_max = 100;

        self.hue2_min = 0;
        self.hue2_max = 100;


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

        self.label_hue_1_floor = QLabel(self)
        self.label_hue_1_floor.setText("HUE 1 MIN: 0")
        self.label_hue_1_floor.setAlignment(Qt.AlignBottom | Qt.AlignLeft)
        self.grid.addWidget(self.label_hue_1_floor, 8, 1)

        self.label_hue_1_ceil = QLabel(self)
        self.label_hue_1_ceil.setText("HUE 1 MAX: 0")
        self.grid.addWidget(self.label_hue_1_ceil, 9, 1)


        self.label_hue_2_floor = QLabel(self)
        self.label_hue_2_floor.setText("HUE 2 MIN: 0")
        self.label_hue_2_floor.setAlignment(Qt.AlignBottom | Qt.AlignLeft)
        self.grid.addWidget(self.label_hue_2_floor, 8, 2)

        self.label_hue_2_ceil = QLabel(self)
        self.label_hue_2_ceil.setText("HUE 2 MAX: 0")
        self.label_hue_2_ceil.setAlignment(Qt.AlignBottom)
        self.grid.addWidget(self.label_hue_2_ceil, 9, 2)

        h1_binary_t = np.zeros_like(self.H_channel);
        h2_binary_t = np.zeros_like(self.H_channel);
        h1_binary_t[(self.H_channel >= self.hue1_min) & (self.H_channel <= self.hue1_max)] = 1;
        h2_binary_t[(self.H_channel >= self.hue2_min) & (self.H_channel <= self.hue2_max)] = 1;

        self.composite_stacked = np.dstack((np.zeros_like(self.H_channel), h1_binary_t, h2_binary_t)) * 255;
        self.imglhs = self.ax1.imshow(self.composite_stacked, cmap="gray");

        combined_binary = np.zeros_like(self.H_channel);
        combined_binary[(h1_binary_t == 1) | (h2_binary_t == 1)] = 255;
        self.imgrhs = self.ax2.imshow(combined_binary, cmap="gray");

        tx = 'Save threholds'
        self.button = QPushButton(tx, self);
        self.bwidth = self.button.fontMetrics().boundingRect(tx).width() + 120;
        self.button.setMaximumWidth(self.bwidth);
        self.button.clicked.connect(self.buttonPressed);
        self.grid.addWidget(self.button, 9, 2, 1, 1, Qt.AlignRight);


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

        composite_stacked = np.dstack((np.zeros_like(self.H_channel), h1_binary_t, h2_binary_t)) * 255;
        self.imglhs.set_data(composite_stacked);
        self.ax1.figure.canvas.draw(); # Left


        combined_binary = np.zeros_like(self.H_channel);
        combined_binary[(h1_binary_t == 1) | (h2_binary_t == 1)] = 255;
        self.imgrhs.set_data(combined_binary);
        self.ax2.figure.canvas.draw(); # Right

    def buttonPressed(self):

        self.dict['hue1_low']  = self.hue1_min;
        self.dict['hue1_high'] = self.hue1_max;
        self.dict['hue2_low']  = self.hue2_min;
        self.dict['hue2_high'] = self.hue2_max;

        pickle.dump(self.dict, open("../pickle_files/hue_filtered_temp.pickle", "wb"));

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
