import numpy as np
import cv2
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from collections import deque
from itertools import islice

from matplotlib.lines import Line2D 


import pickle

# img = cv2.imread('../examples/warped-example.jpg')
# grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

'''
@precentage : how much of the image from the top is cropeed out 
clear
'''
def sliceimg(img, precentage):
    image_length = np.shape(img)[0]
    x_ = np.int(image_length/100.)*precentage
    sliced = img[x_:,:]
    ret = np.copy(sliced)

    return ret


'''
Function to create a slice and it's histogram 
'''
def sliceimg_and_histogram(img, precentage):
    image_length = np.shape(img)[0]
    slice_start = np.int((image_length/100.)*precentage)
    x_ = slice_start if (image_length - slice_start)//nwindows > 0 else image_length - nwindows # to stop crashing poly-fit 
                                                                                                # due to insufficient rows w.r.t nwindows
                                                                                                # we alloate the minimum slice hight to be a function of nwindows.
                                                                                                # however it may still throw an error if there are no 'lane-pixels'
                                                                                                # within the slice. 
    sliced = img[x_:,:]
    norm_sliced = sliced/255
    thist = np.sum(norm_sliced, axis=0)
    retimg = np.copy(sliced)

    return retimg, thist


def update_slicedimg(val):
    sliced, hist = sliceimg_and_histogram(grayimg, np.int(val))


    yy =np.zeros_like(sliced)
    out_img = np.dstack((sliced, sliced, sliced))
    midpoint = np.int(hist.shape[0]//2)


    
    leftx_base = np.argmax(hist[:midpoint])
    rightx_base = np.argmax(hist[midpoint:]) + midpoint 

    window_height = np.int(sliced.shape[0]//nwindows)


    leftx_current = leftx_base
    rightx_curernt = rightx_base

  
    max_lane_pixels = 30000

    LHS_lane_x_coords = deque(maxlen=max_lane_pixels)
    LHS_lane_y_coords = deque(maxlen=max_lane_pixels)

    RHS_lane_x_coords = deque(maxlen=max_lane_pixels)
    RHS_lane_y_coords = deque(maxlen=max_lane_pixels)

    # step through the windows 

    alfa = 0.6

    for window in range(nwindows):


        winy_low = sliced.shape[0] - (window + 1) * window_height
        winy_high = sliced.shape[0] - window * window_height

        winx_left_low = leftx_current - margin
        winx_left_high = leftx_current  + margin

        winx_right_low = rightx_curernt - margin 
        winx_right_high = rightx_curernt + margin 

        cv2.rectangle(out_img, (winx_left_low, winy_low), (winx_left_high, winy_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (winx_right_low, winy_low), (winx_right_high, winy_high), (0, 255, 0), 2)

        '''
            y coordinates are determined by the window slice, to find the x-coordinates 
            we wil get the slice and check how much non-zero pixels fall within the rectangle
        '''

        subset_ = sliced[(nwindows - window - 1 ) * window_height: (nwindows-window ) * window_height, :]
        subset_nonzero = subset_.nonzero()
        subset_nzx = subset_nonzero[1]
        subset_nzy = subset_nonzero[0]

        non_zero_rect_intersects_left  = ((subset_nzx >= winx_left_low)  & (subset_nzx <= winx_left_high)).nonzero()[0]
        non_zero_rect_intersects_right = ((subset_nzx >= winx_right_low) & (subset_nzx <= winx_right_high)).nonzero()[0]

        '''
        now we know the nonzero x coordinates and the index of pixels that are within the box and are not zero
        and these indexes are valid for the base image.

        '''

        left_lane_xs = subset_nzx[non_zero_rect_intersects_left]
        left_lane_ys = subset_nzy[non_zero_rect_intersects_left] + (nwindows - window - 1 ) * window_height
        right_lane_xs = subset_nzx[non_zero_rect_intersects_right]
        right_lane_ys = subset_nzy[non_zero_rect_intersects_right] + (nwindows - window - 1 ) * window_height

        LHS_lane_x_coords.extend(left_lane_xs)
        LHS_lane_y_coords.extend(left_lane_ys)

        RHS_lane_x_coords.extend(right_lane_xs)
        RHS_lane_y_coords.extend(right_lane_ys)

        if(len(non_zero_rect_intersects_left)) > minpix:
            leftx_current = np.int(leftx_current*(1-alfa) + np.mean(subset_nzx[non_zero_rect_intersects_left]) * alfa)
        if(len(non_zero_rect_intersects_right)) > minpix:
            rightx_curernt = np.int(rightx_curernt*(1-alfa) + np.mean(subset_nzx[non_zero_rect_intersects_right]) * alfa)


    out_img[LHS_lane_y_coords, LHS_lane_x_coords] = (255,0,200)
    out_img[RHS_lane_y_coords, RHS_lane_x_coords] = (255,0,100)

    '''
        here we assume there are enough points, implement checks in a more serious implementation.
    '''

    left_fit  = np.polyfit(LHS_lane_y_coords, LHS_lane_x_coords, 2)
    right_fit = np.polyfit(RHS_lane_y_coords, RHS_lane_x_coords, 2)

    ploty = np.linspace(0, sliced.shape[0] - 1, sliced.shape[0])

    try:
        left_fitx  = left_fit[0]*ploty**2  + left_fit[1]  * ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1] * ploty + right_fit[2]
    except TypeError:
        print("polyfit_failed")

       
    # LHS_lane_y_coords_m = [i * ym_per_pix for i in LHS_lane_y_coords]
    # LHS_lane_x_coords_m = [i * xm_per_pix for i in LHS_lane_x_coords] 

    # RHS_lane_y_coords_m = [i * ym_per_pix for i in RHS_lane_y_coords]
    # RHS_lane_x_coords_m = [i * xm_per_pix for i in RHS_lane_x_coords]


    # left_fit_cr  = np.polyfit(LHS_lane_y_coords_m , LHS_lane_x_coords_m , 2)
    # right_fit_cr = np.polyfit(RHS_lane_y_coords_m , RHS_lane_x_coords_m , 2)
    y_eval = np.max(ploty)

    '''
    Testing an alternative way to scale pixels:
    polyfit function finds coefficients to fit a polynomial that looks like 
    x = p[0] * y^2 + p[1] * y + p[2]
    this polynomial can be plotted in a pixel map with x and (horizontal) y(vertical) axes.
    if we want to convert that to meters we will need to parametric relations 
    x' = x * m_per_px_in_x_dir
    y' = y * m_per_px_in_y_dir
    to convert the modified pixmap to a cartesian map of meters we can use the equations above to look like below.
    x' = m_per_px_in_x_dir ( p[0] * (y'/m_per_px_in_y_dir)^2 + p[1] * (y'/m_per_px_in_y_dir) + p[2])
    x' = p[0] * m_per_px_in_x_dir * (1/m_per_px_in_y_dir)^2 * y'^2 + p[1] * m_per_px_in_x_dir * (1/m_per_px_in_y_dir) * y' + p[2] * m_per_px_in_x_dir

    now we can modify the coefficients 
    
    p'[0] = p[0] * m_per_px_in_x_dir * (1/m_per_px_in_y_dir)^2
    p'[1] = p[1] * m_per_px_in_x_dir * (1/m_per_px_in_y_dir
    p'[2] = p[2] * m_per_px_in_x_dir

    The new polynomial becomes 

    x' = p'[0] * y'^2 + p'[1] * y' + p'[2]

    '''

    left_fit_m  = np.copy(left_fit)
    right_fit_m = np.copy(right_fit)


    left_fit_m[0] = left_fit_m[0] * xm_per_pix * (1/ym_per_pix)**2
    left_fit_m[1] = left_fit_m[1] * xm_per_pix/ym_per_pix
    left_fit_m[2] = left_fit_m[2] * xm_per_pix

    right_fit_m[0] = right_fit_m[0] * xm_per_pix * (1/ym_per_pix)**2
    right_fit_m[1] = right_fit_m[1] * xm_per_pix/ym_per_pix
    right_fit_m[2] = right_fit_m[2] * xm_per_pix


    y_eval_m = y_eval * ym_per_pix;
    left_curverad_m  = np.sqrt((1 + (2*left_fit_m[0]*y_eval_m + left_fit_m[1])**2)**3)/np.fabs(2*left_fit_m[0])
    right_curverad_m = np.sqrt((1 + (2*right_fit_m[0]*y_eval_m + right_fit_m[1])**2)**3)/np.fabs(2*right_fit_m[0])

    # left_curverad = np.sqrt((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**3)/np.fabs(2*left_fit_cr[0])
    # right_curverad = np.sqrt((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**3)/np.fabs(2*right_fit_cr[0])

    # avg_radius = (left_curverad + right_curverad) / 2.
    avg_radius_m = (left_curverad_m + right_curverad_m) / 2.

    fontsize = 1.1
    # cv2.putText(out_img, 'Left R  = %.2f m' % left_curverad , (midpoint- 100,200)    , cv2.FONT_HERSHEY_SIMPLEX, fontsize, (252, 186, 3), 2 , cv2.LINE_AA)
    # cv2.putText(out_img, 'Right R = %.2f m' % right_curverad, (midpoint -100, 150)  , cv2.FONT_HERSHEY_SIMPLEX, fontsize, (252, 186, 3), 2 , cv2.LINE_AA)
    cv2.putText(out_img, 'Avg R = %.1f m' % (avg_radius_m)    , (midpoint -100, 50)  , cv2.FONT_HERSHEY_SIMPLEX, fontsize, (252, 186, 3), 2 , cv2.LINE_AA)

    '''
        Vehicle position detection:
        Here we assume the Image centre is aligned with the vehicle centre. 
        Our goal is to find the pixel offset between the midpoint of lanes and the midpoint of the vehicle.

        Note: The actual useful centre point is a function of speed and the delay compensation, however we use y_eval for this project

    '''


    left_x  = left_fit[0]*y_eval**2  + left_fit[1]  * y_eval + left_fit[2]
    right_x = right_fit[0]*y_eval**2 + right_fit[1] * y_eval + right_fit[2]

    # print("left_x = " + np.str(left_x) + " right_x = " + np.str(right_x));

    lane_centre = (left_x + right_x) // 2;
    vehicle_offset = (midpoint - lane_centre) *1.  # This is in pixels

    # print("VC = " + np.str(midpoint) + " LC = " + np.str(lane_centre));

    msg  = "Vehicle is " + "%.2f" % np.abs(vehicle_offset * xm_per_pix * 1.)  + "m " +  ("left" if (vehicle_offset > 0) else "right") + "of centre"; 
    cv2.putText(out_img, msg   , (midpoint -100, 100)  , cv2.FONT_HERSHEY_SIMPLEX, fontsize, (252, 186, 3), 2 , cv2.LINE_AA)




    left_lane.set_data(left_fitx, ploty)
    right_lane.set_data(right_fitx, ploty)
    
    im.set_data(out_img)
    ex =(-0.5, 1279.5, np.shape(out_img)[0], -0.5)
    im.set_extent(ex) # first get extent to get the values 

    l_.set_ydata(hist)
    ax.set_ylim(0, np.max(hist) * 1.2)
    fig.canvas.draw_idle()


if __name__ == "__main__":


    img = cv2.imread('../examples/warped-example.jpg')
    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    plt.subplots_adjust(left=0.1, bottom=0.30)
    ax.margins(x=0)

    # set x axis with the image cols 
    x_ = np.arange(0, len(grayimg[-1]), 1)
    y_ = np.sum(grayimg, axis=0) # axis 0  is the rows axis, will sum each row along each column 
    l_, =  ax.plot(x_, y_)

    llx = []
    lly = []

    linewidth = 3
    linecolor = 'yellow'
    left_lane = Line2D(llx, lly, lw=linewidth, color=linecolor)
    right_lane = Line2D(llx, lly, lw=linewidth, color=linecolor)
    ax2.add_line(left_lane)
    ax2.add_line(right_lane)

    # plt.plot(left_fitx, ploty, color='yellow')
    # print(ax.get_extent())

    axcolor = 'lightgoldenrodyellow'
    vcrop_from_bottom   = plt.axes([0.1, 0.20, 0.8, 0.03], facecolor=axcolor)

    # vinit = 50.0
    vh_adjust = Slider(vcrop_from_bottom, 'V-height', 0, 100.0,  valinit=0, valstep=1)

    sliced = sliceimg(img, 0)
    im = ax2.imshow(sliced, cmap='gray')
    ax2.set_anchor('SW')

    nwindows = 9
    margin = 50
    minpix = 50
    ym_per_pix = 30/720  # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    

    vh_adjust.on_changed(update_slicedimg)

    resetax = plt.axes([0.8, 0.10, 0.1, 0.04])
    button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

    def reset(event):
        vh_adjust.reset()

    button.on_clicked(reset)

    plt.show()