from typing import Text
from matplotlib.patches import Polygon
import numpy as np
import cv2
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from collections import deque
from itertools import islice

from matplotlib.lines import Line2D 
from matplotlib.patches import Polygon

import pickle

offset_msg = "test"
radius_msg = "test"

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

    return retimg, thist, x_

def update_slicedimg(val):
    sliced, hist, offset_to_clip_point = sliceimg_and_histogram(warped, np.int(val))

    sliced = sliced/255;

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

    # min_margin = 15
    t_margin = margin

    for window in range(nwindows):

        # new_margin = t_margin - window * 1
        # t_margin = new_margin if( new_margin >= min_margin) else min_margin

        winy_low = sliced.shape[0] - (window + 1) * window_height
        winy_high = sliced.shape[0] - window * window_height

        winx_left_low = leftx_current - t_margin
        winx_left_high = leftx_current  + t_margin

        winx_right_low = rightx_curernt - t_margin 
        winx_right_high = rightx_curernt + t_margin 


        '''
            y coordinates are determined by the window slice, to find the x-coordinates 
            we will get the slice and check how much non-zero pixels fall within the rectangle
        '''

        subset_ = sliced[(nwindows - window - 1 ) * window_height: (nwindows-window ) * window_height, :]
        subset_nonzero = subset_.nonzero()
        subset_nzx = subset_nonzero[1]
        subset_nzy = subset_nonzero[0]

        non_zero_rect_intersects_left  = ((subset_nzx >= winx_left_low)  & (subset_nzx <= winx_left_high)).nonzero()[0]
        non_zero_rect_intersects_right = ((subset_nzx >= winx_right_low) & (subset_nzx <= winx_right_high)).nonzero()[0]

        print("window:", window, " left = ", np.size(non_zero_rect_intersects_left), " right = ", np.size(non_zero_rect_intersects_right));

        '''
            now we know the nonzero x coordinates and the index of pixels that are within the box and are not zero
            and these indexes are valid for the base image.

        '''

        left_lane_xs = subset_nzx[non_zero_rect_intersects_left]
        left_lane_ys = subset_nzy[non_zero_rect_intersects_left] + (nwindows - window - 1 ) * window_height
        right_lane_xs = subset_nzx[non_zero_rect_intersects_right]
        right_lane_ys = subset_nzy[non_zero_rect_intersects_right] + (nwindows - window - 1 ) * window_height

        if(len(non_zero_rect_intersects_left)) > minpix:
            leftx_current = np.int(leftx_current*(1-alfa) + np.mean(subset_nzx[non_zero_rect_intersects_left]) * alfa)
            LHS_lane_x_coords.extend(left_lane_xs)
            LHS_lane_y_coords.extend(left_lane_ys)
            cv2.rectangle(out_img, (winx_left_low, winy_low), (winx_left_high, winy_high), (0, 255/255, 0), 2)
        else:
            cv2.rectangle(out_img, (winx_left_low, winy_low), (winx_left_high, winy_high), (1., 0., 0), 2)

        if(len(non_zero_rect_intersects_right)) > minpix:
            rightx_curernt = np.int(rightx_curernt*(1-alfa) + np.mean(subset_nzx[non_zero_rect_intersects_right]) * alfa)
            RHS_lane_x_coords.extend(right_lane_xs)
            RHS_lane_y_coords.extend(right_lane_ys)
            cv2.rectangle(out_img, (winx_right_low, winy_low), (winx_right_high, winy_high), (0., 1., 0), 2)
        else:
            cv2.rectangle(out_img, (winx_right_low, winy_low), (winx_right_high, winy_high), (1., 0., 0), 2)
    
    print("total left = ", np.size(LHS_lane_x_coords), " total right = ", np.size(RHS_lane_x_coords));

    do_left  = True if np.size(LHS_lane_x_coords) > minpix else False
    do_right = True if np.size(RHS_lane_x_coords) > minpix else False
    
    out_img[LHS_lane_y_coords, LHS_lane_x_coords] = (255/255,0,200/255)
    out_img[RHS_lane_y_coords, RHS_lane_x_coords] = (255/255,0,100/255)

    '''
        here we assume there are enough points, implement checks in a more serious implementation.
    '''

    try:
        if do_left or do_right:
            ploty = np.linspace(0, sliced.shape[0] - 1, sliced.shape[0])
        if do_left:
            left_fit  = np.polyfit(LHS_lane_y_coords, LHS_lane_x_coords, 2)        
            left_fitx  = left_fit[0]*ploty**2  + left_fit[1]  * ploty + left_fit[2]
            left_fit_m  = np.copy(left_fit)
            left_fit_m[0] = left_fit_m[0] * xm_per_pix * (1/ym_per_pix)**2
            left_fit_m[1] = left_fit_m[1] * xm_per_pix/ym_per_pix
            left_fit_m[2] = left_fit_m[2] * xm_per_pix

        if do_right:
            right_fit = np.polyfit(RHS_lane_y_coords, RHS_lane_x_coords, 2)
            right_fitx = right_fit[0]*ploty**2 + right_fit[1] * ploty + right_fit[2]
            right_fit_m = np.copy(right_fit)
            right_fit_m[0] = right_fit_m[0] * xm_per_pix * (1/ym_per_pix)**2
            right_fit_m[1] = right_fit_m[1] * xm_per_pix/ym_per_pix
            right_fit_m[2] = right_fit_m[2] * xm_per_pix

    except TypeError:
        print("polyfit_failed")
        raise SystemExit

    '''
    y eval makes sense if 
    '''
    y_eval = None
    left_r_contrib = 0
    right_r_contrib = 0


    if do_left or do_right:
        y_eval = np.max(ploty)
        y_eval_m = y_eval * ym_per_pix;

    if do_left:
        left_curverad_m  = np.sqrt((1 + (2*left_fit_m[0]*y_eval_m + left_fit_m[1])**2)**3)/np.fabs(2*left_fit_m[0])
        left_r_contrib = left_curverad_m/2;
        left_x  = left_fit[0]*y_eval**2  + left_fit[1]  * y_eval + left_fit[2]

    if do_right:
        right_curverad_m = np.sqrt((1 + (2*right_fit_m[0]*y_eval_m + right_fit_m[1])**2)**3)/np.fabs(2*right_fit_m[0])
        right_r_contrib = right_curverad_m/2;
        right_x = right_fit[0]*y_eval**2 + right_fit[1] * y_eval + right_fit[2]

    if y_eval is not None:
        avg_radius_m = left_r_contrib + right_r_contrib;

    fontsize = 1.1
    # cv2.putText(out_img, 'Left R  = %.2f m' % left_curverad , (midpoint- 100,100)    , cv2.FONT_HERSHEY_SIMPLEX, fontsize, (252, 186, 3), 2 , cv2.LINE_AA)
    # cv2.putText(out_img, 'Right R = %.2f m' % right_curverad, (midpoint -100, 150)  , cv2.FONT_HERSHEY_SIMPLEX, fontsize, (252, 186, 3), 2 , cv2.LINE_AA)
    # cv2.putText(out_img, 'Avg R = %.1f m' % avg_radius_m    , (midpoint -100, 50)  , cv2.FONT_HERSHEY_SIMPLEX, fontsize, (252, 186, 3), 2 , cv2.LINE_AA)

    if do_left and do_right:

        textax_2.set_text('Avg R = %.1f m' % avg_radius_m)

        '''
            Vehicle position detection:
            Here we assume the Image centre is aligned with the vehicle centre. 
            Our goal is to find the pixel offset between the midpoint of lanes and the midpoint of the vehicle.

            Note: The actual useful centre point is a function of speed and the delay compensation, however we use y_eval for this project

        '''


        lane_centre = (left_x + right_x) // 2;
        vehicle_offset = (midpoint - lane_centre) *1.

        msg  = "Vehicle is " + "%.2f" % np.abs(vehicle_offset * xm_per_pix * 1.)  + "m " +  ("left" if (vehicle_offset > 0) else "right") + " of centre"; 
        # cv2.putText(out_img, msg   , (midpoint -100, 100)  , cv2.FONT_HERSHEY_SIMPLEX, fontsize, (252, 186, 3), 2 , cv2.LINE_AA)
        textax_1.set_text(msg)

        global radius_msg 
        global offset_msg

        radius_msg = 'Avg R = %.1f m' % avg_radius_m 
        offset_msg = msg



    # cv2.putText(ax3, 'Avg R = %.1f m' % avg_radius_m    , (midpoint -100, 50)  , cv2.FONT_HERSHEY_SIMPLEX, fontsize, (252, 186, 3), 2 , cv2.LINE_AA)
    # cv2.putText(ax3, msg   , (midpoint -100, 100)  , cv2.FONT_HERSHEY_SIMPLEX, fontsize, (252, 186, 3), 2 , cv2.LINE_AA)

    if do_left:
        left_lane.set_data(left_fitx, ploty)
    if do_right:
        right_lane.set_data(right_fitx, ploty) # We can flip this and use that for the polygon too

    if do_left and do_right:

        ploty_augmented = ploty + offset_to_clip_point # for disply purposes we position the slice in the base image
        rightx_flipped = np.flipud(right_fitx)
        xs = np.concatenate((left_fitx, rightx_flipped));
        y_flipped = np.flipud(ploty_augmented)
        ys = np.concatenate((ploty_augmented , y_flipped));
        poly_points = np.concatenate((xs.reshape(xs.shape[0], 1), ys.reshape(ys.shape[0], 1)), axis=1, out=None) # can't set datatype in this version (< 1.20.0)
        hom_augment_ones = np.ones((poly_points.shape[0], 1), dtype=float)
        hom_poly_points = np.concatenate((poly_points, hom_augment_ones), axis=1).T # we have to transpose because we multilpy with the 3*3 matrix
        img_lane_poly = m_ppt_inv.dot(hom_poly_points)
        img_lane_poly_hom = (img_lane_poly/img_lane_poly[-1]).T
        img_lane_poly_xy = img_lane_poly_hom[:,0:2]

        lane_area_polygon.set_xy(img_lane_poly_xy)
    
    im.set_data(out_img)
    ex =(-0.5, warp_x_expand , np.shape(out_img)[0], -0.5)
    im.set_extent(ex) # first get extent to get the values 

    l_.set_ydata(hist)
    ax.set_ylim(0, np.max(hist) * 1.2)
    fig.canvas.draw_idle()


if __name__ == "__main__":


    data  = pickle.load(open("../pickle_files/project_data/fianal_params_for_lane_lines.pickle", "rb"))
    mtx   = data['mtx_']
    dist  = data['dist_']
    m_ppt = data['M_perspective_transform']
    
    m_ppt_inv = np.linalg.pinv(m_ppt)

    theta_min = data['theta_min'];
    theta_max = data['theta_max'];

    hue1_min  = data['hue1_low'];
    hue1_max  = data['hue1_high'];

    hue2_min  = data['hue2_low'];
    hue2_max  = data['hue2_high'];

    smin      = data['saturation_min'];
    smax      = data['saturation_max'];

    dst_img_width  = data['dst_img_width']
    dst_img_height = data['dst_img_height']

    derivative_view_thresh = 1;
    saturation_threshold = 1;

    offset_msg = "test"
    radius_msg = "test"

    '''
        No safety for negative values 
    '''
    warp_x_expand_ = 0  # additional x space in the warped image
    warp_y_expand_ = 0



    # road_img = cv2.imread('../test_images/straight_lines2.jpg')
    road_img = cv2.imread('../test_images/test6.jpg')
    # road_img = cv2.imread('../test_images/out_19.png')


    undistorted_road_img = cv2.undistort(road_img, mtx, dist, None, mtx)

    hls        = cv2.cvtColor(undistorted_road_img, cv2.COLOR_BGR2HLS);
    H_channel  = hls[:,:,0];
    S_channel  = hls[:,:,2];

    h1_binary_t = np.zeros_like(H_channel);
    h2_binary_t = np.zeros_like(H_channel);
    h1_binary_t[(H_channel >= hue1_min) & (H_channel <= hue1_max)] = 1;
    h2_binary_t[(H_channel >= hue2_min) & (H_channel <= hue2_max)] = 1;

    combined_binary = np.zeros_like(H_channel);
    combined_binary[(h1_binary_t == 1) | (h2_binary_t == 1)] = 255;

    sobel_kernel_size = 15;
    gray         = cv2.cvtColor(undistorted_road_img, cv2.COLOR_BGR2GRAY);
    norm_image   = cv2.normalize(gray, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    blurred_gray = cv2.GaussianBlur(norm_image, (3,3), cv2.BORDER_DEFAULT)
    sobel_x           = cv2.Sobel(blurred_gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel_size)
    sobel_y           = cv2.Sobel(blurred_gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel_size)
    abs_sobel_x = np.absolute(sobel_y)
    abs_sobel_y = np.absolute(sobel_x)
    scaled_sobel_x = (255*abs_sobel_x/np.max(abs_sobel_x))
    scaled_sobel_y = (255*abs_sobel_y/np.max(abs_sobel_y))
    arct_img = np.arctan2(scaled_sobel_y, scaled_sobel_x)

    filtered_gradient = np.zeros_like(arct_img)
    filtered_gradient[(arct_img >= theta_min*np.pi/180) & (arct_img <= theta_max*np.pi/180)] = derivative_view_thresh ;

    s_binary_t = np.zeros_like(S_channel)
    s_binary_t[(S_channel >= smin) & (S_channel <= smax)] = saturation_threshold

    combined_hue_sat_grad_binary = np.zeros_like(arct_img)
    combined_hue_sat_grad_binary[(combined_binary == 255) & (filtered_gradient == derivative_view_thresh) & (s_binary_t == saturation_threshold)] = 255


    warp_x_expand = warp_x_expand_ + dst_img_width  if (np.shape(combined_hue_sat_grad_binary)[1] > warp_x_expand_ + dst_img_width) else np.shape(combined_hue_sat_grad_binary)[1]
    warp_y_expand = warp_x_expand_ + dst_img_height if (np.shape(combined_hue_sat_grad_binary)[0] > warp_y_expand_ + dst_img_height) else np.shape(combined_hue_sat_grad_binary)[0]

    # print("wx = ", warp_x_expand, " wy = ", warp_y_expand)
    warped = cv2.warpPerspective(combined_hue_sat_grad_binary, m_ppt, (warp_x_expand, warp_y_expand))        

    fig, (ax, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6))
    plt.subplots_adjust(left=0.1, bottom=0.30)
    ax.margins(x=0)


    im_src = ax3.imshow(cv2.cvtColor(undistorted_road_img, cv2.COLOR_BGR2RGB))

    lane_polygon = np.empty((0, 2))
    lane_area_polygon = Polygon(lane_polygon,  facecolor=None, fill=False, alpha=0.4, hatch='//', color='greenyellow')
    ax3.add_patch(lane_area_polygon)

    # set x axis with the image cols 
    x_ = np.arange(0, len(warped[-1]), 1)
    y_ = np.sum(warped, axis=0) # axis 0  is the rows axis, will sum each row along each column 
    l_, =  ax.plot(x_, y_)

    # llx = []
    # lly = []

    linewidth = 3
    linecolor = 'yellow'
    
    # left_lane = Line2D(llx, lly, lw=linewidth, color=linecolor)
    # right_lane = Line2D(llx, lly, lw=linewidth, color=linecolor)

    left_lane  = Line2D(np.empty((0, 1)), np.empty((0, 1)), lw=linewidth, color=linecolor)
    right_lane = Line2D(np.empty((0, 1)), np.empty((0, 1)), lw=linewidth, color=linecolor)

    ax2.add_line(left_lane)
    ax2.add_line(right_lane)

    axcolor = 'lightgoldenrodyellow'
    vcrop_from_bottom   = plt.axes([0.1, 0.20, 0.8, 0.03], facecolor=axcolor)

    vh_adjust = Slider(vcrop_from_bottom, 'V-height', 0, 100.0,  valinit=0, valstep=1)

    sliced = sliceimg(warped, 0)
    im = ax2.imshow(sliced, cmap='gray')
    ax2.set_anchor('SW')

    nwindows = 9
    margin = 40
    minpix = 15
    ym_per_pix = 3.0/210 # meters per pixel in y dimension, supplied data
    xm_per_pix = 3.7/259 # meters per pixel in x dimension, supplied data

    vartext =  None
    textax_1 = plt.text(0, -1.5, vartext)
    textax_2 = plt.text(0, -2.5, vartext)

    vh_adjust.on_changed(update_slicedimg)

    resetax = plt.axes([0.8, 0.10, 0.1, 0.04])
    button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

    saveax = plt.axes([0.65, 0.10, 0.1, 0.04])
    button_save = Button(saveax, 'Save', color=axcolor, hovercolor='0.975')

    
    update_slicedimg(0)

    def reset(event):
        vh_adjust.reset()

    '''
    repeted saves overwrite the label 
    @todo fix repeted saves, overwritng label (if you're really bothered lol)
    '''

    def save(event):
        print("saving..,", offset_msg, radius_msg)
        ax3.text(0.35, 0.9, offset_msg, horizontalalignment='left', verticalalignment='center', color="white", transform=ax3.transAxes)
        ax3.text(0.35, 0.8, radius_msg, horizontalalignment='left', verticalalignment='center', color="white", transform=ax3.transAxes)
        extent = ax3.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig("foo"+ (np.str(save.count) if (save.count > 0) else '') +".png", bbox_inches=extent, dpi=1000)
        save.count += 1
    save.count = 0


    button.on_clicked(reset)
    button_save.on_clicked(save)

    plt.show()

