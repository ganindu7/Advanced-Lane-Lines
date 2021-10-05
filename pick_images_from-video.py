import pickle
import numpy as np 
import cv2
import sys
import time
import matplotlib.pyplot as plt
import glob
import re
import os.path

from moviepy.editor import VideoFileClip
from IPython.display import HTML

# loaded_params = pickle.load(open("fianal_params_for_lane_lines.pickle", "rb"))

# mtx   = loaded_params['mtx_']
# dist  = loaded_params['dist_']

# sobelx_lo_thresh    = loaded_params['sobelX_low_thresh']
# sobelx_hi_threshold = loaded_params['sobelX_hi_thresh']
# sat_lo_threshold    = loaded_params['sat_low_thresh']
# sat_hi_threshold    = loaded_params['sat_hi_thresh']
# M_perspective       = loaded_params['M_perspective_transform']

def processImage(image, outdir_):

    img = image    
    
    timg = np.copy(img)

    processImage.count += 1

    rem = np.mod(processImage.count, 30)

    if(rem == 0):

        processImage.savecount += 1
        outval = np.str(processImage.savecount)

        outname = outdir_ + '/out_'+outval+'.jpg'

        # print('writing, ', outname)

        cv2.imwrite(outname, timg[:,:,::-1])


    return timg
processImage.count = 0
processImage.savecount = 0



# input_video = '/home/ganindu/Workspace/CarND/projects/lanelines/Webcam_videos/converted/2020-07-02-165944.mp4'
# output_video = 'output.mp4'

# clip1 = VideoFileClip(input_video)
# processed_clip = clip1.fl_image(processImage)
# processed_clip.write_videofile(output_video, audio=False)

input_string = '/home/ganindu/Workspace/CarND/projects/class-projects/CarND-Advanced-Lane-Lines/harder_challenge_video.mp4'

foldername = re.search(r'.+\/', input_string).group()
outdir = foldername+'processed/'
timestamp = time.strftime("%Y%m%d-%H%M%S")
outdir += timestamp

if not os.path.isdir(outdir):
    print("creating folder: ", outdir)
    os.makedirs(outdir)
    os.makedirs(outdir+"/images")

videos = glob.glob(input_string)
for video in videos:
    
    filename = re.search(r'[^/]+$', video).group()
    outfile_name = outdir+'/OUT_'+filename 

    print('\n\nprocessing ', filename)

    if not os.path.isfile(outfile_name):
        clip = VideoFileClip(video)
        processed_clip = clip.fl_image(lambda image: processImage(image, outdir+"/images"))
        processed_clip.write_videofile(outfile_name, audio=False)
        
    else:
        print("File exists: ", outfile_name)



