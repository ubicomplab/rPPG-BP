import numpy as np
import cv2
from skimage.util import img_as_float
import matplotlib.pyplot as plt
import time
import scipy.io
from scipy.sparse import spdiags
import glob
from scipy.sparse import spdiags
import pandas as pd
import scipy.signal
import skvideo.io
from scipy.signal import butter
import scipy
import os

def _crop_face_region(sample_image):
    cv2.namedWindow("face", cv2.WINDOW_NORMAL)
    sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)
    r = cv2.selectROI("face", sample_image) # Press enter after selecting box
    print('Coordiantes: ', r)
    imCrop = sample_image[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    cv2.imshow("Image", imCrop)
    cv2.waitKey(0) # Press enter again to close both windows
    cv2.destroyWindow("face")
    cv2.destroyWindow("Image")
    return r


all_frame_path = sorted(glob.glob(".\cropping_frames\*.npy"))
# Parameters
#  Save a middle frame. (60th second)
for frame_path in all_frame_path:
    print('Frame path: ', frame_path)
    frame_basename = os.path.basename(frame_path).split('.')[0]
    selected_frame = np.load(frame_path)
    face_coordinates = _crop_face_region(selected_frame)
    palm_coordinates = _crop_face_region(selected_frame)
    coor_dict = {'face':face_coordinates, 'palm': palm_coordinates}
    np.save(f'./cropping_coor/{frame_basename}.npy', coor_dict)
