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

def find_frame(videoFilePath, duration, fs=60):
    i = 0
    vidObj = cv2.VideoCapture(videoFilePath)
    totalFrames = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT))
    success, img = vidObj.read()
    selected_frame_idx = totalFrames - duration * fs
    FS = fs
    while success:
    #     t.append(vidObj.get(cv2.CAP_PROP_POS_MSEC))# current timestamp in milisecond
        if i < selected_frame_idx:
            success, img = vidObj.read() # read the next one
            i = i + 1
            continue
        else:
            vidLxL = img_as_float(img)
            vidLxL = cv2.cvtColor(vidLxL.astype('float32'), cv2.COLOR_BGR2RGB)
            return vidLxL


onedrive_path = r'D:\OneDrive - UW\rPPG Clinical Study\UW Medicine Data'
all_video_path = sorted(glob.glob(onedrive_path + "\*\Videos\*.mp4"))
# Parameters
FS = 60
DURATION = 120 # in seconds
#  Save a middle frame. (60th second)
for video_path in all_video_path:
    video_basename = os.path.basename(video_path).split('.')[0]
    print(video_basename)
    # Load Video and crop face and palm
    selected_frame = find_frame(video_path, DURATION, fs=FS)
    np.save(f'./cropping_frames/{video_basename}.npy', selected_frame)
