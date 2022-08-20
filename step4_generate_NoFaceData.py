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


def extract_green_channel(videodata):
    mean_avg_video = np.average(videodata, axis=(1, 2))
    green_ppg_video = mean_avg_video[:, 1]
    green_ppg_video = detrend(green_ppg_video, 100)
    green_ppg_video = (green_ppg_video - np.mean(green_ppg_video)) / \
                        np.std(green_ppg_video)
    return green_ppg_video

def detrend(signal, Lambda):
    signal_length = signal.shape[0]

    # observation matrix
    H = np.identity(signal_length)

    # second-order difference matrix

    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index, (signal_length - 2), signal_length).toarray()
    filtered_signal = np.dot((H - np.linalg.inv(H + (Lambda ** 2) * np.dot(D.T, D))), signal)
    return filtered_signal

onedrive_path = r'D:\OneDrive - UW\rPPG Clinical Study\UW Medicine Data'
all_mat_path = sorted(glob.glob(onedrive_path + "\ProcessedData\*.mat"))
FS = 60
LAST_DURATION = 125 # last 125 seconds
RESIZED_DIM = 72
CUTOFF = 5


for mat_path in all_mat_path:
    mat_basename = os.path.basename(mat_path).split('.')[0]
    print('mat path: ', mat_basename)
    mdic = scipy.io.loadmat(mat_path)
    videodata_face = mdic['videodata_face']
    videodata_palm = mdic['videodata_palm']
    gt_ppg = mdic['ppg'][0]
    gt_ekg = mdic['ekg'][0]
    gt_resp = mdic['resp'][0]
    green_ppg_face = extract_green_channel(videodata_face)
    green_ppg_palm = extract_green_channel(videodata_palm)
    new_mdic = {"ppg_face": green_ppg_face,
                "ppg_palm": green_ppg_palm,
                'ppg': gt_ppg, 'ekg': gt_ekg, 'resp': gt_resp}
    scipy.io.savemat(f"{onedrive_path}\ProcessedDataNoVideo\{mat_basename}.mat", new_mdic)
