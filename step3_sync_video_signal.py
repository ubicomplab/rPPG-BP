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

def preprocess_raw_video(video_path, face_coordinates, palm_coordinates,
                        resized_dim=72, duration=125, fs=60):
    #########################################################################
    # set up
    i = 0
    vidObj = cv2.VideoCapture(video_path);
    totalFrames = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT)) # get total frame size
    totalFrames_cropped = duration * fs
    frames_face = np.zeros((totalFrames_cropped, resized_dim, resized_dim, 3), dtype = np.float32)
    frames_palm = np.zeros((totalFrames_cropped, resized_dim, resized_dim, 3), dtype = np.float32)
    height = vidObj.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = vidObj.get(cv2.CAP_PROP_FRAME_WIDTH)
    success, img = vidObj.read()
    resized_dims = img.shape
    #########################################################################
    # Crop each frame size into resized_dim x resized_dim
    while success:
        if i < (totalFrames - totalFrames_cropped): # skipp all the frames until target start frame
            success, img = vidObj.read() # read the next one
            i += 1
            continue
        else: # recording target frames
            img_face = img[int(face_coordinates[1]):int(face_coordinates[1] +
                                    face_coordinates[3]), int(face_coordinates[0]):int(
                                    face_coordinates[0] + face_coordinates[2])]
            img_palm = img[int(palm_coordinates[1]):int(palm_coordinates[1] +
                                    palm_coordinates[3]), int(palm_coordinates[0]):int(
                                    palm_coordinates[0] + palm_coordinates[2])]
            img_face = cv2.resize(img_as_float(img_face), (resized_dim, resized_dim),
                                    interpolation = cv2.INTER_AREA)
            img_palm = cv2.resize(img_as_float(img_palm), (resized_dim, resized_dim),
                                    interpolation = cv2.INTER_AREA)
            img_face = cv2.rotate(img_face, cv2.ROTATE_90_CLOCKWISE) # rotate 90 degree
            img_palm = cv2.rotate(img_palm, cv2.ROTATE_90_CLOCKWISE) # rotate 90 degree
            img_face = cv2.cvtColor(img_face.astype('float32'), cv2.COLOR_BGR2RGB)
            img_palm = cv2.cvtColor(img_palm.astype('float32'), cv2.COLOR_BGR2RGB)
            frames_face[i-totalFrames_cropped, :, :, :] = img_face
            frames_palm[i-totalFrames_cropped, :, :, :] = img_palm
            success, img = vidObj.read() # read the next one
            i = i + 1
    return frames_face, frames_palm


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

def resample_gt_signal(signal, source_fs, target_fs):
    target_len = signal.shape[0]//source_fs*target_fs
    signal = scipy.signal.resample(signal, target_len)
    return signal

def normalize_gt_signal(signal):
    return (signal - np.mean(signal)) / np.std(signal)


def load_ground_truth(txt_path, source_fs=2048, target_fs=60, duration=125):
    # read text file into pandas DataFrame
    header_list = ["time", "ekg", "ppg", 'resp']
    df = pd.read_csv(txt_path, sep=",", names=header_list)
    ppg = np.array(df['ppg'])[9:]
    ekg = np.array(df['ekg'])[9:]
    resp = np.array(df['resp'])[9:]
    ppg_processed = resample_gt_signal(ppg, source_fs, target_fs)
    ppg_processed = normalize_gt_signal(ppg_processed)[-target_fs*duration:]
    ekg_processed = resample_gt_signal(ekg, source_fs, target_fs)
    ekg_processed = normalize_gt_signal(ekg_processed)[-target_fs*duration:]
    resp_processed = resample_gt_signal(resp, source_fs, target_fs)
    resp_processed = normalize_gt_signal(resp_processed)[-target_fs*duration:]
    return ppg_processed, ekg_processed, resp_processed


def extract_green_channel(videodata):
    mean_avg_video = np.average(videodata, axis=(1, 2))
    green_ppg_video = mean_avg_video[:, 1]
    green_ppg_video = detrend(green_ppg_video, 100)
    green_ppg_video = (green_ppg_video - np.mean(green_ppg_video)) / \
                        np.std(green_ppg_video)
    return green_ppg_video

def filter_ppg(signal, low_band=0.75, high_band=2.5, fs=60):
    [b_pulse, a_pulse] = butter(1, [low_band / fs * 2, high_band / fs * 2], btype='bandpass')
    signal = scipy.signal.filtfilt(b_pulse, a_pulse, np.double(signal))
    return signal

def calculate_hr(signal, fs=60):
    filtered_signal = filter_ppg(signal)
    preds_peaks, _ = scipy.signal.find_peaks(filtered_signal)
    hr = 60 / (np.mean(np.diff(preds_peaks)) / fs)
    return hr, filtered_signal

def next_power_of_2(x):
    return 1 if x == 0 else 2**(x - 1).bit_length()

def calculate_hr_fft(signal, low_band=0.75, high_band=2.5, fs=60):
    filtered_signal = filter_ppg(signal)
    N = next_power_of_2(filtered_signal.shape[0])
    f_signal, pxx_signal = scipy.signal.periodogram(filtered_signal, fs=fs, nfft=N, detrend=False)
    fmask_signal = np.argwhere((f_signal >= low_band) & (f_signal <= high_band))
    signal_fft = np.take(f_signal, fmask_signal)
    hr_fft = np.take(signal_fft, np.argmax(np.take(pxx_signal, fmask_signal), 0))[0] * 60
    return hr_fft, filtered_signal


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

onedrive_path = r'D:\OneDrive - UW\rPPG Clinical Study\UW Medicine Data'
all_video_path = sorted(glob.glob(onedrive_path + "\*\Videos\*.mp4"))
all_gt_path = sorted(glob.glob(onedrive_path + "\ProCOmp Txt\*.txt"))
all_frame_path = sorted(glob.glob(".\cropping_frames\*.npy"))
FS = 60
LAST_DURATION = 125 # last 125 seconds
RESIZED_DIM = 72
CUTOFF = 5

for video_path in all_video_path:
    video_basename = os.path.basename(video_path).split('.')[0]
    if video_basename != 'P025a':
        continue
    print('Processing: ', video_basename)
    # Load Video and crop face and palm
    selected_frame_coor = np.load(f'./cropping_coor/{video_basename}.npy', allow_pickle='TRUE').item()
    face_coordinates = selected_frame_coor['face']
    palm_coordinates = selected_frame_coor['palm']

    # Process the video with cropped coordinates
    videodata_face, videodata_palm = preprocess_raw_video(video_path,
                                                         face_coordinates,
                                                         palm_coordinates,
                                                         resized_dim=RESIZED_DIM,
                                                         duration=LAST_DURATION)

    # load ground_truth ppg_sampled
    gt_ppg, gt_ekg, gt_resp = load_ground_truth(f"{onedrive_path}\ProCOmp Txt\{video_basename}.txt",
                                                target_fs=FS,
                                                duration=LAST_DURATION)
    # Time cutoff
    videodata_face = videodata_face[:-CUTOFF*FS]
    videodata_palm = videodata_palm[:-CUTOFF*FS]
    gt_ppg = gt_ppg[:-CUTOFF*FS]
    gt_ekg = gt_ekg[:-CUTOFF*FS]
    gt_resp = gt_resp[:-CUTOFF*FS]
    # print('before saving')
    # print('gt_ppg shape: ', gt_ppg.shape)
    # print('gt_ekg shape: ', gt_ekg.shape)
    # print('gt_resp shape: ', gt_resp.shape)

    # Save data into matfile
    mdic = {"videodata_face": videodata_face, "videodata_palm": videodata_palm,
            'ppg': gt_ppg, 'ekg': gt_ekg, 'resp': gt_resp}
    scipy.io.savemat(f"{onedrive_path}\ProcessedData\{video_basename}.mat", mdic)
    # #Debug
    # mdic = scipy.io.loadmat(f"{onedrive_path}\ProcessedData\{video_basename}.mat")
    # videodata_face = mdic['videodata_face']
    # videodata_palm = mdic['videodata_palm']
    # gt_ppg = mdic['ppg'][0]
    # gt_ekg = mdic['ekg'][0]
    # gt_resp = mdic['resp'][0]
    # print('videodata_face shape: ', videodata_face.shape)
    # print('videodata_palm shape: ', videodata_palm.shape)
    # print('gt_ppg shape: ', gt_ppg.shape)
    # print('gt_ekg shape: ', gt_ekg.shape)
    # print('gt_resp shape: ', gt_resp.shape)
    # Extract green PPG signal from the video
    green_ppg_face = extract_green_channel(videodata_face)
    green_ppg_palm = extract_green_channel(videodata_palm)

    # Calculate ground truth HR
    face_hr, green_ppg_face = calculate_hr_fft(green_ppg_face)
    palm_hr, green_ppg_palm = calculate_hr_fft(green_ppg_palm)
    gt_hr, gt_ppg = calculate_hr_fft(gt_ppg)
    gt_hr_ekg, gt_ekg = calculate_hr_fft(gt_ekg)

    # Plot
    plt.figure(figsize=(24, 10), dpi=100)
    plt.subplot(211)
    plt.plot(green_ppg_face[:3000]) # Just plot first 30 seconds signal
    plt.plot(green_ppg_palm[:3000])
    plt.legend(['Face PPG', 'Palm PPG'])
    plt.title(f"Subject: {video_basename}; Face HR: {face_hr}; Palm HR: {palm_hr}")
    plt.subplot(212)
    plt.plot(gt_ppg[:3000])
    plt.plot(gt_ekg[:3000])
    plt.legend(['GT-PPG', 'GT-EKG'])
    plt.title(f"Subject: {video_basename}; GT PPG HR: {gt_hr}; GT EKG HR: {gt_hr_ekg}")
    plt.savefig(f"{onedrive_path}\ProcessedDataResults\{video_basename}.png")
