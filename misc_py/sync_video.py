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


mutable_object = {}

def preprocess_raw_video(video_path, face_coordinates, palm_coordinates, resized_dim=72):
    #########################################################################
    # set up
    t = []
    i = 0
    vidObj = cv2.VideoCapture(video_path);
    totalFrames = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT)) # get total frame size
    frames_face = np.zeros((totalFrames, resized_dim, resized_dim, 3), dtype = np.float32)
    frames_palm = np.zeros((totalFrames, resized_dim, resized_dim, 3), dtype = np.float32)
    height = vidObj.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = vidObj.get(cv2.CAP_PROP_FRAME_WIDTH)
    success, img = vidObj.read()
    resized_dims = img.shape
    print("Orignal Height", height)
    print("Original width", width)
    #########################################################################
    # Crop each frame size into resized_dim x resized_dim
    while success:
        t.append(vidObj.get(cv2.CAP_PROP_POS_MSEC))# current timestamp in milisecond
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
        frames_face[i, :, :, :] = img_face
        frames_palm[i, :, :, :] = img_palm
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

def load_ground_truth(txt_path):
    # read text file into pandas DataFrame
    header_list = ["time", "ekg", "ppg", 'resp', 'light']
    df = pd.read_csv(txt_path, sep=",", names=header_list)
    idx_light = df.index[df['light'] == 'flash'].tolist()[0] # Find Light on Index
    ppg = np.array(df['ppg'])[idx_light:]
    ppg_sampled = scipy.signal.resample(ppg, ppg.shape[0]//2048*60) # resample PPG from 2048hz to 30hz
    ppg_sampled =  (ppg_sampled - np.mean(ppg_sampled)) / np.std(ppg_sampled)
    return idx_light, ppg_sampled


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


def onclick_select(event, all_ax):
    for idx, ax in enumerate(all_ax):
        if event.inaxes == ax:
            mutable_object['key'] = idx
            plt.close('all')

def _grid_plot(data, data_idx):
    num_width = 15
    num_height = len(data) // num_width
    fig = plt.figure(figsize=(3*num_width, 6*num_height), dpi=80)
    wm = plt.get_current_fig_manager()
    wm.window.state('zoomed')
    all_ax = []
    for i in range(num_height):
        for j in range(num_width):
            idx = i*num_width + j
            if idx >= len(data):
                break
            ax = plt.subplot2grid((num_height, num_width), (i,j), fig=fig)
            img = img_as_float(data[idx])
            img = cv2.resize(img, (int(img.shape[1] * 0.1), int(img.shape[0]*0.1)),
                                    interpolation = cv2.INTER_AREA)
            ax.imshow(img)
            ax.set_title(str(data_idx[idx]))
            ax.get_yaxis().set_visible(False)
            ax.get_xaxis().set_visible(False)
            all_ax.append(ax)
    fig.canvas.mpl_connect("button_press_event", lambda event: onclick_select(event, all_ax))
    plt.show()
    onclick_idx = mutable_object['key']
    selected_frame_idx = data_idx[onclick_idx]
    selected_frame = data[-1] # just use the last frame as hand might be still in the selected_frame_idx frame
    return selected_frame, selected_frame_idx

def find_flash_frame(videoFilePath, start_second, end_second, fs=60):
    i = 0
    vidObj = cv2.VideoCapture(videoFilePath);
    success, img = vidObj.read()
    all_imgs = []
    all_imgs_idx = []
    START_SECOND = start_second
    END_SECOND = end_second
    FS = fs
    TOTAL_FRAMES = (END_SECOND - START_SECOND) * FS
    while success:
    #     t.append(vidObj.get(cv2.CAP_PROP_POS_MSEC))# current timestamp in milisecond
        if i < START_SECOND * FS:
            success, img = vidObj.read() # read the next one
            i = i + 1
            continue
        elif i > (END_SECOND * FS - 1):
            break
        else:
            vidLxL = img_as_float(img)
            vidLxL = cv2.cvtColor(vidLxL.astype('float32'), cv2.COLOR_BGR2RGB)
            vidLxL[vidLxL > 1] = 1
            vidLxL[vidLxL < (1/255)] = 1/255
            all_imgs.append(vidLxL)
            all_imgs_idx.append(i)
            success, img = vidObj.read() # read the next one
            i = i + 1
    return _grid_plot(all_imgs, all_imgs_idx)


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

# subject 8 has afib.
# subject 34 has afib
# subject 35 has pvc and pac
# subject 46 has afib
# subject 48 has afib
# subject 51 -

onedrive_path = r'D:\OneDrive - UW\rPPG Clinical Study\UW Medicine Data\03_29_2022'
all_video_path = sorted(glob.glob(onedrive_path + "\Videos\*.mp4"))
all_gt_path = sorted(glob.glob(onedrive_path + "\*.txt"))


# Parameters
FS = 60
START_TIME = 7
END_TIME = 9
RESIZED_DIM = 256
SIGNAL_LEN_CUTOFF = 200
subject = 'P007a'
video_path =  onedrive_path + f'\Videos\{subject}.mp4'
gt_path = onedrive_path + f"\{subject}.txt"
# # Load ground truth and find the annotated idx
light_idx, sampled_ppg = load_ground_truth(gt_path)
# Load Video and crop face and palm
selected_frame, selected_frame_idx = find_flash_frame(video_path, START_TIME, END_TIME, fs=FS)
face_coordinates = _crop_face_region(selected_frame)
palm_coordinates = _crop_face_region(selected_frame)
print('video_path: ', video_path)
print('selected_frame: ', selected_frame_idx)
# Process the video with cropped coordinates
videodata_face, videodata_palm = preprocess_raw_video(video_path,
                                                     face_coordinates,
                                                     palm_coordinates,
                                                     resized_dim=RESIZED_DIM)
# Use the flash index to trim the videos
videodata_face = videodata_face[selected_frame_idx:]
videodata_palm = videodata_palm[selected_frame_idx:]
# Extract green PPG signal from the videos
green_ppg_face = extract_green_channel(videodata_face)
green_ppg_palm = extract_green_channel(videodata_palm)

# Time cutoff
green_ppg_face = green_ppg_face[SIGNAL_LEN_CUTOFF:-300] # 300 is a buffer for the end
green_ppg_palm = green_ppg_palm[SIGNAL_LEN_CUTOFF:-300]
sampled_ppg = sampled_ppg[SIGNAL_LEN_CUTOFF:-300]
face_hr, green_ppg_face = calculate_hr_fft(green_ppg_face)
palm_hr, green_ppg_palm = calculate_hr_fft(green_ppg_palm)
gt_hr, sampled_ppg = calculate_hr_fft(sampled_ppg)
# Plot
plt.figure(figsize=(12, 6), dpi=100)
plt.plot(green_ppg_face[:3000]) # Just plot first 30 seconds signal
plt.plot(green_ppg_palm[:3000])
plt.plot(sampled_ppg[:3000])
plt.legend(['Face PPG', 'Palm PPG', 'GT'])
plt.title(f"Subject: {subject}; Face HR: {face_hr}; Palm HR: {palm_hr}; GT HR: {gt_hr}")
plt.show()
plt.savefig(f'./{subject}.png')
