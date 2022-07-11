import numpy as np
import cv2
from skimage.util import img_as_float
import matplotlib.pyplot as plt
import time
import scipy.io
from scipy.sparse import spdiags
import glob

onedrive_path = r'D:\OneDrive - UW\rPPG Clinical Study\UW Medicine Data\03_28_2022'
all_video_path = glob.glob(onedrive_path + "\Videos\*.mp4")

mutable_object = {}

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
    print('onclick_idx: ', onclick_idx)
    selected_frame_idx = data_idx[onclick_idx]
    selected_frame = data[onclick_idx]
    print('selected_frame: ', selected_frame_idx)
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

selected_frame, selected_frame_idx = find_flash_frame(all_video_path[0], 0, 2, fs=60)
face_coordinates = _crop_face_region(selected_frame)
