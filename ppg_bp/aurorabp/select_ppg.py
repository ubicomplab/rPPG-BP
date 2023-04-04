import os
import argparse
import numpy as np
import scipy
import scipy.signal
import pandas as pd
from collections import defaultdict
from matplotlib import pyplot as plt
# mat_path = "/gscratch/ubicomp/cm74/clinical_data/ProcessedDataNoVideo/P036a.mat"
# mat_data = scipy.io.loadmat(mat_path)
# print(mat_data)

def normalize_min_max(data):

    return (data - np.min(data)) / (np.max(data) - np.min(data))

def filter_ppg_signal(ppg_signal, LPF, HPF, FS):

    NyquistF = 1 / 2 * FS
    [b_pulse, a_pulse] = scipy.signal.butter(3, [LPF / NyquistF, HPF / NyquistF], btype='bandpass')
    ppg_filt = scipy.signal.filtfilt(b_pulse, a_pulse, np.double(ppg_signal))
    
    return ppg_filt

def concat_chunks(chunks):
    chunk = np.zeros((512))
    i = 0
    for ppg_chunk in chunks:
        chunk[i : i + len(ppg_chunk)] = np.squeeze(ppg_chunk)
        i = i + len(ppg_chunk)
    return chunk, i

def return_chunk_pos(chunks, pos):

    concat = chunks[0]
    map_pos_start = 0
    map_pos_end = 0

    for i in range(1, len(chunks)):
        if i == pos[0]:
            map_pos_start = len(concat)
        concat = np.concatenate((concat, chunks[i]), axis=None)
        if i == pos[1]:
            map_pos_end = len(concat)
    return concat, [map_pos_start, map_pos_end]

def padding_chunk(chunk):
    new_chunk = np.zeros((100))
    new_chunk[:len(chunk)] = chunk
    new_chunk[len(chunk):] = np.mean(chunk)
    return new_chunk

def ppg_template_from_segments(ppg_segments, method = 'mean'):
    """Create a reference template from the segments
    
    Parameters
    ----------
    ppg_segments : array
        Segmented ppg beats with equal length
    method : string
        Method to calculate the ppg template
    
    Returns
    -------
    ppg_template : list
        Calculated ppg beat from the segments 
    """
    
     # include other methods in the future
    if method == 'mean':
        ppg_template = np.nanmean(ppg_segments, axis = 0)
    return ppg_template

def ppg_calculate_template_correlation(ppg_segments,reference_signal):
    """Calculate the normalized cross correlation of the reference signal and per ppg beat
    then calculate the signal quality index. The overall SQI of the input ppg segment is 
    obtained by getting the mean of all the SQIs.
    
    Parameters
    ----------
    ppg_segments : array
        Array of the ppg beats segmented
    reference_signal : list
        The template signal 
    
    Returns
    -------
    sqi_vals_continuous : list
        Calculated SQI for each segment
    sqi_vals_categorical : list
        Calculated SQI for each segment
    sqi_mean : float
        Average SQI value
    """
    # Check if ppg_segments length = reference_signal
    if len(reference_signal) != max([len(i) for i in ppg_segments]):
        return ValueError("Reference signal doesnt have the same dimension with the ppg segments!")
    sqi_vals_continuous = []
    # sqi_vals_categorical = []
  
    for segment in ppg_segments:
        # segment = ppg_segments[i]
        reference_norm = (reference_signal - np.mean(reference_signal)) / (np.std(reference_signal) * len(reference_signal))
        segment_norm = (segment - np.nanmean(segment)) / (np.nanstd(segment))
        segment_norm = segment_norm[~np.isnan(segment_norm)]
        sqi_calc = np.correlate(reference_norm, segment_norm)[0]
        sqi_vals_continuous.append(sqi_calc)
        # sqi_cat = "Acceptable" if sqi_calc> template_sqi_threshold else "Not Acceptable"
        # sqi_vals_categorical.append(sqi_cat)
 
    sqi_mean = np.nanmean(sqi_vals_continuous)
    # sqi_category = "Acceptable" if sqi_mean > template_sqi_threshold else "Not Acceptable"
    return sqi_mean,sqi_vals_continuous

if __name__ == "__main__":
    # mat_dir = "/gscratch/ubicomp/cm74/bp/bp_preprocess/ppg_chunks_mat_face_230201"
    parser = argparse.ArgumentParser(description="settings")
    parser.add_argument("-id", "--processed_data", type=str, default="./measurements_oscillometric_ambulatory_preprocessed", help="input processed data folder")
    parser.add_argument("-od", "--selected_data", type=str, default="./measurements_oscillometric_ambulatory_seleceted", help="output selected data folder")
    opt = parser.parse_args()
    ori_path = opt.processed_data
    result_path = opt.selected_data
    pids = os.listdir(ori_path)
    LPF = 0.7
    HPF = 2
    FS = 60
    threshold = 0.9
    chunk_amount = 5
    for pid in pids:
        print(pid)
        measurements = os.listdir(os.path.join(ori_path, pid))
        if not os.path.exists(os.path.join(result_path, pid)):
            os.makedirs(os.path.join(result_path, pid))
        for ms in measurements:
            temp = list()
            temp_chunks = list()
            interval = list()
            select_chunks = list()
            mat_data = np.load(os.path.join(ori_path, pid, ms), allow_pickle=True)
            mat_data = scipy.signal.resample(mat_data, int(len(mat_data) / (500 / FS)))
            signal_2 = filter_ppg_signal(mat_data.copy(), LPF, HPF, FS)
            signal_2 = normalize_min_max(signal_2)
            signal = normalize_min_max(mat_data)
            peaks, _ = scipy.signal.find_peaks(1 - signal_2, height=0.65, distance=20)
            # print(peaks)
            for i in range(len(peaks) - 1):
                temp.append(signal[peaks[i] : peaks[i + 1]])
            
            chunks = np.array(temp)
            filtered_chunk = list()
            for chunk in chunks:
                if len(chunk) > 100:
                    continue
                filtered_chunk.append(chunk)
                pad_chunk = padding_chunk(chunk)
                temp_chunks.append(pad_chunk)
            if len(temp_chunks) < 1:
                print(ms)
                continue
            ppg_template = ppg_template_from_segments(temp_chunks)
            sqi_mean, score = ppg_calculate_template_correlation(temp_chunks, ppg_template)
            for j in range(len(score) - chunk_amount):
                cur_score = np.mean(score[j : j + chunk_amount])
                if cur_score > threshold:
                    if len(interval) < 1:
                        interval.append([j, j + chunk_amount])
                    elif j <= interval[-1][1]:
                        interval[-1][1] = j + 5
                    else:
                        interval.append([j, j + chunk_amount])
            
            if len(interval) < 1:
                print(pid, ms)
                continue
            
            for pos in interval:
                select_chunks.append(filtered_chunk[pos[0] : pos[1]])
            
            np.save(os.path.join(result_path, pid, ms[:-4].replace(".", "_") + ".npy"), np.array(select_chunks))
                        # select_chunks.append(score[j : j + chunk_amount])
            
            # all, pos = return_chunk_pos(filtered_chunk, interval[0])
            # plt.figure(figsize=(60, 10), dpi=60)
            # plt.plot(all)
            # for pos in interval:
            #     all, pos = return_chunk_pos(filtered_chunk, pos)
            #     plt.axvspan(pos[0], pos[1], alpha=0.3)
            # plt.savefig(os.path.join(result_path, pid, ms + ".png"))
            