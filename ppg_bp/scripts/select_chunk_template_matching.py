import os
import numpy as np
import scipy
import scipy.io
import pandas as pd
from matplotlib import pyplot as plt
from preprocess import normalize_min_max, filter_ppg_signal
from ppg_assess import ppg_signal_quality
# mat_path = "/gscratch/ubicomp/cm74/clinical_data/ProcessedDataNoVideo/P036a.mat"
# mat_data = scipy.io.loadmat(mat_path)
# print(mat_data)


# def calculate_snr(waveform):
#     """Calculate the signal-to-noise ratio (SNR) of a waveform.
    
#     Args:
#         waveform (numpy.ndarray): Input waveform signal.
    
#     Returns:
#         float: Signal-to-noise ratio (SNR) in decibels (dB).
#     """
#     # Calculate the root-mean-square (RMS) of the signal
#     signal_rms = np.sqrt(np.mean(waveform ** 2))
    
#     # Calculate the noise power
#     noise_power = np.mean(waveform ** 2) - signal_rms ** 2
    
#     # Calculate the SNR in decibels (dB)
#     snr = 10 * np.log10(signal_rms ** 2 / noise_power)
    
#     return snr

# def calculate_snr(ppg_signal):
#     """Calculates the signal-to-noise ratio (SNR) of a PPG signal.
    
#     Args:
#         ppg_signal (np.ndarray): PPG signal array.
        
#     Returns:
#         float: Signal-to-noise ratio (SNR) in dB.
#     """
#     # Calculate the peak amplitude of the signal
#     signal_average = np.mean(ppg_signal)
    
#     # Calculate the standard deviation of the noise
#     signal_noise = ppg_signal - np.mean(ppg_signal)
#     noise_std = np.std(signal_noise)
    
#     # Calculate the SNR in dB
#     snr = 20 * np.log10(signal_average / noise_std)
    
#     return snr


def concat_chunks(chunks):
    chunk = np.zeros((512))
    i = 0
    for ppg_chunk in chunks:
        chunk[i : i + len(ppg_chunk)] = np.squeeze(ppg_chunk)
        i = i + len(ppg_chunk)
    return chunk, i

def concat_chunks_longer(chunks):
    i = 0
    for ppg_chunk in chunks:
        i = i + len(ppg_chunk)
    chunk = np.zeros((i))
    i = 0
    for ppg_chunk in chunks:
        chunk[i : i + len(ppg_chunk)] = np.squeeze(ppg_chunk)
        i = i + len(ppg_chunk)
    return chunk

def padding_chunk(chunk):
    new_chunk = np.zeros((100))
    new_chunk[:len(chunk)] = chunk
    new_chunk[len(chunk):] = np.mean(chunk)
    return new_chunk

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
    mat_dir = "/gscratch/ubicomp/cm74/bp/bp_preprocess/ppg_chunks_mat_face_v2_sqi_ex_r2_230504"
    new_mat_dir = "/gscratch/ubicomp/cm74/bp/bp_preprocess/ppg_chunks_mat_face_v2_sqi_selected_t_85_0504"
    figure_savepath = "/gscratch/ubicomp/cm74/bp/bp_preprocess/test_figure_plots_v2_face_patch_0_3"
    # new_mat_dir = "/gscratch/ubicomp/cm74/bp/bp_preprocess/ppg_chunks_mat_face_manual_230201"
    chunk_amount = 5
    LPF = 0.7
    HPF = 16
    FS = 60
    threshold = 0.85
    empty = dict()
    df = pd.read_csv("/gscratch/ubicomp/cm74/bp/bp_preprocess/ppg_manual_selection_0225.csv")
    # dirty_data = ["P016a", "P043b", "P058a", "P091a", "P098a", "P114b", "P116a", "P136b",
    #     "P014b", "P040b", "P045a", "P113a", "P118a", "P124b", "P126b", "P136a", "P143b"]
    # dirty_data = ["P016a", "P027a", "P039a", "P043b", "P058a", "P091a", "P095b", "P098a", "P110b", "P114a", "P116a",
    #     "P014b", "P040b", "P045a", "P113a", "P118a", "P126b", "P136a", "P143b"]
    dirty_data = ["P016a", "P043b", "P074b", "P098a", "P114b", "P116a", "P136b"]
    dirty_data = ["P027a", "P054b", "P058b", "P078b"]
    ### for face patch 0_3_8_11
    dirty_data = ["P027a", "P051a", "P054b", "P063b", "P110b", "P116a"]
    dirty_data = ["P027a", "P051a", "P098a", "P116a"]
    ### for face patch 0_3
    # dirty_data = ["P016a", "P043b", "P074b", "P098a", "P114b", "P116a", "P136b"]
    ### for palm
    # dirty_data = ["P014a", "P036a", "P036b", "P042a", "P057a", "P058b", "P079a", "P080b", "P093a", "P096b", "P116a", "P122a", "P132a", "P137a"]
    ### for finger
    # dirty_data = ["P007a", "P007b", "P058a", "P075a", "P090b", "P102b", "P110a"]
    # dirty_data = ["P070b", "P090b", "P110a"]
    count = 0

    drop_rate_list = list()
    snr = list()
    for i, row in df.iterrows():

        if row["session"] in dirty_data:
            continue

        mat_data = np.load(os.path.join(mat_dir, row["session"] + ".npy"), allow_pickle=True)
        chunks = list()
        template_list = list()
        for chunk in mat_data:
            pad_chunk = padding_chunk(chunk)
            template_list.append(pad_chunk)

        ppg_template = ppg_template_from_segments(template_list)
        sqi_mean, score = ppg_calculate_template_correlation(template_list, ppg_template)
        interval = list()
        for j in range(len(score) - 5):
            quality = 1
            for k in range(5):
                if score[j + k] < threshold:
                    quality = 0
            if quality == 1:
                if len(interval) < 1:
                    interval.append([j, j + 5])
                elif j <= interval[-1][1]:
                    interval[-1][1] = j + 5
                else:
                    interval.append([j, j + 5])
        print("=================================")
        
        # for pos in interval:
        #     count_qualified += pos[1] - pos[0] + 1
        
        # drop_rate = (count_total - count_qualified) / count_total
        # print(drop_rate)
        # drop_rate_list.append(drop_rate)
        
        if len(interval) < 1:
            count += 1
            print(row["session"])
            continue
        for pos in interval:
            chunks.append(mat_data[pos[0] : pos[1]])

        np.save(os.path.join(new_mat_dir, row["session"] + ".npy"), np.array(chunks))
        
        # for chunk in chunks:
        #     continuous_chunk = concat_chunks_longer(chunk)
        #     print(calculate_snr(continuous_chunk))
        #     snr.append(calculate_snr(continuous_chunk))
    # drop_rate = (count_total - count_qualified) / count_total
    # print(drop_rate)
    # print(np.mean(snr))
        # fig, ax = plt.subplots(2, figsize=(60, 10), dpi=80)

        # for m in range(len(interval)):
        #     all, select_map_pos = return_chunk_pos(mat_data, interval[m])
        #     if m == 0:
        #         ax[0].plot(all)
        #     ax[0].axvspan(select_map_pos[0], select_map_pos[1], color="blue", alpha=0.3)

        #     # all2, select_map_pos2 = return_chunk_pos(mat_data[1], interval[m])
        #     # if m == 0:
        #     #     ax[1].plot(all2)
        #     # ax[1].axvspan(select_map_pos2[0], select_map_pos2[1], color="blue", alpha=0.3)

        # plt.savefig(os.path.join(figure_savepath, row["session"] + ".png"))


    print(count)
        # ppg_chunks, index = concat_chunks(mat_data)
        # score = ppg_signal_quality(ppg_chunks, FS, method="template_matching")
        # interval = list()
        # for j in range(len(score[2]) - 5):
        #     quality = 1
        #     for k in range(5):
        #         if score[2][j + k] < 0.9:
        #             quality = 0
        #     if quality == 1:
        #         if len(interval) < 1:
        #             interval.append([j, j + 5])
        #         elif j <= interval[-1][1]:
        #             interval[-1][1] = j + 5
        #         else:
        #             interval.append([j, j + 5])
        # print("=================================")
        # if len(interval) < 1:
        #     count += 1
        #     print(row["session"])
        #     continue
        # print(interval)
        # print(len(mat_data))
        # for pos in interval:
        #     chunks.append(mat_data[pos[0] : pos[1]])
        # np.save(os.path.join(new_mat_dir, row["session"] + ".npy"), np.array(chunks))
        # print(interval)
    
    # print(count)
        # if score[1] != "Acceptable":
        #     print("=================================")
        #     print(row["session"])
        # intervals = row["pulse_seg_id"].split(";")
        # count = 0
        
        # for interval in intervals:
        #     se = interval.split(",")
        #     # print(row["session"])
        #     start = int(se[0][1:])
        #     end = int(se[1][:-1])
        #     for j in range(start, end):
        #         if score[3][j] != "Acceptable":
        #             print("=================================")
        #             print(row["session"])
        #             print(start, end, j)
        #     if end > len(mat_data):
        #         # print(row["session"])
        #         while end > len(mat_data) - 1 and end - start > 5:
        #             end = end - 1
        #         if end > len(mat_data) - 1 or end - start < 5:
        #             count += 1
        #             continue
        #     chunks.append(mat_data[start : end])
        
        # if count == len(intervals):
        #     print(row["session"])

        # if len(chunks) < 1:
        #     print(row["session"])
        # np.save(os.path.join(new_mat_dir, row["session"] + ".npy"), np.array(chunks))
    # for mat_path in os.listdir(mat_dir):
    #     pid = mat_path.split("/")[-1].split(".")[0]
    #     df[df["session"]==pid]
    #     score = dict()
    #     mat_data = np.load(os.path.join(mat_dir, mat_path), allow_pickle=True)
        
    #     # print(mat_data.shape)
    #     # chunk_pos = random.randint(0, len(mat_data) - chunk_amount - 1)
    #     # chunks = mat_data[chunk_pos : chunk_pos + chunk_amount]

    #     for i in range(len(mat_data) - chunk_amount):
    #         skew1 = scipy.stats.skew(mat_data[i])
    #         skew2 = scipy.stats.skew(mat_data[i + 1])
    #         skew3 = scipy.stats.skew(mat_data[i + 2])
    #         skew4 = scipy.stats.skew(mat_data[i + 3])
    #         skew5 = scipy.stats.skew(mat_data[i + 4])
    #         average = (skew1 + skew2 + skew3 + skew4 + skew5) / chunk_amount
    #         score[i] = average
        
    #     # chunk, pos = concat_chunks(chunks)
    #     # # chunk = 1 - chunk
    #     # chunk[pos:] = np.mean(chunk[:pos])
    #     # chunk_filt = filter_ppg_signal(chunk, LPF, HPF, FS)
    #     # chunk_filt_norm = normalize_min_max(chunk_filt)
    # # plt.figure()
    # # plt.plot(chunk_filt_norm)
    # # plt.savefig("finger_ppg_input.png")
    #     # print(score)
    #     temp_score = sorted(score.items(), key=lambda item: item[1], reverse=True)
    #     top3 = temp_score[:3]
        
    #     top3_chunk_list = list()
    #     for pos in top3:
    #         new_chunks = mat_data[int(pos[0]) : int(pos[0]) + chunk_amount]
    #         if len(new_chunks) != 5:
    #             print(mat_path)
    #             # print(len(new_chunks))
    #             empty[mat_path] = len(new_chunks)
    #             continue
    #         top3_chunk_list.append(new_chunks)
    #     # print(np.array(top3_chunk_list).shape)

    #     # for i in range(len(mat_data)):
    #     #     skew1 = scipy.stats.skew(mat_data[i])
    #     #     score[i] = skew1

    #     # temp_score = sorted(score.items(), key=lambda item: abs(item[1]))
    #     # top3 = temp_score[:3]

    #     # top3_chunk_list = list()
    #     # for pos in top3:
    #     #     new_chunks = mat_data[int(pos[0])]
    #     #     # new_chunks = mat_data[int(pos[0]) : int(pos[0]) + chunk_amount]
    #     #     # if len(new_chunks) != 5:
    #     #     #     # print(mat_path)
    #     #     #     # print(len(new_chunks))
    #     #     #     empty[mat_path] = len(new_chunks)
    #     #     #     continue
    #     #     top3_chunk_list.append(new_chunks)

    #     if len(top3_chunk_list) < 1:
    #         print(mat_path)
    #         continue
    #     np.save(os.path.join(new_mat_dir, mat_path), np.array(top3_chunk_list))
        # for key, value in score.items():
        #     if value
    
    
    # print(len(empty))
    # print(empty)
        