import os
import argparse
import numpy as np
import pandas as pd


def concat_chunks(chunks):
    chunk = np.zeros((512))
    i = 0
    for ppg_chunk in chunks:
        chunk[i : i + len(ppg_chunk)] = np.squeeze(ppg_chunk)
        i = i + len(ppg_chunk)
    return chunk, i

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
    parser = argparse.ArgumentParser(description="settings")
    parser.add_argument("-m", "--mat_path", type=str, default="./bp_preprocess/ppg_chunks_mat_finger_v2_sqi", help="ppg preprocessed mat folder path")
    parser.add_argument("-s", "--save_path", type=str, default="./bp_preprocess/ppg_chunks_mat_finger_v2_sqi_t_8_top10_0511", help="save path of ppg filtered by sqi")
    parser.add_argument("-d", "--df_path", type=str, default="./BPData_230223_demograph.csv", help="patient info csv")
    opt = parser.parse_args()
    
    mat_dir = opt.mat_path
    new_mat_dir = opt.save_path
    df = pd.read_csv(opt.df_path)

    chunk_amount = 5
    LPF = 0.7
    HPF = 16
    FS = 60
    threshold = 0.8
    empty = dict()
    
    for i, row in df.iterrows():

        chunks = list()
        mat_data = np.load(os.path.join(mat_dir, row["session"] + ".npy"), allow_pickle=True)
        template_list = list()
        for chunk in mat_data:
            pad_chunk = padding_chunk(chunk)
            template_list.append(pad_chunk)

        ppg_template = ppg_template_from_segments(template_list)
        sqi_mean, score = ppg_calculate_template_correlation(template_list, ppg_template)
        interval = list()
        interval_score = list()
        for j in range(len(score) - 5):
            quality = 1
            score_sum = 0
            for k in range(5):
                score_sum += score[j + k]
                if score[j + k] < threshold:
                    quality = 0
            if quality == 1:
                interval.append([[j, j + 5], score_sum])
        print("=================================")
        if len(interval) < 1:
            print("discard sessions: ", row["session"])
            continue
        interval = sorted(interval, key=lambda x: x[1], reverse=True)
        top_k = 10
        if len(interval) < top_k:
            for pos in interval:
                chunks.append(mat_data[pos[0][0] : pos[0][1]])
        else:
            for pos in interval[:top_k]:
                chunks.append(mat_data[pos[0][0] : pos[0][1]])

        np.save(os.path.join(new_mat_dir, row["session"] + ".npy"), np.array(chunks))
