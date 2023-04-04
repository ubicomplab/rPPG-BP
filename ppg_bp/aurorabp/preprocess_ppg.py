import os
import argparse
import numpy as np
import pandas as pd
import scipy.signal
import matplotlib.pyplot as plt


def filter_ppg_signal(ppg_signal, LPF, HPF, FS):

    NyquistF = 1 / 2 * FS
    [b_pulse, a_pulse] = scipy.signal.butter(3, [LPF / NyquistF, HPF / NyquistF], btype='bandpass')
    ppg_filt = scipy.signal.filtfilt(b_pulse, a_pulse, np.double(ppg_signal))
    
    return ppg_filt

def remove_artifact(ppg_signal):

    ppg_signal_mean = np.mean(ppg_signal)
    ppg_signal_std = np.std(ppg_signal)
    ### TODO: Xin replaces the value with the mean; Danial replaces the value with 0
    # ppg_signal = np.where(ppg_signal < ppg_signal_mean + 3 * ppg_signal_std, ppg_signal, 0)
    # ppg_signal = np.where(ppg_signal > ppg_signal_mean - 3 * ppg_signal_std, ppg_signal, 0)
    ppg_signal = np.where(ppg_signal < ppg_signal_mean + 3 * ppg_signal_std, ppg_signal, ppg_signal_mean)
    ppg_signal = np.where(ppg_signal > ppg_signal_mean - 3 * ppg_signal_std, ppg_signal, ppg_signal_mean)

    return ppg_signal


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="settings")
    parser.add_argument("-ic", "--csv_path", type=str, default="./auro_data_ambulatory.csv", help="input csv file")
    parser.add_argument("-id", "--raw_data", type=str, default="/gscratch/ubicomp/cm74/MS_aurorabp/measurements_oscillometric", help="raw signal data folder")
    parser.add_argument("-od", "--processed_data", type=str, default="./measurements_oscillometric_ambulatory_preprocessed", help="raw signal data folder")
    opt = parser.parse_args()

    ori_path = opt.raw_Data
    result_path = opt.processed_data
    filtered_df_path = opt.csv_path
    
    filtered_df = pd.read_csv(filtered_df_path)
    pids = os.listdir(ori_path)
    LPF = 0.7
    HPF = 16
    FS = 500
    for pid in pids:
        measurements = os.listdir(os.path.join(ori_path, pid))
        temp_filtered_df = filtered_df[filtered_df["pid"] == pid]
        for ms in measurements:
            session_split = ms[:-4].split(".")[-1]
            session = session_split.replace("_", " ")
            if session not in temp_filtered_df["measurement"].values:
                print(ms)
                continue
            df = pd.read_csv(os.path.join(ori_path, pid, ms), sep = '\t')
            signal = np.array(df["optical"].values)
            signal = remove_artifact(signal)
            signal = filter_ppg_signal(signal, LPF, HPF, FS)
            normalized_signal = (signal - np.mean(signal)) / np.std(signal)
            if not os.path.exists(os.path.join(result_path, pid)):
                os.makedirs(os.path.join(result_path, pid))
            np.save(os.path.join(result_path, pid, ms), signal)
            
            # plt.figure(figsize=(60, 10), dpi=30)
            # plt.plot(normalized_signal)
            # plt.savefig(os.path.join(result_path, pid, ms + ".png"))

    
    # bp_df = pd.read_csv(tsv_path, sep = '\t')
    # pid = list()
    # measurement = list()
    # sbp = list()
    # dbp = list()
    # baseline_sbp = list()
    # baseline_dbp = list()
    # for i, row in bp_df.iterrows():
    #     if np.isnan(row["sbp"]):