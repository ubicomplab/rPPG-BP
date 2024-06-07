import numpy as np
import pandas as pd
import os
import glob
import argparse
import scipy
import scipy.io
from matplotlib import pyplot as plt
from hrvanalysis import remove_ectopic_beats, interpolate_nan_values


def concat_chunks(chunks):

    concat = chunks[0]
    pos_list = [0, len(chunks[0]) - 1]
    for i in range(1, len(chunks)):
        concat = np.concatenate((concat, chunks[i]), axis=None)
        pos_list.append(len(concat) - 1)
    return concat, pos_list

def remove_artifact(ppg_signal):

    ppg_signal_mean = np.mean(ppg_signal)
    ppg_signal_std = np.std(ppg_signal[:200])

    ppg_signal = np.where(ppg_signal < ppg_signal_mean + 3 * ppg_signal_std, ppg_signal, ppg_signal_mean)
    ppg_signal = np.where(ppg_signal > ppg_signal_mean - 3 * ppg_signal_std, ppg_signal, ppg_signal_mean)

    return ppg_signal

def filter_ppg_signal(ppg_signal, LPF, HPF, FS):

    NyquistF = 1 / 2 * FS
    [b_pulse, a_pulse] = scipy.signal.butter(3, [LPF / NyquistF, HPF / NyquistF], btype='bandpass')
    ppg_filt = scipy.signal.filtfilt(b_pulse, a_pulse, np.double(ppg_signal))
    
    return ppg_filt

def normalize_min_max(data):

    return (data - np.min(data)) / (np.max(data) - np.min(data))


def remove_ectopic(peaks):

    rr_intervals_list = np.diff(peaks)
    # This remove ectopic beats from signal
    nn_intervals_list = remove_ectopic_beats(rr_intervals=rr_intervals_list, method="malik")
    # This replace ectopic beats nan values with linear interpolation
    interpolated_nn_intervals = interpolate_nan_values(rr_intervals=nn_intervals_list)
    # tmp = np.cumsum(interpolated_nn_intervals).astype(np.int)
    return interpolated_nn_intervals


def pulse_segmentation(ppg, peaks, peak_no, FS, tdia=0):

    f = list()
    # peak_no is peak number
    if (((peaks[peak_no + 1] - peaks[peak_no]) / FS) > 1.5) or (((peaks[peak_no + 1] - peaks[peak_no]) / FS) < 0.5):
        return f, np.zeros((1,1))
    
    ppg_chunk = 1 - ppg[peaks[peak_no] : peaks[peak_no + 1] + 1]

    # First deriv of ppg
    ppg_chunk_return = ppg_chunk.copy()
    ppg_chunk_fd = np.diff(ppg_chunk)
    
    tmp_pos = np.asarray([(i / FS) for i in range(1, len(ppg_chunk))])
    
    axs[5, 1].plot(tmp_pos,  ppg_chunk_fd, "r")
    # Second deriv of ppg
    ppg_chunk_sd = np.diff(ppg_chunk_fd)
    ppg_chunk_sd_peaks, _ = scipy.signal.find_peaks(ppg_chunk_sd)
    
    tsys = np.argmax(ppg_chunk)
    # f1 => (tsys-tdia)/fs
    f.append((tsys - tdia) / FS)
    selected_index = np.where(ppg_chunk_sd_peaks > tsys, 1, 0)
    max_index = np.argmax(selected_index)
    Pd = ppg_chunk[ppg_chunk_sd_peaks[max_index]]
    td = ppg_chunk_sd_peaks[max_index]

    # f2 => td-tdia
    f.append((td - tdia) / FS)

    # f3 => (td-tdia)/RR
    f.append(f[1] / (len(ppg_chunk) / FS))

    # f4 => (Pd-Pdia)/|Psys-Pdia|
    Pdia = ppg_chunk[tdia + 1]
    f.append((Pd - Pdia) / np.abs(ppg_chunk[tsys] - Pdia))

    # f5 => max(dP/dt)/|Psys-Pdia|
    ppg_chunk_fd_maxi = np.argmax(ppg_chunk_fd)
    f.append(ppg_chunk_fd[ppg_chunk_fd_maxi] / np.abs(ppg_chunk[tsys] - Pdia))

    # f6 => RR
    f.append(len(ppg_chunk) / FS)

    # f7 => tdia - td
    f.append(len(ppg_chunk) / FS - td / FS)

    return f, ppg_chunk_return

def calculate_plot_correlation(feature1, feature2, fig, title):

    fig.scatter(feature1,  feature2)

    slope, intercept, r_coef, p_values, se = scipy.stats.linregress(np.transpose(feature1), np.transpose(feature2))

    fig.set_title(title + ": r: " + str(r_coef)[:6] + ", p: " + str(p_values)[:6])
    fig.set_box_aspect(1)

    return fig

def stat_ppg_chunk(ppg_chunks):
    
    # temp = np.array(ppg_chunks)
    max_len = max(len(x) for x in ppg_chunks)
    ppg_chunks_mat = np.zeros((len(ppg_chunks), max_len))
    for i in range(len(ppg_chunks)):
        ppg_chunks_mat[i, :len(ppg_chunks[i])] = ppg_chunks[i]

    template_mean = np.mean(ppg_chunks_mat, axis=0)
    template_median = np.median(ppg_chunks_mat, axis=0)
    zero_index = np.argmin(template_median)
    return template_mean[:zero_index], template_median[:zero_index]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="settings")
    parser.add_argument("-m", "--mat_path", type=str, default=".Datasets/UWMedicine/ProcessedNoVideoDataV2/*.npy", help="ppg mat folder path")
    parser.add_argument("-s", "--save_path", type=str, default="./bp_preprocess/ppg_chunks_mat_face_v2_region_0_3_8_11", help="save path of ppg filtered by sqi")
    parser.add_argument("-d", "--df_path", type=str, default="./BPData_230223_demograph.csv", help="patient bp csv")
    opt = parser.parse_args()

    stat_template_ppg = False
    visualize_segmentation = False

    all_mat_folder = opt.mat_path
    chunk_mat_savepath = opt.save_path
    bp_csv_path = opt.df_path
    all_mat_path = sorted(glob.glob(all_mat_folder))
    bp_df = pd.read_csv(bp_csv_path)

    all_features = list()
    bp_sys = list()
    bp_dia = list()
    hr = list()
    age = list()
    wgt = list()
    hgt = list()

    LPF = 0.7 # low cutoff frequency (Hz) - 0.7 Hz in reference
    FS = 60

    for mat_path in all_mat_path:
        video_name = mat_path.split("/")[-1].split(".")[0]

        mat = np.load(mat_path, allow_pickle=True)
        mat_data = mat.item()
        finger_ppg = mat_data["gt_ppg"]
        face_ppg = (mat_data["face_patch_green"][1] + mat_data["face_patch_green"][2] + mat_data["face_patch_green"][0] + mat_data["face_patch_green"][3] + 
                    mat_data["face_patch_green"][8] + mat_data["face_patch_green"][9] + mat_data["face_patch_green"][10] + mat_data["face_patch_green"][11] ) / 8
        face_ppg = np.reshape(face_ppg, (face_ppg.shape[0]))

        mat_pid = mat_path.split(".")[0].split("/")[-1][:-1]
        df_tmp = bp_df[bp_df["PID"]==mat_pid]
        bp_sys.append(df_tmp.iloc[0, 4])
        bp_dia.append(df_tmp.iloc[0, 8])
        hr.append(df_tmp.iloc[0, 12])

        plt.figure()
        fig, axs = plt.subplots(6, 2, figsize=(60, 15), dpi=120)
        # Plot Raw Green Channel
        axs[0, 0].plot(finger_ppg)
        axs[0, 0].set_title("Original finger ppg")
        axs[0, 1].plot(face_ppg)
        axs[0, 1].set_title("Original face ppg")

        # Remove artifacts
        face_t = np.asarray([i for i in range(len(face_ppg))])
        finger_t = np.asarray([i for i in range(len(finger_ppg))])
        face_ppg = remove_artifact(face_ppg)
        
        axs[1, 1].plot(face_ppg)
        axs[1, 1].set_title("Remove artifacts face ppg")


        ### Filter
        HPF = 2 # high cutoff frequency (Hz) - 2.0 Hz in reference
        face_ppg_filt = filter_ppg_signal(face_ppg, LPF, HPF, FS)
        HPF = 16 # high cutoff frequency (Hz) - 8.0 Hz in reference
        face_ppg_filt_16 = filter_ppg_signal(face_ppg, LPF, HPF, FS)
        
        axs[2, 1].plot(face_ppg_filt)
        axs[2, 1].set_title("Filtered face ppg")

        ### Detect Peaks
        # Normalize
        face_ppg_norm = normalize_min_max(face_ppg_filt)
        peaks, _ = scipy.signal.find_peaks(face_ppg_norm, height=0.7, distance=20)
        axs[3, 1].plot(face_t, face_ppg_norm)
        axs[3, 1].plot(face_t[peaks], face_ppg_norm[peaks], "ro")
        axs[3, 1].set_title("Peak and Troughs face ppg")
    
        ### Do this for finger ppg
        finger_ppg_norm = normalize_min_max(finger_ppg)
        finger_peaks, _ = scipy.signal.find_peaks(finger_ppg_norm, height=0.7, distance=20)
        axs[3, 0].plot(finger_t, finger_ppg_norm)
        axs[3, 0].plot(finger_t[finger_peaks], finger_ppg_norm[finger_peaks], "ro")
        axs[3, 0].set_title("Peak finger ppg")

        ### Detect Troughs
        troughs, _ = scipy.signal.find_peaks(1 - face_ppg_norm, height=0.7, distance=20)
        finger_troughs, _ = scipy.signal.find_peaks(1 - finger_ppg_norm, height=0.7, distance=20)

        axs[3, 0].plot(finger_t[finger_troughs], finger_ppg_norm[finger_troughs], "go")
        axs[3, 1].plot(face_t[troughs], face_ppg_norm[troughs], "go")

        ### Remove Ectopic Beats
        if len(peaks) > 1:
            interpolated_nn_intervals = remove_ectopic(peaks)
            axs[4, 1].plot(peaks[1:], np.array(interpolated_nn_intervals) / FS)
            axs[4, 1].set_title("IBIs")
            axs[4, 0].plot(finger_peaks[1:], np.diff(finger_peaks) / FS)
        
        ### Pulse Segmentation
        # tdia = 0
        tmp_features = list()
        chunk_list = list()

        try:
            for peak_no in range(2, 50):

                feature, ppg_chunk = pulse_segmentation(face_ppg_filt_16, peaks, peak_no, FS)
                if len(feature) < 1:
                    continue
                tmp_features.append(feature)

                chunk_list.append(ppg_chunk)
            
            if visualize_segmentation:
                whole_ppg, pos = concat_chunks(chunk_list)
                fig, ax = plt.subplots(figsize=(60, 10), dpi=80)
                ax.plot(whole_ppg)
                ax.plot(pos, whole_ppg[pos], "ro")
                for i in range(len(pos)):
                    ax.annotate(str(i), xy=(pos[i], 0.95), fontsize=10)
                plt.savefig(os.path.join(opt.save_path, video_name + "_ppg_visualization.png"))
                plt.close()

            if stat_template_ppg:
                template_mean_chunk, template_median_chunk = stat_ppg_chunk(chunk_list)
                plt.figure()
                plt.plot(template_mean_chunk, label="mean")
                plt.plot(template_median_chunk, label="medium")
                save_chunk_folder = os.path.join(chunk_mat_savepath, video_name)
                if not os.path.isdir(save_chunk_folder):
                        os.makedirs(save_chunk_folder)
                plt.savefig(os.path.join(save_chunk_folder, "template.png"))

            all_features.append(np.median(tmp_features, axis=0))
            np.save(os.path.join(chunk_mat_savepath, video_name + ".npy"), np.array(chunk_list))
        except IndexError:
            print("index out of range")
            all_features.append(np.asarray([0 for i in range(7)]))