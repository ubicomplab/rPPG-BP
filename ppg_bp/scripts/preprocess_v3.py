import numpy as np
import pandas as pd
import os
import glob
import scipy
import scipy.io
from matplotlib import pyplot as plt
from hrvanalysis import remove_ectopic_beats, interpolate_nan_values


def concat_chunks(chunks, pos):

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


def remove_artifact(ppg_signal):

    ppg_signal_mean = np.mean(ppg_signal)
    ppg_signal_std = np.std(ppg_signal[:200])
    ### TODO: Xin replaces the value with the mean; Danial replaces the value with 0
    # ppg_signal = np.where(ppg_signal < ppg_signal_mean + 3 * ppg_signal_std, ppg_signal, 0)
    # ppg_signal = np.where(ppg_signal > ppg_signal_mean - 3 * ppg_signal_std, ppg_signal, 0)
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
    # ppg_chunk = 1 - ppg[peaks[peak_no] : peaks[peak_no + 5] + 1]
    # First deriv of ppg
    ppg_chunk_return = ppg_chunk.copy()
    ppg_chunk_fd = np.diff(ppg_chunk)
    
    tmp_pos = np.asarray([(i / FS) for i in range(1, len(ppg_chunk))])
    
    axs[5, 1].plot(tmp_pos,  ppg_chunk_fd, "r")
    # Second deriv of ppg
    ppg_chunk_sd = np.diff(ppg_chunk_fd)
    ppg_chunk_sd_peaks, _ = scipy.signal.find_peaks(ppg_chunk_sd)
    
    # axs[5, 0].plot(tmp_pos[:-1],  tmp_face_ppg_second_deriv, "r")
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

    # f5 => max(dP/dt)/|Psys-Pdia|ßß
    ppg_chunk_fd_maxi = np.argmax(ppg_chunk_fd)
    f.append(ppg_chunk_fd[ppg_chunk_fd_maxi] / np.abs(ppg_chunk[tsys] - Pdia))

    # f6 => RR
    f.append(len(ppg_chunk) / FS)

    # f7 => tdia - td
    f.append(len(ppg_chunk) / FS - td / FS)

    return f, ppg_chunk_return

def calculate_plot_correlation(feature1, feature2, fig, title):

    fig.scatter(feature1,  feature2)
    # r_coef = np.corrcoef(np.transpose(feature1), np.transpose(feature2))
    # _, p_values = scipy.stats.pearsonr(temp_feature[temp_index], bp_sys[temp_index])
    slope, intercept, r_coef, p_values, se = scipy.stats.linregress(np.transpose(feature1), np.transpose(feature2))
    # fig.set_title(title + ": r: " + str(r_coef[0, 1]) + ", p: " + str(p_values))
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

def select_chunk(mat_data, chunk_amount):
    
    score = dict()
    start = 0
    end = len(mat_data[0]) + len(mat_data[1]) + len(mat_data[2]) + len(mat_data[3]) + len(mat_data[4])
    for i in range(len(mat_data) - chunk_amount - 1):

        skew1 = scipy.stats.skew(mat_data[i])
        skew2 = scipy.stats.skew(mat_data[i + 1])
        skew3 = scipy.stats.skew(mat_data[i + 2])
        skew4 = scipy.stats.skew(mat_data[i + 3])
        skew5 = scipy.stats.skew(mat_data[i + 4])
        average = (skew1 + skew2 + skew3 + skew4 + skew5) / chunk_amount

        score[i] = [average, [start, end]]

        start += len(mat_data[i])
        end += len(mat_data[i + 5])
        

    temp_score = sorted(score.items(), key=lambda item: item[1][0], reverse=True)
    top3 = temp_score[:3]
    print(top3)
    top3_chunk_list = list()
    top3_chunk_pos = list()
    for pos in top3:
        new_chunks = mat_data[int(pos[0]) : int(pos[0]) + chunk_amount]
        if len(new_chunks) != 5:
            print(mat_path)
            # print(len(new_chunks))
            # empty[mat_path] = len(new_chunks)
            continue
        top3_chunk_list.append(new_chunks)
        # top3_chunk_pos.append((int(pos[0]), int(pos[0]) + chunk_amount))
        top3_chunk_pos.append((int(pos[1][1][0]), int(pos[1][1][1])))
        # top3_chunk_list.append([pos[1], (int(pos[0]), int(pos[0]) + chunk_amount)])
    return top3_chunk_list, top3_chunk_pos


if __name__ == "__main__":

    save_wave_fig = False
    save_chunk_fig = False
    stat_template_ppg = False
    draw_select_ppg = False

    # all_mat_folder = "/gscratch/ubicomp/cm74/clinical_data/ProcessedDataNoVideo/*.mat"
    # all_mat_folder = "/gscratch/ubicomp/xliu0/rppg_clinical_study/tscan_ppg/*.mat"
    all_mat_folder = "/gscratch/ubicomp/xliu0/data3/mnt/Datasets/UWMedicine/ProcessedDataNoVideo/*.mat"
    # bp_csv_path = "/gscratch/ubicomp/cm74/clinical_data/BPData_header.csv"
    bp_csv_path = "/gscratch/ubicomp/cm74/clinical_data/BPData_230103.csv"
    figure_savepath = "/gscratch/ubicomp/cm74/bp/bp_preprocess/test_figure_plots_5"
    chunk_mat_savepath = "/gscratch/ubicomp/cm74/bp/bp_preprocess/ppg_chunks_mat_face_palm_230124"
    all_mat_path = sorted(glob.glob(all_mat_folder))
    bp_df = pd.read_csv(bp_csv_path)

    all_features = list()
    bp_sys = list()
    bp_dia = list()
    hr = list()
    age = list()
    wgt = list()
    hgt = list()

    ppg_rank = dict()

    count = 0
    for mat_path in all_mat_path:
        video_name = mat_path.split("/")[-1].split(".")[0]
        mat_data = scipy.io.loadmat(mat_path)
        finger_ppg = mat_data["ppg"][0]
        # finger_ppg = mat_data["ppg"][0]
        face_ppg = mat_data["ppg_face"][0]
        # palm_ppg = mat_data["ppg_palm"][0]
        palm_ppg = 1 - mat_data["ppg_palm"][0]

        mat_pid = mat_path.split(".")[0].split("/")[-1][:-1]
        df_tmp = bp_df[bp_df["PID"]==mat_pid]
        bp_sys.append(df_tmp.iloc[0, 4])
        bp_dia.append(df_tmp.iloc[0, 8])
        hr.append(df_tmp.iloc[0, 12])
        # age.append(df_tmp.iloc[0, 13])
        # wgt.append(df_tmp.iloc[0, 14])
        # hgt.append(df_tmp.iloc[0, 15])

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
        palm_ppg = remove_artifact(palm_ppg)
        
        axs[1, 1].plot(face_ppg)
        axs[1, 1].set_title("Remove artifacts face ppg")


        ### Filter
        LPF = 0.7 # low cutoff frequency (Hz) - 0.7 Hz in reference
        HPF = 2 # high cutoff frequency (Hz) - 2.0 Hz in reference
        FS = 60
        face_ppg_filt = filter_ppg_signal(face_ppg, LPF, HPF, FS)
        HPF = 10 # high cutoff frequency (Hz) - 8.0 Hz in reference
        face_ppg_filt_8 = filter_ppg_signal(face_ppg, LPF, HPF, FS)
        palm_ppg_filt_8 = filter_ppg_signal(palm_ppg, LPF, HPF, FS)
        
        axs[2, 1].plot(face_ppg_filt)
        axs[2, 1].set_title("Filtered face ppg")

        ### Detect Peaks
        # Normalize
        face_ppg_norm = normalize_min_max(face_ppg_filt)
        peaks, _ = scipy.signal.find_peaks(face_ppg_norm, height=0.7, distance=20)
        axs[3, 1].plot(face_t, face_ppg_norm)
        axs[3, 1].plot(face_t[peaks], face_ppg_norm[peaks], "ro")
        axs[3, 1].set_title("Peak and Troughs face ppg")

        ### Do this for palm ppg
        # face_ppg_norm = normalize_min_max(_ppg_filt)

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
        select_ppg_chunk = list()
        try:
            for peak_no in range(2, 50):
                feature, ppg_chunk = pulse_segmentation(face_ppg_filt_8, peaks, peak_no, FS)
                if len(feature) < 1:
                    continue
                tmp_features.append(feature)
                # if save_chunk_fig:
                #     plt.figure()
                #     plt.plot(ppg_chunk)
                #     save_chunk_folder = os.path.join(chunk_mat_savepath, video_name)
                #     if not os.path.isdir(save_chunk_folder):
                #          os.makedirs(save_chunk_folder)
                #     plt.savefig(os.path.join(save_chunk_folder, str(peak_no) + ".png"))
                chunk_list.append(ppg_chunk)
            ### select pulse segmentation by skewness
            top3_chunk_list, top3_chunk_pos = select_chunk(np.array(chunk_list), chunk_amount=5)
            # if draw_select_ppg:
            #     plt.figure()
            #     plt.plot(np.array(top3_chunk_list), label="selected")
            #     plt.plot(np.array(chunk_list), label="all")
            for pos in top3_chunk_pos:
                select_ppg_chunk.append(palm_ppg_filt_8[pos[0]:pos[1]])
            # for selected in top3_chunk_list:
            #     ppg_rank[selected[0]] = [video_name, selected[1], np.array(chunk_list)]
                # ppg_rank[selected[0]] = [video_name]

            if stat_template_ppg:
                template_mean_chunk, template_median_chunk = stat_ppg_chunk(chunk_list)
                plt.figure()
                plt.plot(template_mean_chunk, label="mean")
                plt.plot(template_median_chunk, label="medium")
                save_chunk_folder = os.path.join(chunk_mat_savepath, video_name)
                if not os.path.isdir(save_chunk_folder):
                        os.makedirs(save_chunk_folder)
                plt.savefig(os.path.join(save_chunk_folder, "template.png"))

            if save_wave_fig:
                # plt.figure(1)
                plt.savefig(os.path.join(figure_savepath, video_name + "_ppg_subplots.png"))
                # np.save(os.path.join(chunk_mat_savepath, video_name + ".npy"), np.array(chunk_list))
            all_features.append(np.median(tmp_features, axis=0))
            np.save(os.path.join(chunk_mat_savepath, video_name + ".npy"), np.array([top3_chunk_list, select_ppg_chunk]))
        except IndexError:
            print("index out of range")
            all_features.append(np.asarray([0 for i in range(7)]))
    

    # ppg_rank_sort = sorted(ppg_rank.items(), key=lambda item: item[0], reverse=True)
    
    # fig_length = 5
    # fig3, axs3 = plt.subplots(fig_length, fig_length, figsize=(90, 30), dpi=160)

    # for i in range(fig_length):
    #     for j in range(fig_length):
    #         rank_chunk = ppg_rank_sort[i * fig_length + j]
    #         concat, select_map_pos = concat_chunks(rank_chunk[1][2], rank_chunk[1][1])
    #         axs3[i, j].set_box_aspect(0.2)
    #         axs3[i, j].plot(concat, label="all")
    #         axs3[i, j].axvspan(select_map_pos[0], select_map_pos[1], color="blue", alpha=0.3)
    #         axs3[i, j].set_title(rank_chunk[1][0] + " score: " + str(rank_chunk[0])[:6])
    #         # axs3[i, j].plot(concat, label="all")
    #         # axs3[i, j].plot(ppg_rank_sort[i * 5 + j][1][2], label="all")
    #         # axs3[i, j].title(ppg_rank_sort[i * 5 + j][1][0] + str(ppg_rank_sort[i * 5 + j][0])[:6])
    

    # plt.savefig(os.path.join(figure_savepath, "select_top_ppg_subplots.png"))

    # fig4, axs4 = plt.subplots(fig_length, fig_length, figsize=(90, 30), dpi=160)

    # for i in range(fig_length):
    #     for j in range(fig_length):
    #         rank_chunk = ppg_rank_sort[-(i * fig_length + j) - 1]
    #         concat, select_map_pos = concat_chunks(rank_chunk[1][2], rank_chunk[1][1])
    #         axs4[i, j].set_box_aspect(0.2)
    #         axs4[i, j].plot(concat, label="all")
    #         axs4[i, j].axvspan(select_map_pos[0], select_map_pos[1], color="blue", alpha=0.3)
    #         axs4[i, j].set_title(rank_chunk[1][0] + " score: " + str(rank_chunk[0])[:6])
    #         # axs3[i, j].plot(concat, label="all")
    #         # axs3[i, j].plot(ppg_rank_sort[i * 5 + j][1][2], label="all")
    #         # axs3[i, j].title(ppg_rank_sort[i * 5 + j][1][0] + str(ppg_rank_sort[i * 5 + j][0])[:6])
    

    # plt.savefig(os.path.join(figure_savepath, "select_bottom_ppg_subplots.png"))
    
    ### Correlation
    # Systolic
    # fig2, axs2 = plt.subplots(5, 7, figsize=(40, 40), dpi=120)
    # feature_names = ['Sys. Ramp','Eject Dur.','Eject Dur./RR','Norm. Dia. Notch Hei.','Max Sys. Ramp','RR','Dia. Notch to Foot','Sys. Peak to Foot']
    # all_features_array = np.asarray(all_features)
    # bp_sys = np.array(bp_sys)
    # bp_dia = np.array(bp_dia)
    # for i in range(7):
    #     temp_feature = all_features_array[:,i]
    #     feature_index = np.argwhere(temp_feature!=0)
    #     temp_index = np.array(feature_index.flatten(), dtype=np.int8)
    #     axs2[0, i] = calculate_plot_correlation(temp_feature[temp_index], bp_sys[temp_index],  axs2[0, i], feature_names[i])

    
    # hr_array = np.reshape(np.transpose(np.array(hr)), (len(hr), 1))
    # all_features_normal = all_features_array / hr_array
    # for i in range(7):
    #     temp_feature = all_features_normal[:,i]
    #     feature_index = np.argwhere(temp_feature!=0)
    #     temp_index = np.array(feature_index.flatten(), dtype=np.int8)
    #     axs2[1, i] = calculate_plot_correlation(temp_feature[temp_index], bp_sys[temp_index], axs2[1, i], feature_names[i])

    
    # axs2[2, 0] = calculate_plot_correlation(age, bp_sys, axs2[2, 0], "AGE")

    # axs2[2, 1] = calculate_plot_correlation(wgt, bp_sys, axs2[2, 1], "WEIGHT")

    # axs2[2, 2] = calculate_plot_correlation(hgt, bp_sys, axs2[2, 2], "HEIGHT")

    # w_h = np.array(wgt) / np.array(hgt)
    # axs2[2, 3] = calculate_plot_correlation(w_h, bp_sys, axs2[2, 3], "WEIGHT / HEIGHT")

    # axs2[2, 4] = calculate_plot_correlation(hr, bp_sys, axs2[2, 4], "HR")

    # axs2[2, 5].set_box_aspect(1)
    # axs2[2, 6].set_box_aspect(1)

    # ### Diastolic

    # for i in range(7):
    #     temp_feature = all_features_normal[:,i]
    #     feature_index = np.argwhere(temp_feature!=0)
    #     temp_index = np.array(feature_index.flatten(), dtype=np.int8)
    #     axs2[3, i] = calculate_plot_correlation(temp_feature[temp_index], bp_dia[temp_index],  axs2[3, i], feature_names[i])
    
    # axs2[4, 0] = calculate_plot_correlation(age, bp_dia, axs2[4, 0], "AGE")

    # axs2[4, 1] = calculate_plot_correlation(wgt, bp_dia, axs2[4, 1], "WEIGHT")

    # axs2[4, 2] = calculate_plot_correlation(hgt, bp_dia, axs2[4, 2], "HEIGHT")

    # w_h = np.array(wgt) / np.array(hgt)
    # axs2[4, 3] = calculate_plot_correlation(w_h, bp_dia, axs2[4, 3], "WEIGHT / HEIGHT")

    # axs2[4, 4] = calculate_plot_correlation(hr, bp_dia, axs2[4, 4], "HR")

    # axs2[4, 5].set_box_aspect(1)
    # axs2[4, 6].set_box_aspect(1)
    
    # plt.savefig(os.path.join(figure_savepath, "overall_ppg_subplots.png"))