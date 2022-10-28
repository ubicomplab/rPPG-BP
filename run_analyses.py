import numpy as np
import scipy.io
from matplotlib import pyplot as plt
import glob
from scipy.signal import butter
import scipy
from hrvanalysis import remove_outliers, remove_ectopic_beats, interpolate_nan_values
from scipy import stats
import os
from scipy.sparse import spdiags
import pandas as pd
import math


if __name__ == "__main__":

    all_mat_folder = "/gscratch/ubicomp/cm74/clinical_data/ProcessedDataNoVideo/*.mat"
    bp_csv_path = "/gscratch/ubicomp/cm74/clinical_data/BPData_header.csv"
    figure_savepath = "/gscratch/ubicomp/cm74/bp/bp_preprocess/test_figure_plots"
    all_mat_path = sorted(glob.glob(all_mat_folder))
    bp_df = pd.read_csv(bp_csv_path)

    all_features = list()
    bp_sys = list()
    bp_dia = list()
    hr = list()
    age = list()
    wgt = list()
    hgt = list()

    count = 0
    for mat_path in all_mat_path:
        
        mat_data = scipy.io.loadmat(mat_path)
        finger_ppg = 1 - mat_data["ppg"][0]
        face_ppg =  mat_data["ppg_face"][0]

        mat_pid = mat_path.split(".")[0].split("/")[-1][:-1]
        df_tmp = bp_df[bp_df["PID"]==mat_pid]
        bp_sys.append(df_tmp.iloc[0, 4])
        bp_dia.append(df_tmp.iloc[0, 8])
        hr.append(df_tmp.iloc[0, 12])
        age.append(df_tmp.iloc[0, 13])
        wgt.append(df_tmp.iloc[0, 14])
        hgt.append(df_tmp.iloc[0, 15])

        # Plot Raw Green Channel
        plt.figure()
        fig, axs = plt.subplots(6, 2, figsize=(60, 15), dpi=120)
        axs[0, 0].plot(finger_ppg)
        axs[0, 0].set_title("Original finger ppg")
        axs[0, 1].plot(face_ppg)
        axs[0, 1].set_title("Original face ppg")
       
        # plt.figure(figsize=(15, 6), dpi=80)
        # plt.plot(finger_ppg)
        # plt.savefig("test_ppg.png")

        # plt.figure(figsize=(15, 6), dpi=80)
        # plt.plot(face_ppg)
        # plt.savefig("test_face_ppg.png")

        # Remove artifacts
        face_t = np.asarray([i for i in range(len(face_ppg))])
        finger_t = np.asarray([i for i in range(len(finger_ppg))])

        face_ppg_mean = np.mean(face_ppg)
        face_ppg_std = np.std(face_ppg[:200])
        ### TODO: Xin replaces the value with the mean; Danial replaces the value with 0
        face_ppg = np.where(face_ppg < face_ppg_mean + 3 * face_ppg_std, face_ppg, 0)
        face_ppg = np.where(face_ppg > face_ppg_mean - 3 * face_ppg_std, face_ppg, 0)

        axs[1, 1].plot(face_ppg)
        axs[1, 1].set_title("Remove artifacts face ppg")
        # plt.figure(figsize=(15, 6), dpi=80)
        # plt.plot(face_ppg)
        # plt.savefig("test_face_ppg_new.png")
        # index_1 = np.argwhere(face_ppg > face_ppg_mean + 3 * face_ppg_std)
        # index_2 = np.argwhere(face_ppg < face_ppg_mean - 3 * face_ppg_std)
        # index_intersect = np.intersect1d(index_1, index_2)
        # print(index_intersect)

        ### Filter
        LPF = 0.7 # low cutoff frequency (Hz) - 0.7 Hz in reference
        HPF = 2 # high cutoff frequency (Hz) - 4.0 Hz in reference
        FS = 60
        NyquistF = 1 /2 * FS
        [b_pulse, a_pulse] = scipy.signal.butter(3, [LPF / NyquistF, HPF / NyquistF], btype='bandpass')
        face_ppg_filt = scipy.signal.filtfilt(b_pulse, a_pulse, np.double(face_ppg))
        
        HPF = 8 # high cutoff frequency (Hz) - 4.0 Hz in reference
        [b_pulse, a_pulse] = scipy.signal.butter(3, [LPF / NyquistF, HPF / NyquistF], btype='bandpass')
        face_ppg_filt_4 = scipy.signal.filtfilt(b_pulse, a_pulse, np.double(face_ppg))
        ### Matlab Line 59-70 ?
        # plt.figure(figsize=(15, 6), dpi=80)
        # plt.plot(face_ppg_filt)
        # plt.savefig("test_face_ppg_filtered.png")
        axs[2, 1].plot(face_ppg_filt)
        axs[2, 1].set_title("Filtered face ppg")

        ### Detect Peaks
        # Normalize
        face_ppg_filt_normal = (face_ppg_filt - np.min(face_ppg_filt)) / (np.max(face_ppg_filt) - np.min(face_ppg_filt))
        peaks, _ = scipy.signal.find_peaks(face_ppg_filt_normal, height=0.7, distance=20)
        # plt.figure(figsize=(15, 6), dpi=80)
        # plt.plot(face_t[peaks], face_ppg_filt[peaks])
        # plt.savefig("test_face_ppg_peak.png")
        axs[3, 1].plot(face_t, face_ppg_filt_normal)
        axs[3, 1].plot(face_t[peaks], face_ppg_filt_normal[peaks], "ro")
        axs[3, 1].set_title("Peak and Troughs face ppg")

        ### Detect Troughs
        

        ### Do this for finger ppg
        finger_ppg_normal = (finger_ppg - np.min(finger_ppg)) / (np.max(finger_ppg) - np.min(finger_ppg))
        finger_peaks, _ = scipy.signal.find_peaks(finger_ppg_normal, height=0.7, distance=20)
        axs[3, 0].plot(finger_t, finger_ppg_normal)
        axs[3, 0].plot(finger_t[finger_peaks], finger_ppg_normal[finger_peaks], "ro")
        axs[3, 0].set_title("Peak finger ppg")
        # plt.figure(figsize=(15, 6), dpi=80)
        # plt.plot(finger_t[finger_peaks], finger_ppg_normal[finger_peaks])
        # plt.savefig("test_finger_ppg_peak.png")

        ### Detect Troughs
        troughs, _ = scipy.signal.find_peaks(1 - face_ppg_filt_normal, height=0.7, distance=20)
        finger_troughs, _ = scipy.signal.find_peaks(1 - finger_ppg_normal, height=0.7, distance=20)

        axs[3, 0].plot(finger_t[finger_troughs], finger_ppg_normal[finger_troughs], "go")
        axs[3, 1].plot(face_t[troughs], face_ppg_filt_normal[troughs], "go")

        ### Remove Ectopic Beats TODO
        rr_intervals_list = np.diff(peaks)
        # # This remove ectopic beats from signal
        nn_intervals_list = remove_ectopic_beats(rr_intervals=rr_intervals_list, method="malik")
        # This replace ectopic beats nan values with linear interpolation
        interpolated_nn_intervals = interpolate_nan_values(rr_intervals=nn_intervals_list)
        tmp = np.cumsum(interpolated_nn_intervals).astype(np.int)
        axs[4, 1].plot(peaks[1:], np.array(interpolated_nn_intervals) / FS)
        # plt.savefig(os.path.join(figure_savepath, mat_pid + "_ppg_subplots.png"))

        ### Pulse Segmentation
        tdia = 0
        tmp_features = list()

        try:
            for peak_no in range(2, 50):
                f = list()
                # peak_no is peak number
                if (((peaks[peak_no + 1] - peaks[peak_no]) / FS) > 1.5) or (((peaks[peak_no + 1] - peaks[peak_no]) / FS) < 0.5):
                    continue
                
                tmp_face_ppg = 1 - face_ppg_filt_4[peaks[peak_no] : peaks[peak_no + 1] + 1]
                
                tmp_pos = np.asarray([(i / 60) for i in range(1, len(tmp_face_ppg))])
                tmp_face_ppg_first_deriv = np.diff(tmp_face_ppg)
                
                axs[5, 1].plot(tmp_pos,  tmp_face_ppg_first_deriv, "r")
                
                tmp_face_ppg_second_deriv = np.diff(tmp_face_ppg_first_deriv)
                tmp_face_ppg_second_deriv_peaks, _ = scipy.signal.find_peaks(tmp_face_ppg_second_deriv)
                
                tsys = np.argmax(tmp_face_ppg)
                # f1 => (tsys-tdia)/fs
                f.append((tsys - tdia) / FS)
                selected_index = np.where(tmp_face_ppg_second_deriv_peaks > tsys, 1, 0)
                max_index = np.argmax(selected_index)
                Pd = tmp_face_ppg[tmp_face_ppg_second_deriv_peaks[max_index]]
                td = tmp_face_ppg_second_deriv_peaks[max_index]

                # f2 => td-tdia
                f.append((td - tdia) / FS)

                # f3 => (td-tdia)/RR
                f.append(f[1] / (len(tmp_face_ppg) / FS))

                # f4 => (Pd-Pdia)/|Psys-Pdia|
                Pdia = tmp_face_ppg[tdia + 1]
                f.append((Pd - Pdia) / np.abs(tmp_face_ppg[tsys] - Pdia))

                # f5 => max(dP/dt)/|Psys-Pdia|ßß
                max_peak_tmp_face_ppg_first_deriv_index = np.argmax(tmp_face_ppg_first_deriv)
                f.append(tmp_face_ppg_first_deriv[max_peak_tmp_face_ppg_first_deriv_index] / np.abs(tmp_face_ppg[tsys] - Pdia))

                # f6 => RR
                f.append(len(tmp_face_ppg) / FS)

                # f7 => tdia - td
                f.append(len(tmp_face_ppg) / FS - td / FS)

                tmp_features.append(f)
        
            # plt.savefig(os.path.join(figure_savepath, mat_pid + "_ppg_subplots.png"))
            
            all_features.append(np.median(tmp_features, axis=0))

        except IndexError:
            print("index out of range")
            all_features.append(np.asarray([0 for i in range(7)]))
            
        # break
    # print(bp_csv)
    fig2, axs2 = plt.subplots(5, 7, figsize=(40, 40), dpi=120)
    feature_names = ['Sys. Ramp','Eject Dur.','Eject Dur./RR','Norm. Dia. Notch Hei.','Max Sys. Ramp','RR','Dia. Notch to Foot','Sys. Peak to Foot']
    all_features_array = np.asarray(all_features)
    bp_sys = np.array(bp_sys)
    bp_dia = np.array(bp_dia)
    for i in range(7):
        temp_feature = all_features_array[:,i]
        feature_index = np.argwhere(temp_feature!=0)
        temp_index = np.array(feature_index.flatten(), dtype=np.int8)
        axs2[0, i].scatter(temp_feature[temp_index],  bp_sys[temp_index])
        ### Tried different lib function to caculate the r and p 
        # r_coef = np.corrcoef(temp_feature[temp_index], bp_sys[temp_index])
        # _, p_values = scipy.stats.pearsonr(temp_feature[temp_index], bp_sys[temp_index])
        slope, intercept, r_coef, p_values, se = scipy.stats.linregress(np.transpose(temp_feature[temp_index]), np.transpose(bp_sys[temp_index]))
        # axs2[0, i].set_title(feature_names[i] + ": r: " + str(r_coef[0, 1]) + ", p: " + str(p_values))
        axs2[0, i].set_title(feature_names[i] + ": r: " + str(r_coef)[:6] + ", p: " + str(p_values)[:6])
        axs2[0, i].set_box_aspect(1)
    
    hr_array = np.reshape(np.transpose(np.array(hr)), (len(hr), 1))
    all_features_normal = all_features_array / hr_array
    for i in range(7):
        temp_feature = all_features_normal[:,i]
        feature_index = np.argwhere(temp_feature!=0)
        temp_index = np.array(feature_index.flatten(), dtype=np.int8)
        axs2[1, i].scatter(temp_feature[temp_index],  bp_sys[temp_index])
        # r_coef = np.corrcoef(temp_feature[temp_index], bp_sys[temp_index])
        slope, intercept, r_coef, p_values, se = scipy.stats.linregress(np.transpose(temp_feature[temp_index]), np.transpose(bp_sys[temp_index]))
        axs2[1, i].set_title(feature_names[i] + ": r: " + str(r_coef)[:6] + ", p: " + str(p_values)[:6])
        axs2[1, i].set_box_aspect(1)
    
    axs2[2, 0].scatter(age,  bp_sys)
    # r_coef = np.corrcoef(age, bp_sys)
    slope, intercept, r_coef, p_values, se = scipy.stats.linregress(np.transpose(age), np.transpose(bp_sys))
    axs2[2, 0].set_title("AGE" + ": r: " + str(r_coef)[:6] + ", p: " + str(p_values)[:6])
    axs2[2, 0].set_box_aspect(1)

    axs2[2, 1].scatter(wgt,  bp_sys)
    # r_coef = np.corrcoef(np.transpose(wgt), np.transpose(bp_sys))
    slope, intercept, r_coef, p_values, se = scipy.stats.linregress(np.transpose(wgt), np.transpose(bp_sys))
    axs2[2, 1].set_title("WEIGHT" + ": r: " + str(r_coef)[:6] + ", p: " + str(p_values)[:6])
    axs2[2, 1].set_box_aspect(1)

    axs2[2, 2].scatter(hgt,  bp_sys)
    # r_coef = np.corrcoef(hgt, bp_sys)
    slope, intercept, r_coef, p_values, se = scipy.stats.linregress(np.transpose(hgt), np.transpose(bp_sys))
    axs2[2, 2].set_title("HEIGHT" + ": r: " + str(r_coef)[:6] + ", p: " + str(p_values)[:6])
    axs2[2, 2].set_box_aspect(1)

    w_h = np.array(wgt) / np.array(hgt)
    axs2[2, 3].scatter(w_h,  bp_sys)
    # r_coef = np.corrcoef(w_h, bp_sys)
    slope, intercept, r_coef, p_values, se = scipy.stats.linregress(np.transpose(w_h), np.transpose(bp_sys))
    axs2[2, 3].set_title("WEIGHT / HEIGHT" + ": r: " + str(r_coef)[:6] + ", p: " + str(p_values)[:6])
    axs2[2, 3].set_box_aspect(1)

    axs2[2, 4].scatter(hr,  bp_sys)
    # r_coef = np.corrcoef(hr, bp_sys)
    slope, intercept, r_coef, p_values, se = scipy.stats.linregress(np.transpose(hr), np.transpose(bp_sys))
    axs2[2, 4].set_title("HR" + ": r: " + str(r_coef)[:6] + ", p: " + str(p_values)[:6])
    axs2[2, 4].set_box_aspect(1)

    axs2[2, 5].set_box_aspect(1)
    axs2[2, 6].set_box_aspect(1)

    ### TODO: Diastolic

    for i in range(7):
        temp_feature = all_features_normal[:,i]
        feature_index = np.argwhere(temp_feature!=0)
        temp_index = np.array(feature_index.flatten(), dtype=np.int8)
        axs2[3, i].scatter(temp_feature[temp_index],  bp_dia[temp_index])
        # r_coef = np.corrcoef(temp_feature[temp_index], bp_dia[temp_index])
        slope, intercept, r_coef, p_values, se = scipy.stats.linregress(np.transpose(temp_feature[temp_index]), np.transpose(bp_dia[temp_index]))
        axs2[3, i].set_title(feature_names[i] + ": r: " + str(r_coef)[:6] + ", p: " + str(p_values)[:6])
        axs2[3, i].set_box_aspect(1)
    

    axs2[4, 0].scatter(age,  bp_dia)
    # r_coef = np.corrcoef(age, bp_dia)
    slope, intercept, r_coef, p_values, se = scipy.stats.linregress(np.transpose(age), np.transpose(bp_dia))
    axs2[4, 0].set_title("AGE" + ": r: " + str(r_coef)[:6] + ", p: " + str(p_values)[:6])
    axs2[4, 0].set_box_aspect(1)

    axs2[4, 1].scatter(wgt,  bp_dia)
    # r_coef = np.corrcoef(wgt, bp_dia)
    slope, intercept, r_coef, p_values, se = scipy.stats.linregress(np.transpose(wgt), np.transpose(bp_dia))
    axs2[4, 1].set_title("WEIGHT" + ": r: " + str(r_coef)[:6] + ", p: " + str(p_values)[:6])
    axs2[4, 1].set_box_aspect(1)

    axs2[4, 2].scatter(hgt,  bp_dia)
    # r_coef = np.corrcoef(hgt, bp_dia)
    slope, intercept, r_coef, p_values, se = scipy.stats.linregress(np.transpose(hgt), np.transpose(bp_dia))
    axs2[4, 2].set_title("HEIGHT" + ": r: " + str(r_coef)[:6] + ", p: " + str(p_values)[:6])
    axs2[4, 2].set_box_aspect(1)

    w_h = np.array(wgt) / np.array(hgt)
    axs2[4, 3].scatter(w_h,  bp_dia)
    # r_coef = np.corrcoef(w_h, bp_dia)
    slope, intercept, r_coef, p_values, se = scipy.stats.linregress(np.transpose(w_h), np.transpose(bp_dia))
    axs2[4, 3].set_title("WEIGHT / HEIGHT" + ": r: " + str(r_coef)[:6] + ", p: " + str(p_values)[:6])
    axs2[4, 3].set_box_aspect(1)

    axs2[4, 4].scatter(hr,  bp_dia)
    # r_coef = np.corrcoef(hr, bp_dia)
    slope, intercept, r_coef, p_values, se = scipy.stats.linregress(np.transpose(hr), np.transpose(bp_dia))
    axs2[4, 4].set_title("HR" + ": r: " + str(r_coef)[:6] + ", p: " + str(p_values)[:6])
    axs2[4, 4].set_box_aspect(1)

    axs2[4, 5].set_box_aspect(1)
    axs2[4, 6].set_box_aspect(1)
    
    plt.savefig(os.path.join(figure_savepath, mat_pid + "_ppg_subplots.png"))
    # plt.savefig("test_1028_new_dia_scipy.png")