import os
import random
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from preprocess import normalize_min_max, filter_ppg_signal


class UWBP_train_manual_demo(Dataset):

    def __init__(self, config):
        super().__init__()
        self.data_list_path = config["train_dataset_path"]
        self.data_list = self._load_data_list(self.data_list_path)
        self.bp_gt = pd.read_csv(config["bp_csv_path"])
        self.chunk_amount = config["chunk_amount"]
        self.chunk_length = config["chunk_length"]
        self.LPF = config["filter_lpf"]
        self.HPF = config["filter_hpf"]
        self.FS = config["frequency_sample"]
        self.frequency_field = config["frequency_field"]
        self.derivative = config["derivative_input"]
        # self.demo_shape = (1, 1)
        self.demo_shape = (1, 124)

    def _load_data_list(self, data_list_path):
        with open(data_list_path, "r") as f:
            lines = f.readlines()
        data_list = []
        for line in lines:
            data_item = line.strip("\n ").split(",")[0]
            data_list.append(data_item)
        return data_list
    
    def _concat_chunks(self, chunks):
        chunk = np.zeros((self.chunk_length))
        i = 0
        for ppg_chunk in chunks:
            chunk[i : i + len(ppg_chunk)] = np.squeeze(ppg_chunk)
            i = i + len(ppg_chunk)
        return chunk, i

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        mat_path = self.data_list[index]
        mat_data = np.load(mat_path, allow_pickle=True)
        if len(mat_data) < 1:
            print(mat_path)
        chunk_pos = random.randint(0, len(mat_data) - 1)
        chunk_ori = mat_data[chunk_pos]
        if len(chunk_ori) < self.chunk_amount:
            print(mat_path)
        if len(chunk_ori) != self.chunk_amount:
            sub_chunk_pos = random.randint(0, len(chunk_ori) - self.chunk_amount)
            chunk_ori = chunk_ori[sub_chunk_pos : sub_chunk_pos + self.chunk_amount]
        chunk, pos = self._concat_chunks(chunk_ori)
        chunk[pos:] = np.mean(chunk[:pos])
        chunk_filt = filter_ppg_signal(chunk, self.LPF, self.HPF, self.FS)
        chunk_filt_norm = normalize_min_max(chunk_filt)
        if self.frequency_field:
            chunk_filt_norm = np.real(np.fft.rfft(chunk_filt_norm))

        mat_pid = mat_path.split("/")[-1].split(".")[0][:-1]
        session = mat_path.split("/")[-1].split(".")[0][-1]
        df = self.bp_gt[self.bp_gt["PID"] == mat_pid]
        age = (df["age"].values[0] - 34) / (96 - 34) # Now the min age is 34 years old and oldest person is 96
        bmi = (df["bmi"].values[0] - 17) / (41 - 17) # Now the min bmi is 17.327 and the max bmi is 40.88
        gender = df["gender"].values[0]
        if gender == "M":
            gender = 0
        else:
            gender = 1
        age_array = np.zeros(self.demo_shape, dtype=np.float32)
        bmi_array = np.zeros(self.demo_shape, dtype=np.float32)
        gender_array = np.zeros(self.demo_shape, dtype=np.float32)
        age_array[:] = age
        bmi_array[:] = bmi
        gender_array[:] = gender
        # bp_sys = max(min((df.iloc[0, 4] - 100) / (180 - 100), 1), 0)
        # bp_dia = max(min((df.iloc[0, 8] - 55) / (100 - 55), 1), 0)
        if session == "a":
            bp_sys = (df.iloc[0, 1] + df.iloc[0, 2]) / 2
            bp_dia = (df.iloc[0, 5] + df.iloc[0, 6]) / 2
            bp_sys = float(max(min((bp_sys - 100) / (180 - 100), 1), 0))
            bp_dia = float(max(min((bp_dia - 55) / (100 - 55), 1), 0))
        else:
            bp_sys = (df.iloc[0, 2] + df.iloc[0, 3]) / 2
            bp_dia = (df.iloc[0, 6] + df.iloc[0, 7]) / 2
            bp_sys = float(max(min((bp_sys - 100) / (180 - 100), 1), 0))
            bp_dia = float(max(min((bp_dia - 55) / (100 - 55), 1), 0))

        if self.derivative:
            chunk_filt_norm_fd = np.diff(chunk_filt_norm)
            chunk_filt_norm_fd = np.insert(chunk_filt_norm_fd, 0, 0)
            chunk_filt_norm_sd = np.diff(chunk_filt_norm_fd)
            chunk_filt_norm_sd = np.insert(chunk_filt_norm_sd, 0, 0)
            item = {
                "ppg_chunk": np.vstack((np.reshape(chunk_filt_norm, (1, len(chunk_filt_norm))),
                            np.reshape(chunk_filt_norm_fd, (1, len(chunk_filt_norm_fd))),
                            np.reshape(chunk_filt_norm_sd, (1, len(chunk_filt_norm_sd))))),
                "bp": np.reshape(np.array([bp_sys, bp_dia]), (1, 2)),
                "age": np.reshape(age_array, self.demo_shape),
                "bmi": np.reshape(bmi_array,self.demo_shape),
                "gender": np.reshape(gender_array, self.demo_shape)
            }
        else:
            item = {
                "ppg_chunk": np.reshape(chunk_filt_norm, (1, len(chunk_filt_norm))),
                "bp": np.reshape(np.array([bp_sys, bp_dia]), (1, 2)),
                # "bp": np.reshape(np.array([bp_sys]), (1, 1)),
                "age": np.reshape(age_array, self.demo_shape),
                "bmi": np.reshape(bmi_array, self.demo_shape),
                "gender": np.reshape(gender_array, self.demo_shape)
            }

        return item


class UWBP_train_whole_demo_arry(Dataset):

    def __init__(self, config):
        super().__init__()
        self.data_list_path = config["train_dataset_path"]
        self.data_list = self._load_data_list(self.data_list_path)
        self.bp_gt = pd.read_csv(config["bp_csv_path"])
        self.chunk_amount = config["chunk_amount"]
        self.chunk_length = config["chunk_length"]
        self.LPF = config["filter_lpf"]
        self.HPF = config["filter_hpf"]
        self.FS = config["frequency_sample"]
        self.frequency_field = config["frequency_field"]
        self.derivative = config["derivative_input"]
        self.output_folder = config["output_dir"]
        self.feature_label = pd.read_csv(config["ekg_label_path"]).fillna(0)
        self.demo_shape = (1, 1)
        self.feature_shape = (1, 35)

    def _load_data_list(self, data_list_path):
        with open(data_list_path, "r") as f:
            lines = f.readlines()
        data_list = []
        for line in lines:
            data_item = line.strip("\n ").split(",")[0]
            data_list.append(data_item)
        return data_list
    
    def _concat_chunks(self, chunks):
        chunk = np.zeros((self.chunk_length))
        i = 0
        for ppg_chunk in chunks:
            if len(ppg_chunk) > self.chunk_length - i:
                chunk[i:] = np.squeeze(ppg_chunk[:self.chunk_length - i])
                i = self.chunk_length
            else:
                chunk[i : i + len(ppg_chunk)] = np.squeeze(ppg_chunk)
                i = i + len(ppg_chunk)
        return chunk, i

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        mat_path = self.data_list[index]
        mat_pid = mat_path.split("/")[-1].split(".")[0][:-1]
        chunk_ori = np.load(mat_path, allow_pickle=True)

        if len(chunk_ori) < self.chunk_amount:
            print(mat_path)
        if len(chunk_ori) != self.chunk_amount:
            sub_chunk_pos = random.randint(0, len(chunk_ori) - self.chunk_amount)
            chunk_ori = chunk_ori[sub_chunk_pos : sub_chunk_pos + self.chunk_amount]
        chunk, pos = self._concat_chunks(chunk_ori)
        chunk[pos:] = np.mean(chunk[:pos])
        chunk_filt = filter_ppg_signal(chunk, self.LPF, self.HPF, self.FS)
        chunk_filt_norm = normalize_min_max(chunk_filt)
        if self.frequency_field:
            chunk_filt_norm = np.real(np.fft.rfft(chunk_filt_norm))

        mat_pid = mat_path.split("/")[-1].split(".")[0][:-1]
        session = mat_path.split("/")[-1].split(".")[0][-1]
        df = self.bp_gt[self.bp_gt["PID"] == mat_pid]
        age = (df["age"].values[0] - 34) / (96 - 34) # Now the min age is 34 years old and oldest person is 96
        bmi = (df["bmi"].values[0] - 17) / (41 - 17) # Now the min bmi is 17.327 and the max bmi is 40.88
        gender = df["gender"].values[0]
        if gender == "M":
            gender = 0
        else:
            gender = 1
        age_array = np.zeros(self.demo_shape, dtype=np.float32)
        bmi_array = np.zeros(self.demo_shape, dtype=np.float32)
        gender_array = np.zeros(self.demo_shape, dtype=np.float32)
        feature_array = np.zeros(self.feature_shape, dtype=np.float32)
        age_array[:] = age
        bmi_array[:] = bmi
        gender_array[:] = gender

        tmp_feature_label_df = self.feature_label[self.feature_label["ID"] == mat_pid]
        tmp_feature_label_df = tmp_feature_label_df.drop(["ID", "Age", "BMI"], axis=1)
        feature_array[:] = tmp_feature_label_df.values[0]

        if session == "a":
            bp_sys = (df.iloc[0, 1] + df.iloc[0, 2]) / 2
            bp_dia = (df.iloc[0, 5] + df.iloc[0, 6]) / 2
            bp_sys = float(max(min((bp_sys - 100) / (180 - 100), 1), 0))
            bp_dia = float(max(min((bp_dia - 55) / (100 - 55), 1), 0))

        else:
            bp_sys = (df.iloc[0, 2] + df.iloc[0, 3]) / 2
            bp_dia = (df.iloc[0, 6] + df.iloc[0, 7]) / 2
            bp_sys = float(max(min((bp_sys - 100) / (180 - 100), 1), 0))
            bp_dia = float(max(min((bp_dia - 55) / (100 - 55), 1), 0))

        if self.derivative:
            chunk_filt_norm_fd = np.diff(chunk_filt_norm)
            chunk_filt_norm_fd = np.insert(chunk_filt_norm_fd, 0, 0)
            chunk_filt_norm_sd = np.diff(chunk_filt_norm_fd)
            chunk_filt_norm_sd = np.insert(chunk_filt_norm_sd, 0, 0)
            item = {
                "ppg_chunk": np.vstack((np.reshape(chunk_filt_norm, (1, len(chunk_filt_norm))),
                            np.reshape(chunk_filt_norm_fd, (1, len(chunk_filt_norm_fd))),
                            np.reshape(chunk_filt_norm_sd, (1, len(chunk_filt_norm_sd))))),
                "bp": np.reshape(np.array([bp_sys, bp_dia]), (1, 2)),
                "age": np.reshape(age_array, self.demo_shape),
                "bmi": np.reshape(bmi_array,self.demo_shape),
                "gender": np.reshape(gender_array, self.demo_shape)
            }
        else:
            item = {
                "ppg_chunk": np.reshape(chunk_filt_norm, (1, len(chunk_filt_norm))),
                "bp": np.reshape(np.array([bp_sys, bp_dia]), (1, 2)),
                # "bp": np.reshape(np.array([bp_sys]), (1, 1)),
                "age": np.reshape(age_array, self.demo_shape),
                "bmi": np.reshape(bmi_array, self.demo_shape),
                "feature": np.reshape(feature_array, self.feature_shape),
            }

        return item


class UWBP_train_demo_whole_cls(Dataset):

    def __init__(self, config):
        super().__init__()
        self.data_list_path = config["train_dataset_path"]
        self.data_list = self._load_data_list(self.data_list_path)
        self.bp_gt = pd.read_csv(config["bp_csv_path"])
        self.chunk_amount = config["chunk_amount"]
        self.chunk_length = config["chunk_length"]
        self.LPF = config["filter_lpf"]
        self.HPF = config["filter_hpf"]
        self.FS = config["frequency_sample"]
        self.frequency_field = config["frequency_field"]
        self.derivative = config["derivative_input"]
        self.output_folder = config["output_dir"]
        self.feature_label = pd.read_csv(config["ekg_label_path"]).fillna(0)
        self.demo_shape = (1, 1)
        self.feature_shape = (1, 35)

    def _load_data_list(self, data_list_path):
        with open(data_list_path, "r") as f:
            lines = f.readlines()
        data_list = []
        for line in lines:
            data_item = line.strip("\n ").split(",")[0]
            data_list.append(data_item)
        return data_list
    
    def _concat_chunks(self, chunks):
        chunk = np.zeros((self.chunk_length))
        i = 0
        for ppg_chunk in chunks:
            if len(ppg_chunk) > self.chunk_length - i:
                chunk[i:] = np.squeeze(ppg_chunk[:self.chunk_length - i])
                i = self.chunk_length
            else:
                chunk[i : i + len(ppg_chunk)] = np.squeeze(ppg_chunk)
                i = i + len(ppg_chunk)
        return chunk, i

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        mat_path = self.data_list[index]
        mat_pid = mat_path.split("/")[-1].split(".")[0][:-1]
        chunk_ori = np.load(mat_path, allow_pickle=True)

        if len(chunk_ori) < self.chunk_amount:
            print(mat_path)
        if len(chunk_ori) != self.chunk_amount:
            sub_chunk_pos = random.randint(0, len(chunk_ori) - self.chunk_amount)
            chunk_ori = chunk_ori[sub_chunk_pos : sub_chunk_pos + self.chunk_amount]
        chunk, pos = self._concat_chunks(chunk_ori)
        chunk[pos:] = np.mean(chunk[:pos])
        chunk_filt = filter_ppg_signal(chunk, self.LPF, self.HPF, self.FS)
        chunk_filt_norm = normalize_min_max(chunk_filt)
        if self.frequency_field:
            chunk_filt_norm = np.real(np.fft.rfft(chunk_filt_norm))

        mat_pid = mat_path.split("/")[-1].split(".")[0][:-1]
        session = mat_path.split("/")[-1].split(".")[0][-1]
        df = self.bp_gt[self.bp_gt["PID"] == mat_pid]
        age = (df["age"].values[0] - 34) / (96 - 34) # Now the min age is 34 years old and oldest person is 96
        bmi = (df["bmi"].values[0] - 17) / (41 - 17) # Now the min bmi is 17.327 and the max bmi is 40.88
        gender = df["gender"].values[0]
        if gender == "M":
            gender = 0
        else:
            gender = 1
        age_array = np.zeros(self.demo_shape, dtype=np.float32)
        bmi_array = np.zeros(self.demo_shape, dtype=np.float32)
        gender_array = np.zeros(self.demo_shape, dtype=np.float32)
        feature_array = np.zeros(self.feature_shape, dtype=np.float32)
        age_array[:] = age
        bmi_array[:] = bmi
        gender_array[:] = gender

        tmp_feature_label_df = self.feature_label[self.feature_label["ID"] == mat_pid]
        tmp_feature_label_df = tmp_feature_label_df.drop(["ID", "Age", "BMI"], axis=1)
        feature_array[:] = tmp_feature_label_df.values[0]

        if session == "a":
            bp_sys = (df.iloc[0, 1] + df.iloc[0, 2]) / 2
            bp_dia = (df.iloc[0, 5] + df.iloc[0, 6]) / 2

        else:
            bp_sys = (df.iloc[0, 2] + df.iloc[0, 3]) / 2
            bp_dia = (df.iloc[0, 6] + df.iloc[0, 7]) / 2
        
        if bp_sys >= 130:
            bp_sys = 1
        else:
            bp_sys = 0

        if self.derivative:
            chunk_filt_norm_fd = np.diff(chunk_filt_norm)
            chunk_filt_norm_fd = np.insert(chunk_filt_norm_fd, 0, 0)
            chunk_filt_norm_sd = np.diff(chunk_filt_norm_fd)
            chunk_filt_norm_sd = np.insert(chunk_filt_norm_sd, 0, 0)
            item = {
                "ppg_chunk": np.vstack((np.reshape(chunk_filt_norm, (1, len(chunk_filt_norm))),
                            np.reshape(chunk_filt_norm_fd, (1, len(chunk_filt_norm_fd))),
                            np.reshape(chunk_filt_norm_sd, (1, len(chunk_filt_norm_sd))))),
                "bp": np.reshape(np.array([bp_sys, bp_dia]), (1, 2)),
                "age": np.reshape(age_array, self.demo_shape),
                "bmi": np.reshape(bmi_array,self.demo_shape),
                "gender": np.reshape(gender_array, self.demo_shape)
            }
        else:
            item = {
                "ppg_chunk": np.reshape(chunk_filt_norm, (1, len(chunk_filt_norm))),
                # "bp": np.reshape(np.array([bp_sys, bp_dia]), (1, 2)),
                "bp": np.reshape(np.array([bp_sys]), (1, 1)),
                "age": np.reshape(age_array, self.demo_shape),
                "bmi": np.reshape(bmi_array, self.demo_shape),
                "feature": np.reshape(feature_array, self.feature_shape),
            }

        return item


class UWBP_val_manual_demo(UWBP_train_manual_demo):

    def __init__(self, config):
        super().__init__(config)
        self.data_list_path = config["val_dataset_path"]
        self.data_list = self._load_data_list(self.data_list_path)
    
    def __getitem__(self, index):
        mat_path = self.data_list[index]
        session_id = mat_path.split(".")[0][-5:]
        mat_data = np.load(mat_path, allow_pickle=True)
        # chunk_pos = random.randint(0, len(mat_data) - 1)
        data = list()
        for i in range(len(mat_data)):
            chunk_ori = mat_data[i]
            mat_pid = mat_path.split("/")[-1].split(".")[0][:-1]
            session = mat_path.split("/")[-1].split(".")[0][-1]
            df = self.bp_gt[self.bp_gt["PID"] == mat_pid]

            age = (df["age"].values[0] - 34) / (96 - 34) # Now the min age is 34 years old and oldest person is 96
            bmi = (df["bmi"].values[0] - 17) / (41 - 17) # Now the min bmi is 17.327 and the max bmi is 40.88
            gender = df["gender"].values[0]
            if gender == "M":
                gender = 0
            else:
                gender = 1
            
            age_array = np.zeros(self.demo_shape, dtype=np.float32)
            bmi_array = np.zeros(self.demo_shape, dtype=np.float32)
            gender_array = np.zeros(self.demo_shape, dtype=np.float32)
            age_array[:] = age
            bmi_array[:] = bmi
            gender_array[:] = gender
            # bp_sys = max(min((df.iloc[0, 4] - 100) / (180 - 100), 1), 0)
            # bp_dia = max(min((df.iloc[0, 8] - 55) / (100 - 55), 1), 0)
            if session == "a":
                bp_sys = (df.iloc[0, 1] + df.iloc[0, 2]) / 2
                bp_dia = (df.iloc[0, 5] + df.iloc[0, 6]) / 2
                bp_sys = float(max(min((bp_sys - 100) / (180 - 100), 1), 0))
                bp_dia = float(max(min((bp_dia - 55) / (100 - 55), 1), 0))
            else:
                bp_sys = (df.iloc[0, 2] + df.iloc[0, 3]) / 2
                bp_dia = (df.iloc[0, 6] + df.iloc[0, 7]) / 2
                bp_sys = float(max(min((bp_sys - 100) / (180 - 100), 1), 0))
                bp_dia = float(max(min((bp_dia - 55) / (100 - 55), 1), 0))
            if len(chunk_ori) == 5:
                temp_chunk = chunk_ori
                chunk, pos = self._concat_chunks(temp_chunk)
                chunk[pos:] = np.mean(chunk[:pos])
                chunk_filt = filter_ppg_signal(chunk, self.LPF, self.HPF, self.FS)
                chunk_filt_norm = normalize_min_max(chunk_filt)
                
                if self.derivative:
                    chunk_filt_norm_fd = np.diff(chunk_filt_norm)
                    chunk_filt_norm_fd = np.insert(chunk_filt_norm_fd, 0, 0)
                    chunk_filt_norm_sd = np.diff(chunk_filt_norm_fd)
                    chunk_filt_norm_sd = np.insert(chunk_filt_norm_sd, 0, 0)
                    item = {
                        "ppg_chunk": np.vstack((np.reshape(chunk_filt_norm, (1, len(chunk_filt_norm))),
                                    np.reshape(chunk_filt_norm_fd, (1, len(chunk_filt_norm_fd))),
                                    np.reshape(chunk_filt_norm_sd, (1, len(chunk_filt_norm_sd))))),
                        "bp": np.reshape(np.array([bp_sys, bp_dia]), (1, 2)),
                        "session": session_id,
                        "age": np.reshape(age_array, self.demo_shape),
                        "bmi": np.reshape(bmi_array, self.demo_shape),
                        "gender": np.reshape(gender_array, self.demo_shape)
                    }
                else:
                    item = {
                        "ppg_chunk": np.reshape(chunk_filt_norm, (1, len(chunk_filt_norm))),
                        "bp": np.reshape(np.array([bp_sys, bp_dia]), (1, 2)),
                        "session": session_id,
                        "age": np.reshape(age_array, self.demo_shape),
                        "bmi": np.reshape(bmi_array, self.demo_shape),
                        "gender": np.reshape(gender_array, self.demo_shape)
                    }
                
                data.append(item)

            else:

                for j in range(len(chunk_ori) - self.chunk_amount):
                    temp_chunk = chunk_ori[j : j + self.chunk_amount]
                    chunk, pos = self._concat_chunks(temp_chunk)
                    chunk[pos:] = np.mean(chunk[:pos])
                    chunk_filt = filter_ppg_signal(chunk, self.LPF, self.HPF, self.FS)
                    chunk_filt_norm = normalize_min_max(chunk_filt)

                    if self.derivative:
                        chunk_filt_norm_fd = np.diff(chunk_filt_norm)
                        chunk_filt_norm_fd = np.insert(chunk_filt_norm_fd, 0, 0)
                        chunk_filt_norm_sd = np.diff(chunk_filt_norm_fd)
                        chunk_filt_norm_sd = np.insert(chunk_filt_norm_sd, 0, 0)
                        item = {
                            "ppg_chunk": np.vstack((np.reshape(chunk_filt_norm, (1, len(chunk_filt_norm))),
                                        np.reshape(chunk_filt_norm_fd, (1, len(chunk_filt_norm_fd))),
                                        np.reshape(chunk_filt_norm_sd, (1, len(chunk_filt_norm_sd))))),
                            "bp": np.reshape(np.array([bp_sys, bp_dia]), (1, 2)),
                            "session": session_id,
                            "age": np.reshape(age_array, self.demo_shape),
                            "bmi": np.reshape(bmi_array, self.demo_shape),
                            "gender": np.reshape(gender_array, self.demo_shape)
                        }
                    else:
                        item = {
                            "ppg_chunk": np.reshape(chunk_filt_norm, (1, len(chunk_filt_norm))),
                            "bp": np.reshape(np.array([bp_sys, bp_dia]), (1, 2)),
                            "session": session_id,
                            "age": np.reshape(age_array, self.demo_shape),
                            "bmi": np.reshape(bmi_array, self.demo_shape),
                            "gender": np.reshape(gender_array, self.demo_shape)
                        }
                    
                    data.append(item)

        return data


class UWBP_val_whole_demo_arry(UWBP_train_whole_demo_arry):

    def __init__(self, config):
        super().__init__(config)
        self.data_list_path = config["val_dataset_path"]
        self.data_list = self._load_data_list(self.data_list_path)
    
    def __getitem__(self, index):
        mat_path = self.data_list[index]
        session_id = mat_path.split(".")[0][-5:]
        mat_data = np.load(mat_path, allow_pickle=True)
        data = list()
        for i in range(len(mat_data) - self.chunk_amount):
            chunk_ori = mat_data[i : i + self.chunk_amount]
            mat_pid = mat_path.split("/")[-1].split(".")[0][:-1]
            session = mat_path.split("/")[-1].split(".")[0][-1]
            df = self.bp_gt[self.bp_gt["PID"] == mat_pid]

            age = (df["age"].values[0] - 34) / (96 - 34) # Now the min age is 34 years old and oldest person is 96
            bmi = (df["bmi"].values[0] - 17) / (41 - 17) # Now the min bmi is 17.327 and the max bmi is 40.88
            gender = df["gender"].values[0]
            if gender == "M":
                gender = 0
            else:
                gender = 1
            
            age_array = np.zeros(self.demo_shape, dtype=np.float32)
            bmi_array = np.zeros(self.demo_shape, dtype=np.float32)
            gender_array = np.zeros(self.demo_shape, dtype=np.float32)
            feature_array = np.zeros(self.feature_shape, dtype=np.float32)
            age_array[:] = age
            bmi_array[:] = bmi
            gender_array[:] = gender

            tmp_feature_label_df = self.feature_label[self.feature_label["ID"] == mat_pid]
            tmp_feature_label_df = tmp_feature_label_df.drop(["ID", "Age", "BMI"], axis=1)
            feature_array[:] = tmp_feature_label_df.values[0]

            if session == "a":
                bp_sys = (df.iloc[0, 1] + df.iloc[0, 2]) / 2
                bp_dia = (df.iloc[0, 5] + df.iloc[0, 6]) / 2
                bp_sys = float(max(min((bp_sys - 100) / (180 - 100), 1), 0))
                bp_dia = float(max(min((bp_dia - 55) / (100 - 55), 1), 0))

            else:
                bp_sys = (df.iloc[0, 2] + df.iloc[0, 3]) / 2
                bp_dia = (df.iloc[0, 6] + df.iloc[0, 7]) / 2
                bp_sys = float(max(min((bp_sys - 100) / (180 - 100), 1), 0))
                bp_dia = float(max(min((bp_dia - 55) / (100 - 55), 1), 0))

            if len(chunk_ori) == self.chunk_amount:
                temp_chunk = chunk_ori
                chunk, pos = self._concat_chunks(temp_chunk)
                chunk[pos:] = np.mean(chunk[:pos])
                chunk_filt = filter_ppg_signal(chunk, self.LPF, self.HPF, self.FS)
                chunk_filt_norm = normalize_min_max(chunk_filt)
                
                if self.derivative:
                    chunk_filt_norm_fd = np.diff(chunk_filt_norm)
                    chunk_filt_norm_fd = np.insert(chunk_filt_norm_fd, 0, 0)
                    chunk_filt_norm_sd = np.diff(chunk_filt_norm_fd)
                    chunk_filt_norm_sd = np.insert(chunk_filt_norm_sd, 0, 0)
                    item = {
                        "ppg_chunk": np.vstack((np.reshape(chunk_filt_norm, (1, len(chunk_filt_norm))),
                                    np.reshape(chunk_filt_norm_fd, (1, len(chunk_filt_norm_fd))),
                                    np.reshape(chunk_filt_norm_sd, (1, len(chunk_filt_norm_sd))))),
                        "bp": np.reshape(np.array([bp_sys, bp_dia]), (1, 2)),
                        "session": session_id,
                        "age": np.reshape(age_array, self.demo_shape),
                        "bmi": np.reshape(bmi_array, self.demo_shape),
                        "gender": np.reshape(gender_array, self.demo_shape)
                    }
                else:
                    item = {
                        "ppg_chunk": np.reshape(chunk_filt_norm, (1, len(chunk_filt_norm))),
                        "bp": np.reshape(np.array([bp_sys, bp_dia]), (1, 2)),
                        # "bp": np.reshape(np.array([bp_sys]), (1, 1)),
                        "age": np.reshape(age_array, self.demo_shape),
                        "bmi": np.reshape(bmi_array, self.demo_shape),
                        "feature": np.reshape(feature_array, self.feature_shape),
                    }
                
                data.append(item)

            else:

                for j in range(len(chunk_ori) - self.chunk_amount):
                    temp_chunk = chunk_ori[j : j + self.chunk_amount]
                    chunk, pos = self._concat_chunks(temp_chunk)
                    chunk[pos:] = np.mean(chunk[:pos])
                    chunk_filt = filter_ppg_signal(chunk, self.LPF, self.HPF, self.FS)
                    chunk_filt_norm = normalize_min_max(chunk_filt)
                    
                    if self.derivative:
                        chunk_filt_norm_fd = np.diff(chunk_filt_norm)
                        chunk_filt_norm_fd = np.insert(chunk_filt_norm_fd, 0, 0)
                        chunk_filt_norm_sd = np.diff(chunk_filt_norm_fd)
                        chunk_filt_norm_sd = np.insert(chunk_filt_norm_sd, 0, 0)
                        item = {
                            "ppg_chunk": np.vstack((np.reshape(chunk_filt_norm, (1, len(chunk_filt_norm))),
                                        np.reshape(chunk_filt_norm_fd, (1, len(chunk_filt_norm_fd))),
                                        np.reshape(chunk_filt_norm_sd, (1, len(chunk_filt_norm_sd))))),
                            "bp": np.reshape(np.array([bp_sys, bp_dia]), (1, 2)),
                            "session": session_id,
                            "age": np.reshape(age_array, self.demo_shape),
                            "bmi": np.reshape(bmi_array, self.demo_shape),
                            "gender": np.reshape(gender_array, self.demo_shape)
                        }
                    else:
                        item = {
                            "ppg_chunk": np.reshape(chunk_filt_norm, (1, len(chunk_filt_norm))),
                            "bp": np.reshape(np.array([bp_sys, bp_dia]), (1, 2)),
                            # "bp": np.reshape(np.array([bp_sys]), (1, 1)),
                            "age": np.reshape(age_array, self.demo_shape),
                            "bmi": np.reshape(bmi_array, self.demo_shape),
                            "feature": np.reshape(feature_array, self.feature_shape),
                        }
                    
                    data.append(item)

        return data


class UWBP_val_demo_whole_cls(UWBP_train_demo_whole_cls):

    def __init__(self, config):
        super().__init__(config)
        self.data_list_path = config["val_dataset_path"]
        self.data_list = self._load_data_list(self.data_list_path)
    
    def __getitem__(self, index):
        mat_path = self.data_list[index]
        session_id = mat_path.split(".")[0][-5:]
        mat_data = np.load(mat_path, allow_pickle=True)
        data = list()
        for i in range(len(mat_data) - self.chunk_amount):
            chunk_ori = mat_data[i : i + self.chunk_amount]
            mat_pid = mat_path.split("/")[-1].split(".")[0][:-1]
            session = mat_path.split("/")[-1].split(".")[0][-1]
            df = self.bp_gt[self.bp_gt["PID"] == mat_pid]

            age = (df["age"].values[0] - 34) / (96 - 34) # Now the min age is 34 years old and oldest person is 96
            bmi = (df["bmi"].values[0] - 17) / (41 - 17) # Now the min bmi is 17.327 and the max bmi is 40.88
            gender = df["gender"].values[0]
            if gender == "M":
                gender = 0
            else:
                gender = 1
            
            age_array = np.zeros(self.demo_shape, dtype=np.float32)
            bmi_array = np.zeros(self.demo_shape, dtype=np.float32)
            gender_array = np.zeros(self.demo_shape, dtype=np.float32)
            feature_array = np.zeros(self.feature_shape, dtype=np.float32)
            age_array[:] = age
            bmi_array[:] = bmi
            gender_array[:] = gender

            tmp_feature_label_df = self.feature_label[self.feature_label["ID"] == mat_pid]
            tmp_feature_label_df = tmp_feature_label_df.drop(["ID", "Age", "BMI"], axis=1)
            feature_array[:] = tmp_feature_label_df.values[0]

            if session == "a":
                bp_sys = (df.iloc[0, 1] + df.iloc[0, 2]) / 2
                bp_dia = (df.iloc[0, 5] + df.iloc[0, 6]) / 2

            else:
                bp_sys = (df.iloc[0, 2] + df.iloc[0, 3]) / 2
                bp_dia = (df.iloc[0, 6] + df.iloc[0, 7]) / 2

            if bp_sys >= 130:
                bp_sys = 1
            else:
                bp_sys = 0

            if len(chunk_ori) == self.chunk_amount:
                temp_chunk = chunk_ori
                chunk, pos = self._concat_chunks(temp_chunk)
                chunk[pos:] = np.mean(chunk[:pos])
                chunk_filt = filter_ppg_signal(chunk, self.LPF, self.HPF, self.FS)
                chunk_filt_norm = normalize_min_max(chunk_filt)
                
                if self.derivative:
                    chunk_filt_norm_fd = np.diff(chunk_filt_norm)
                    chunk_filt_norm_fd = np.insert(chunk_filt_norm_fd, 0, 0)
                    chunk_filt_norm_sd = np.diff(chunk_filt_norm_fd)
                    chunk_filt_norm_sd = np.insert(chunk_filt_norm_sd, 0, 0)
                    item = {
                        "ppg_chunk": np.vstack((np.reshape(chunk_filt_norm, (1, len(chunk_filt_norm))),
                                    np.reshape(chunk_filt_norm_fd, (1, len(chunk_filt_norm_fd))),
                                    np.reshape(chunk_filt_norm_sd, (1, len(chunk_filt_norm_sd))))),
                        "bp": np.reshape(np.array([bp_sys, bp_dia]), (1, 2)),
                        "session": session_id,
                        "age": np.reshape(age_array, self.demo_shape),
                        "bmi": np.reshape(bmi_array, self.demo_shape),
                        "gender": np.reshape(gender_array, self.demo_shape)
                    }
                else:
                    item = {
                        "ppg_chunk": np.reshape(chunk_filt_norm, (1, len(chunk_filt_norm))),
                        # "bp": np.reshape(np.array([bp_sys, bp_dia]), (1, 2)),
                        "bp": np.reshape(np.array([bp_sys]), (1, 1)),
                        "age": np.reshape(age_array, self.demo_shape),
                        "bmi": np.reshape(bmi_array, self.demo_shape),
                        "feature": np.reshape(feature_array, self.feature_shape),
                    }
                
                data.append(item)

            else:

                for j in range(len(chunk_ori) - self.chunk_amount):
                    temp_chunk = chunk_ori[j : j + self.chunk_amount]
                    chunk, pos = self._concat_chunks(temp_chunk)
                    chunk[pos:] = np.mean(chunk[:pos])
                    chunk_filt = filter_ppg_signal(chunk, self.LPF, self.HPF, self.FS)
                    chunk_filt_norm = normalize_min_max(chunk_filt)
                    
                    if self.derivative:
                        chunk_filt_norm_fd = np.diff(chunk_filt_norm)
                        chunk_filt_norm_fd = np.insert(chunk_filt_norm_fd, 0, 0)
                        chunk_filt_norm_sd = np.diff(chunk_filt_norm_fd)
                        chunk_filt_norm_sd = np.insert(chunk_filt_norm_sd, 0, 0)
                        item = {
                            "ppg_chunk": np.vstack((np.reshape(chunk_filt_norm, (1, len(chunk_filt_norm))),
                                        np.reshape(chunk_filt_norm_fd, (1, len(chunk_filt_norm_fd))),
                                        np.reshape(chunk_filt_norm_sd, (1, len(chunk_filt_norm_sd))))),
                            "bp": np.reshape(np.array([bp_sys, bp_dia]), (1, 2)),
                            "session": session_id,
                            "age": np.reshape(age_array, self.demo_shape),
                            "bmi": np.reshape(bmi_array, self.demo_shape),
                            "gender": np.reshape(gender_array, self.demo_shape)
                        }
                    else:
                        item = {
                            "ppg_chunk": np.reshape(chunk_filt_norm, (1, len(chunk_filt_norm))),
                            # "bp": np.reshape(np.array([bp_sys, bp_dia]), (1, 2)),
                            "bp": np.reshape(np.array([bp_sys]), (1, 1)),
                            "age": np.reshape(age_array, self.demo_shape),
                            "bmi": np.reshape(bmi_array, self.demo_shape),
                            "feature": np.reshape(feature_array, self.feature_shape),
                        }
                    
                    data.append(item)

        return data
    

class UWBP_test_manual_demo(UWBP_train_manual_demo):

    def __init__(self, config, fold):
        super().__init__(config)
        if fold == 1:
            self.data_list_path = config["val_dataset_path1"]
        elif fold == 2:
            self.data_list_path = config["val_dataset_path2"]
        elif fold == 3:
            self.data_list_path = config["val_dataset_path3"]
        elif fold == 4:
            self.data_list_path = config["val_dataset_path4"]
        else:
            self.data_list_path = config["val_dataset_path5"]

        self.data_list = self._load_data_list(self.data_list_path)
    
    def __getitem__(self, index):
        mat_path = self.data_list[index]
        session_id = mat_path.split(".")[0][-5:]
        mat_data = np.load(mat_path, allow_pickle=True)
        # chunk_pos = random.randint(0, len(mat_data) - 1)
        data = list()
        for i in range(len(mat_data)):
            chunk_ori = mat_data[i]
            mat_pid = mat_path.split("/")[-1].split(".")[0][:-1]
            session = mat_path.split("/")[-1].split(".")[0][-1]
            df = self.bp_gt[self.bp_gt["PID"] == mat_pid]

            age = (df["age"].values[0] - 34) / (96 - 34) # Now the min age is 34 years old and oldest person is 96
            bmi = (df["bmi"].values[0] - 17) / (41 - 17) # Now the min bmi is 17.327 and the max bmi is 40.88
            gender = df["gender"].values[0]
            if gender == "M":
                gender = 0
            else:
                gender = 1
            
            age_array = np.zeros(self.demo_shape, dtype=np.float32)
            bmi_array = np.zeros(self.demo_shape, dtype=np.float32)
            gender_array = np.zeros(self.demo_shape, dtype=np.float32)
            age_array[:] = age
            bmi_array[:] = bmi
            gender_array[:] = gender

            if session == "a":
                bp_sys = (df.iloc[0, 1] + df.iloc[0, 2]) / 2
                bp_dia = (df.iloc[0, 5] + df.iloc[0, 6]) / 2
                bp_sys = float(max(min((bp_sys - 100) / (180 - 100), 1), 0))
                bp_dia = float(max(min((bp_dia - 55) / (100 - 55), 1), 0))
            else:
                bp_sys = (df.iloc[0, 2] + df.iloc[0, 3]) / 2
                bp_dia = (df.iloc[0, 6] + df.iloc[0, 7]) / 2
                bp_sys = float(max(min((bp_sys - 100) / (180 - 100), 1), 0))
                bp_dia = float(max(min((bp_dia - 55) / (100 - 55), 1), 0))
        
            if len(chunk_ori) == 5:
                temp_chunk = chunk_ori
                chunk, pos = self._concat_chunks(temp_chunk)
                chunk[pos:] = np.mean(chunk[:pos])
                chunk_filt = filter_ppg_signal(chunk, self.LPF, self.HPF, self.FS)
                chunk_filt_norm = normalize_min_max(chunk_filt)
                
                if self.derivative:
                    chunk_filt_norm_fd = np.diff(chunk_filt_norm)
                    chunk_filt_norm_fd = np.insert(chunk_filt_norm_fd, 0, 0)
                    chunk_filt_norm_sd = np.diff(chunk_filt_norm_fd)
                    chunk_filt_norm_sd = np.insert(chunk_filt_norm_sd, 0, 0)
                    item = {
                        "ppg_chunk": np.vstack((np.reshape(chunk_filt_norm, (1, len(chunk_filt_norm))),
                                    np.reshape(chunk_filt_norm_fd, (1, len(chunk_filt_norm_fd))),
                                    np.reshape(chunk_filt_norm_sd, (1, len(chunk_filt_norm_sd))))),
                        "bp": np.reshape(np.array([bp_sys, bp_dia]), (1, 2)),
                        "session": session_id,
                        "age": np.reshape(age_array, self.demo_shape),
                        "bmi": np.reshape(bmi_array, self.demo_shape),
                        "gender": np.reshape(gender_array, self.demo_shape)
                    }
                else:
                    item = {
                        "ppg_chunk": np.reshape(chunk_filt_norm, (1, len(chunk_filt_norm))),
                        "bp": np.reshape(np.array([bp_sys, bp_dia]), (1, 2)),
                        "session": session_id,
                        "age": np.reshape(age_array, self.demo_shape),
                        "bmi": np.reshape(bmi_array, self.demo_shape),
                        "gender": np.reshape(gender_array, self.demo_shape)
                    }
                
                data.append(item)

            else:

                for j in range(len(chunk_ori) - self.chunk_amount):
                    temp_chunk = chunk_ori[j : j + self.chunk_amount]
                    chunk, pos = self._concat_chunks(temp_chunk)
                    chunk[pos:] = np.mean(chunk[:pos])
                    chunk_filt = filter_ppg_signal(chunk, self.LPF, self.HPF, self.FS)
                    chunk_filt_norm = normalize_min_max(chunk_filt)
                    
                    if self.derivative:
                        chunk_filt_norm_fd = np.diff(chunk_filt_norm)
                        chunk_filt_norm_fd = np.insert(chunk_filt_norm_fd, 0, 0)
                        chunk_filt_norm_sd = np.diff(chunk_filt_norm_fd)
                        chunk_filt_norm_sd = np.insert(chunk_filt_norm_sd, 0, 0)
                        item = {
                            "ppg_chunk": np.vstack((np.reshape(chunk_filt_norm, (1, len(chunk_filt_norm))),
                                        np.reshape(chunk_filt_norm_fd, (1, len(chunk_filt_norm_fd))),
                                        np.reshape(chunk_filt_norm_sd, (1, len(chunk_filt_norm_sd))))),
                            "bp": np.reshape(np.array([bp_sys, bp_dia]), (1, 2)),
                            "session": session_id,
                            "age": np.reshape(age_array, self.demo_shape),
                            "bmi": np.reshape(bmi_array, self.demo_shape),
                            "gender": np.reshape(gender_array, self.demo_shape)
                        }
                    else:
                        item = {
                            "ppg_chunk": np.reshape(chunk_filt_norm, (1, len(chunk_filt_norm))),
                            "bp": np.reshape(np.array([bp_sys, bp_dia]), (1, 2)),
                            "session": session_id,
                            "age": np.reshape(age_array, self.demo_shape),
                            "bmi": np.reshape(bmi_array, self.demo_shape),
                            "gender": np.reshape(gender_array, self.demo_shape)
                        }
                    
                    data.append(item)

        return data


class UWBP_test_whole_demo_arry(UWBP_train_whole_demo_arry):

    def __init__(self, config, fold):
        super().__init__(config)
        if fold == 1:
            self.data_list_path = config["val_dataset_path1"]
        elif fold == 2:
            self.data_list_path = config["val_dataset_path2"]
        elif fold == 3:
            self.data_list_path = config["val_dataset_path3"]
        elif fold == 4:
            self.data_list_path = config["val_dataset_path4"]
        else:
            self.data_list_path = config["val_dataset_path5"]

        self.data_list = self._load_data_list(self.data_list_path)
    
    def __getitem__(self, index):
        mat_path = self.data_list[index]
        session_id = mat_path.split(".")[0][-5:]
        mat_data = np.load(mat_path, allow_pickle=True)
        data = list()
        for i in range(len(mat_data) - self.chunk_amount):
            chunk_ori = mat_data[i : i + self.chunk_amount]
            mat_pid = mat_path.split("/")[-1].split(".")[0][:-1]
            session = mat_path.split("/")[-1].split(".")[0][-1]
            df = self.bp_gt[self.bp_gt["PID"] == mat_pid]

            age = (df["age"].values[0] - 34) / (96 - 34) # Now the min age is 34 years old and oldest person is 96
            bmi = (df["bmi"].values[0] - 17) / (41 - 17) # Now the min bmi is 17.327 and the max bmi is 40.88
            gender = df["gender"].values[0]
            if gender == "M":
                gender = 0
            else:
                gender = 1
            
            age_array = np.zeros(self.demo_shape, dtype=np.float32)
            bmi_array = np.zeros(self.demo_shape, dtype=np.float32)
            gender_array = np.zeros(self.demo_shape, dtype=np.float32)
            feature_array = np.zeros(self.feature_shape, dtype=np.float32)
            age_array[:] = age
            bmi_array[:] = bmi
            gender_array[:] = gender

            tmp_feature_label_df = self.feature_label[self.feature_label["ID"] == mat_pid]
            tmp_feature_label_df = tmp_feature_label_df.drop(["ID", "Age", "BMI"], axis=1)
            feature_array[:] = tmp_feature_label_df.values[0]

            if session == "a":
                bp_sys = (df.iloc[0, 1] + df.iloc[0, 2]) / 2
                bp_dia = (df.iloc[0, 5] + df.iloc[0, 6]) / 2
                bp_sys = float(max(min((bp_sys - 100) / (180 - 100), 1), 0))
                bp_dia = float(max(min((bp_dia - 55) / (100 - 55), 1), 0))
            else:
                bp_sys = (df.iloc[0, 2] + df.iloc[0, 3]) / 2
                bp_dia = (df.iloc[0, 6] + df.iloc[0, 7]) / 2
                bp_sys = float(max(min((bp_sys - 100) / (180 - 100), 1), 0))
                bp_dia = float(max(min((bp_dia - 55) / (100 - 55), 1), 0))

            if len(chunk_ori) == self.chunk_amount:
                temp_chunk = chunk_ori
                chunk, pos = self._concat_chunks(temp_chunk)
                chunk[pos:] = np.mean(chunk[:pos])
                chunk_filt = filter_ppg_signal(chunk, self.LPF, self.HPF, self.FS)
                chunk_filt_norm = normalize_min_max(chunk_filt)
                
                if self.derivative:
                    chunk_filt_norm_fd = np.diff(chunk_filt_norm)
                    chunk_filt_norm_fd = np.insert(chunk_filt_norm_fd, 0, 0)
                    chunk_filt_norm_sd = np.diff(chunk_filt_norm_fd)
                    chunk_filt_norm_sd = np.insert(chunk_filt_norm_sd, 0, 0)
                    item = {
                        "ppg_chunk": np.vstack((np.reshape(chunk_filt_norm, (1, len(chunk_filt_norm))),
                                    np.reshape(chunk_filt_norm_fd, (1, len(chunk_filt_norm_fd))),
                                    np.reshape(chunk_filt_norm_sd, (1, len(chunk_filt_norm_sd))))),
                        "bp": np.reshape(np.array([bp_sys, bp_dia]), (1, 2)),
                        "session": session_id,
                        "age": np.reshape(age_array, self.demo_shape),
                        "bmi": np.reshape(bmi_array, self.demo_shape),
                        "gender": np.reshape(gender_array, self.demo_shape)
                    }
                else:
                    item = {
                        "ppg_chunk": np.reshape(chunk_filt_norm, (1, len(chunk_filt_norm))),
                        "bp": np.reshape(np.array([bp_sys, bp_dia]), (1, 2)),
                        # "bp": np.reshape(np.array([bp_sys]), (1, 1)),
                        "session": session_id,
                        "age": np.reshape(age_array, self.demo_shape),
                        "bmi": np.reshape(bmi_array, self.demo_shape),
                        "feature": np.reshape(feature_array, self.feature_shape),
                    }
                
                data.append(item)

            else:

                for j in range(len(chunk_ori) - self.chunk_amount):
                    temp_chunk = chunk_ori[j : j + self.chunk_amount]
                    chunk, pos = self._concat_chunks(temp_chunk)
                    chunk[pos:] = np.mean(chunk[:pos])
                    chunk_filt = filter_ppg_signal(chunk, self.LPF, self.HPF, self.FS)
                    chunk_filt_norm = normalize_min_max(chunk_filt)
                    
                    if self.derivative:
                        chunk_filt_norm_fd = np.diff(chunk_filt_norm)
                        chunk_filt_norm_fd = np.insert(chunk_filt_norm_fd, 0, 0)
                        chunk_filt_norm_sd = np.diff(chunk_filt_norm_fd)
                        chunk_filt_norm_sd = np.insert(chunk_filt_norm_sd, 0, 0)
                        item = {
                            "ppg_chunk": np.vstack((np.reshape(chunk_filt_norm, (1, len(chunk_filt_norm))),
                                        np.reshape(chunk_filt_norm_fd, (1, len(chunk_filt_norm_fd))),
                                        np.reshape(chunk_filt_norm_sd, (1, len(chunk_filt_norm_sd))))),
                            "bp": np.reshape(np.array([bp_sys, bp_dia]), (1, 2)),
                            "session": session_id,
                            "age": np.reshape(age_array, self.demo_shape),
                            "bmi": np.reshape(bmi_array, self.demo_shape),
                            "gender": np.reshape(gender_array, self.demo_shape)
                        }
                    else:
                        item = {
                            "ppg_chunk": np.reshape(chunk_filt_norm, (1, len(chunk_filt_norm))),
                            "bp": np.reshape(np.array([bp_sys, bp_dia]), (1, 2)),
                            # "bp": np.reshape(np.array([bp_sys]), (1, 1)),
                            "session": session_id,
                            "age": np.reshape(age_array, self.demo_shape),
                            "bmi": np.reshape(bmi_array, self.demo_shape),
                            "feature": np.reshape(feature_array, self.feature_shape),
                        }
                    
                    data.append(item)

        return data

class UWBP_test_demo_whole_cls(UWBP_train_demo_whole_cls):

    def __init__(self, config, fold):
        super().__init__(config)
        if fold == 1:
            self.data_list_path = config["val_dataset_path1"]
        elif fold == 2:
            self.data_list_path = config["val_dataset_path2"]
        elif fold == 3:
            self.data_list_path = config["val_dataset_path3"]
        elif fold == 4:
            self.data_list_path = config["val_dataset_path4"]
        else:
            self.data_list_path = config["val_dataset_path5"]

        self.data_list = self._load_data_list(self.data_list_path)
    
    def __getitem__(self, index):
        mat_path = self.data_list[index]
        session_id = mat_path.split(".")[0][-5:]
        mat_data = np.load(mat_path, allow_pickle=True)
        # chunk_pos = random.randint(0, len(mat_data) - 1)
        data = list()
        for i in range(len(mat_data) - self.chunk_amount):
            # chunk_ori = mat_data[i]
            chunk_ori = mat_data[i : i + self.chunk_amount]
            mat_pid = mat_path.split("/")[-1].split(".")[0][:-1]
            session = mat_path.split("/")[-1].split(".")[0][-1]
            df = self.bp_gt[self.bp_gt["PID"] == mat_pid]

            age = (df["age"].values[0] - 34) / (96 - 34) # Now the min age is 34 years old and oldest person is 96
            bmi = (df["bmi"].values[0] - 17) / (41 - 17) # Now the min bmi is 17.327 and the max bmi is 40.88
            gender = df["gender"].values[0]
            if gender == "M":
                gender = 0
            else:
                gender = 1
            
            age_array = np.zeros(self.demo_shape, dtype=np.float32)
            bmi_array = np.zeros(self.demo_shape, dtype=np.float32)
            gender_array = np.zeros(self.demo_shape, dtype=np.float32)
            feature_array = np.zeros(self.feature_shape, dtype=np.float32)
            age_array[:] = age
            bmi_array[:] = bmi
            gender_array[:] = gender

            tmp_feature_label_df = self.feature_label[self.feature_label["ID"] == mat_pid]
            tmp_feature_label_df = tmp_feature_label_df.drop(["ID", "Age", "BMI"], axis=1)
            feature_array[:] = tmp_feature_label_df.values[0]

            if session == "a":
                bp_sys = (df.iloc[0, 1] + df.iloc[0, 2]) / 2
                bp_dia = (df.iloc[0, 5] + df.iloc[0, 6]) / 2

            else:
                bp_sys = (df.iloc[0, 2] + df.iloc[0, 3]) / 2
                bp_dia = (df.iloc[0, 6] + df.iloc[0, 7]) / 2
            
            if bp_sys >= 130:
                bp_sys = 1
            else:
                bp_sys = 0

            if len(chunk_ori) == self.chunk_amount:
                temp_chunk = chunk_ori
                chunk, pos = self._concat_chunks(temp_chunk)
                chunk[pos:] = np.mean(chunk[:pos])
                chunk_filt = filter_ppg_signal(chunk, self.LPF, self.HPF, self.FS)
                chunk_filt_norm = normalize_min_max(chunk_filt)
                
                if self.derivative:
                    chunk_filt_norm_fd = np.diff(chunk_filt_norm)
                    chunk_filt_norm_fd = np.insert(chunk_filt_norm_fd, 0, 0)
                    chunk_filt_norm_sd = np.diff(chunk_filt_norm_fd)
                    chunk_filt_norm_sd = np.insert(chunk_filt_norm_sd, 0, 0)
                    item = {
                        "ppg_chunk": np.vstack((np.reshape(chunk_filt_norm, (1, len(chunk_filt_norm))),
                                    np.reshape(chunk_filt_norm_fd, (1, len(chunk_filt_norm_fd))),
                                    np.reshape(chunk_filt_norm_sd, (1, len(chunk_filt_norm_sd))))),
                        "bp": np.reshape(np.array([bp_sys, bp_dia]), (1, 2)),
                        "session": session_id,
                        "age": np.reshape(age_array, self.demo_shape),
                        "bmi": np.reshape(bmi_array, self.demo_shape),
                        "gender": np.reshape(gender_array, self.demo_shape)
                    }
                else:
                    item = {
                        "ppg_chunk": np.reshape(chunk_filt_norm, (1, len(chunk_filt_norm))),
                        # "bp": np.reshape(np.array([bp_sys, bp_dia]), (1, 2)),
                        "session": session_id,
                        "bp": np.reshape(np.array([bp_sys]), (1, 1)),
                        "age": np.reshape(age_array, self.demo_shape),
                        "bmi": np.reshape(bmi_array, self.demo_shape),
                        "feature": np.reshape(feature_array, self.feature_shape),
                    }
                
                data.append(item)

            else:

                for j in range(len(chunk_ori) - self.chunk_amount):
                    temp_chunk = chunk_ori[j : j + self.chunk_amount]
                    chunk, pos = self._concat_chunks(temp_chunk)
                    chunk[pos:] = np.mean(chunk[:pos])
                    chunk_filt = filter_ppg_signal(chunk, self.LPF, self.HPF, self.FS)
                    chunk_filt_norm = normalize_min_max(chunk_filt)
                    
                    if self.derivative:
                        chunk_filt_norm_fd = np.diff(chunk_filt_norm)
                        chunk_filt_norm_fd = np.insert(chunk_filt_norm_fd, 0, 0)
                        chunk_filt_norm_sd = np.diff(chunk_filt_norm_fd)
                        chunk_filt_norm_sd = np.insert(chunk_filt_norm_sd, 0, 0)
                        item = {
                            "ppg_chunk": np.vstack((np.reshape(chunk_filt_norm, (1, len(chunk_filt_norm))),
                                        np.reshape(chunk_filt_norm_fd, (1, len(chunk_filt_norm_fd))),
                                        np.reshape(chunk_filt_norm_sd, (1, len(chunk_filt_norm_sd))))),
                            "bp": np.reshape(np.array([bp_sys, bp_dia]), (1, 2)),
                            "session": session_id,
                            "age": np.reshape(age_array, self.demo_shape),
                            "bmi": np.reshape(bmi_array, self.demo_shape),
                            "gender": np.reshape(gender_array, self.demo_shape)
                        }
                    else:
                        item = {
                            "ppg_chunk": np.reshape(chunk_filt_norm, (1, len(chunk_filt_norm))),
                            # "bp": np.reshape(np.array([bp_sys, bp_dia]), (1, 2)),
                            "session": session_id,
                            "bp": np.reshape(np.array([bp_sys]), (1, 1)),
                            "age": np.reshape(age_array, self.demo_shape),
                            "bmi": np.reshape(bmi_array, self.demo_shape),
                            "feature": np.reshape(feature_array, self.feature_shape),
                        }
                    
                    data.append(item)

        return data
