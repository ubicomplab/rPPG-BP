import os
import random
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from preprocess import normalize_min_max, filter_ppg_signal

class UWBP_train(Dataset):

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
        chunk_pos = random.randint(0, len(mat_data) - self.chunk_amount - 1)
        chunks = mat_data[chunk_pos : chunk_pos + self.chunk_amount]
        chunk, pos = self._concat_chunks(chunks)
        chunk[pos:] = np.mean(chunk[:pos])
        chunk_filt = filter_ppg_signal(chunk, self.LPF, self.HPF, self.FS)
        chunk_filt_norm = normalize_min_max(chunk_filt)
        if self.frequency_field:
            chunk_filt_norm = np.real(np.fft.rfft(chunk_filt_norm))

        mat_pid = mat_path.split("/")[-1].split(".")[0][:-1]
        df = self.bp_gt[self.bp_gt["PID"] == mat_pid]
        bp_sys = max(min((df.iloc[0, 4] - 100) / (180 - 100), 1), 0)
        bp_dia = max(min((df.iloc[0, 8] - 55) / (100 - 55), 1), 0)

        if self.derivative:
            chunk_filt_norm_fd = np.diff(chunk_filt_norm)
            chunk_filt_norm_fd = np.insert(chunk_filt_norm_fd, 0, 0)
            chunk_filt_norm_sd = np.diff(chunk_filt_norm_fd)
            chunk_filt_norm_sd = np.insert(chunk_filt_norm_sd, 0, 0)
            item = {
                "ppg_chunk": np.vstack((np.reshape(chunk_filt_norm, (1, len(chunk_filt_norm))),
                            np.reshape(chunk_filt_norm_fd, (1, len(chunk_filt_norm_fd))),
                            np.reshape(chunk_filt_norm_sd, (1, len(chunk_filt_norm_sd))))),
                "bp": np.reshape(np.array([bp_sys, bp_dia]), (1, 2))
            }
        else:
            item = {
                "ppg_chunk": np.reshape(chunk_filt_norm, (1, len(chunk_filt_norm))),
                "bp": np.reshape(np.array([bp_sys, bp_dia]), (1, 2))
            }

        return item



class UWBP_val(UWBP_train):

    def __init__(self, config):
        super().__init__(config)
        self.data_list_path = config["val_dataset_path"]
        self.data_list = self._load_data_list(self.data_list_path)


class UWBP_test(UWBP_train):

    def __init__(self, config):
        super().__init__(config)
        self.data_list_path = config["val_dataset_path"]
        self.data_list = self._load_data_list(self.data_list_path)
    
    def _concat_chunks(self, chunks):
        chunk = np.zeros((self.chunk_length))
        i = 0
        for ppg_chunk in chunks:
            chunk[i : i + len(ppg_chunk)] = np.squeeze(ppg_chunk)
            i = i + len(ppg_chunk)
        return chunk, i
    
    def __getitem__(self, index):
        mat_path = self.data_list[index]
        mat_data = np.load(mat_path, allow_pickle=True)
        
        # chunk_pos = random.randint(0, len(mat_data) - self.chunk_amount - 1)
        # chunks = mat_data[chunk_pos : chunk_pos + self.chunk_amount]
        chunk, pos = self._concat_chunks(mat_data)
        chunk[pos:] = np.mean(chunk[:pos])
        chunk_filt = filter_ppg_signal(chunk, self.LPF, self.HPF, self.FS)
        chunk_filt_norm = normalize_min_max(chunk_filt)
        if self.frequency_field:
            chunk_filt_norm = np.real(np.fft.rfft(chunk_filt_norm))

        mat_pid = mat_path.split("/")[-1].split(".")[0][:-1]
        df = self.bp_gt[self.bp_gt["PID"] == mat_pid]
        bp_sys = max(min((df.iloc[0, 4] - 100) / (180 - 100), 1), 0)
        bp_dia = max(min((df.iloc[0, 8] - 55) / (100 - 55), 1), 0)

        item = {
            "ppg_chunks": np.reshape(chunk_filt_norm, (1, len(chunk_filt_norm))),
            "bp": np.reshape(np.array([bp_sys, bp_dia]), (1, 2))
        }

        return item


class UWBP_train_skew(Dataset):

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
        chunk_pos = random.randint(0, len(mat_data) - 1)
        chunk_ori = mat_data[chunk_pos]
        chunk, pos = self._concat_chunks(chunk_ori)
        chunk[pos:] = np.mean(chunk[:pos])
        chunk_filt = filter_ppg_signal(chunk, self.LPF, self.HPF, self.FS)
        chunk_filt_norm = normalize_min_max(chunk_filt)
        if self.frequency_field:
            chunk_filt_norm = np.real(np.fft.rfft(chunk_filt_norm))

        mat_pid = mat_path.split("/")[-1].split(".")[0][:-1]
        session = mat_path.split("/")[-1].split(".")[0][-1]
        df = self.bp_gt[self.bp_gt["PID"] == mat_pid]
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
                "bp": np.reshape(np.array([bp_sys, bp_dia]), (1, 2))
            }
        else:
            item = {
                "ppg_chunk": np.reshape(chunk_filt_norm, (1, len(chunk_filt_norm))),
                "bp": np.reshape(np.array([bp_sys, bp_dia]), (1, 2))
                # "bp": np.reshape(np.array([bp_sys]), (1, 1))
            }

        return item


class UWBP_train_face_palm_skew(Dataset):

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

    # def _load_data_list(self, data_list_path):
    #     with open(data_list_path, "r") as f:
    #         lines = f.readlines()
    #     data_list = []
    #     for line in lines:
    #         # data_item = line.strip("\n ").split(",")[0]
    #         data_item = line.strip("\n ").split(",")
    #         data_list.append(data_item)
    #     return data_list

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
        mat_path_list = self.data_list[index]
        mat_data = np.load(mat_path_list, allow_pickle=True)
        face_mat_data = mat_data[0]
        palm_mat_data = mat_data[1]
        # face_mat_data = np.load(mat_path_list[0], allow_pickle=True)
        # palm_mat_data = np.load(mat_path_list[1], allow_pickle=True)
        chunk_pos = random.randint(0, len(face_mat_data) - 1)

        face_chunk_ori = face_mat_data[chunk_pos]
        face_chunk, face_pos = self._concat_chunks(face_chunk_ori)
        face_chunk[face_pos:] = np.mean(face_chunk[:face_pos])

        palm_chunk_ori = palm_mat_data[chunk_pos]
        palm_chunk = np.zeros((self.chunk_length))
        palm_chunk[:len(palm_chunk_ori)] = palm_chunk_ori
        palm_chunk[len(palm_chunk_ori):] = np.mean(palm_chunk_ori)

        face_chunk_filt = filter_ppg_signal(face_chunk, self.LPF, self.HPF, self.FS)
        face_chunk_filt_norm = normalize_min_max(face_chunk_filt)
        palm_chunk_filt = filter_ppg_signal(palm_chunk, self.LPF, self.HPF, self.FS)
        palm_chunk_filt_norm = normalize_min_max(palm_chunk_filt)
        face_palm_diff = face_chunk_filt_norm - palm_chunk_filt_norm
        if self.frequency_field:
            chunk_filt_norm = np.real(np.fft.rfft(chunk_filt_norm))

        mat_pid = mat_path_list.split("/")[-1].split(".")[0][:-1]
        df = self.bp_gt[self.bp_gt["PID"] == mat_pid]
        bp_sys = max(min((df.iloc[0, 4] - 100) / (180 - 100), 1), 0)
        bp_dia = max(min((df.iloc[0, 8] - 55) / (100 - 55), 1), 0)
        if self.derivative:
            face_chunk_filt_norm_fd = np.diff(face_chunk_filt_norm)
            face_chunk_filt_norm_fd = np.insert(face_chunk_filt_norm_fd, 0, 0)
            face_chunk_filt_norm_sd = np.diff(face_chunk_filt_norm_fd)
            face_chunk_filt_norm_sd = np.insert(face_chunk_filt_norm_sd, 0, 0)

            palm_chunk_filt_norm_fd = np.diff(palm_chunk_filt_norm)
            palm_chunk_filt_norm_fd = np.insert(palm_chunk_filt_norm_fd, 0, 0)
            palm_chunk_filt_norm_sd = np.diff(palm_chunk_filt_norm_fd)
            palm_chunk_filt_norm_sd = np.insert(palm_chunk_filt_norm_sd, 0, 0)

            item = {
                "ppg_chunk": np.vstack((np.reshape(face_chunk_filt_norm, (1, len(face_chunk_filt_norm))),
                            np.reshape(face_chunk_filt_norm_fd, (1, len(face_chunk_filt_norm_fd))),
                            np.reshape(face_chunk_filt_norm_sd, (1, len(face_chunk_filt_norm_sd))),
                            np.reshape(palm_chunk_filt_norm, (1, len(palm_chunk_filt_norm))),
                            np.reshape(palm_chunk_filt_norm_fd, (1, len(palm_chunk_filt_norm_fd))),
                            np.reshape(palm_chunk_filt_norm_sd, (1, len(palm_chunk_filt_norm_sd))))),
                "bp": np.reshape(np.array([bp_sys, bp_dia]), (1, 2))
                # "bp": np.reshape(np.array([bp_sys]), (1, 1))
            }
        else:
            item = {
                "ppg_chunk": np.vstack((np.reshape(face_chunk_filt_norm, (1, len(face_chunk_filt_norm))),
                            np.reshape(palm_chunk_filt_norm, (1, len(palm_chunk_filt_norm))),    
                            np.reshape(face_palm_diff, (1, len(face_palm_diff))),
                )),
                # "bp": np.reshape(np.array([bp_sys, bp_dia]), (1, 2))
                "bp": np.reshape(np.array([bp_sys]), (1, 1))
            }

        return item


class UWBP_val_skew(UWBP_train_skew):

    def __init__(self, config):
        super().__init__(config)
        self.data_list_path = config["val_dataset_path"]
        self.data_list = self._load_data_list(self.data_list_path)


class UWBP_val_face_palm_skew(UWBP_train_face_palm_skew):

    def __init__(self, config):
        super().__init__(config)
        self.data_list_path = config["val_dataset_path"]
        self.data_list = self._load_data_list(self.data_list_path)


class UWBP_test_skew(UWBP_train_skew):

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
        chunk_pos = 0
        chunk_ori = mat_data[chunk_pos]
        chunk, pos = self._concat_chunks(chunk_ori)
        chunk[pos:] = np.mean(chunk[:pos])
        chunk_filt = filter_ppg_signal(chunk, self.LPF, self.HPF, self.FS)
        chunk_filt_norm = normalize_min_max(chunk_filt)
        if self.frequency_field:
            chunk_filt_norm = np.real(np.fft.rfft(chunk_filt_norm))

        mat_pid = mat_path.split("/")[-1].split(".")[0][:-1]
        session = mat_path.split("/")[-1].split(".")[0][-1]
        df = self.bp_gt[self.bp_gt["PID"] == mat_pid]
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
                "session": session_id
            }
        else:
            item = {
                "ppg_chunk": np.reshape(chunk_filt_norm, (1, len(chunk_filt_norm))),
                "bp": np.reshape(np.array([bp_sys, bp_dia]), (1, 2)),
                "session": session_id
            }

        return item

class UWBP_test_face_palm_skew(UWBP_train_face_palm_skew):

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
        face_mat_data = mat_data[0]
        palm_mat_data = mat_data[1]

        chunk_pos = 0
        face_chunk_ori = face_mat_data[chunk_pos]
        face_chunk, face_pos = self._concat_chunks(face_chunk_ori)
        face_chunk[face_pos:] = np.mean(face_chunk[:face_pos])

        palm_chunk_ori = palm_mat_data[chunk_pos]
        palm_chunk = np.zeros((self.chunk_length))
        palm_chunk[:len(palm_chunk_ori)] = palm_chunk_ori
        palm_chunk[len(palm_chunk_ori):] = np.mean(palm_chunk_ori)

        face_chunk_filt = filter_ppg_signal(face_chunk, self.LPF, self.HPF, self.FS)
        face_chunk_filt_norm = normalize_min_max(face_chunk_filt)
        palm_chunk_filt = filter_ppg_signal(palm_chunk, self.LPF, self.HPF, self.FS)
        palm_chunk_filt_norm = normalize_min_max(palm_chunk_filt)
        if self.frequency_field:
            chunk_filt_norm = np.real(np.fft.rfft(chunk_filt_norm))

        mat_pid = mat_path.split("/")[-1].split(".")[0][:-1]
        df = self.bp_gt[self.bp_gt["PID"] == mat_pid]
        bp_sys = max(min((df.iloc[0, 4] - 100) / (180 - 100), 1), 0)
        bp_dia = max(min((df.iloc[0, 8] - 55) / (100 - 55), 1), 0)

        if self.derivative:
            face_chunk_filt_norm_fd = np.diff(face_chunk_filt_norm)
            face_chunk_filt_norm_fd = np.insert(face_chunk_filt_norm_fd, 0, 0)
            face_chunk_filt_norm_sd = np.diff(face_chunk_filt_norm_fd)
            face_chunk_filt_norm_sd = np.insert(face_chunk_filt_norm_sd, 0, 0)

            palm_chunk_filt_norm_fd = np.diff(palm_chunk_filt_norm)
            palm_chunk_filt_norm_fd = np.insert(palm_chunk_filt_norm_fd, 0, 0)
            palm_chunk_filt_norm_sd = np.diff(palm_chunk_filt_norm_fd)
            palm_chunk_filt_norm_sd = np.insert(palm_chunk_filt_norm_sd, 0, 0)

            item = {
                "ppg_chunk": np.vstack((np.reshape(face_chunk_filt_norm, (1, len(face_chunk_filt_norm))),
                            np.reshape(face_chunk_filt_norm_fd, (1, len(face_chunk_filt_norm_fd))),
                            np.reshape(face_chunk_filt_norm_sd, (1, len(face_chunk_filt_norm_sd))),
                            np.reshape(palm_chunk_filt_norm, (1, len(palm_chunk_filt_norm))),
                            np.reshape(palm_chunk_filt_norm_fd, (1, len(palm_chunk_filt_norm_fd))),
                            np.reshape(palm_chunk_filt_norm_sd, (1, len(palm_chunk_filt_norm_sd))))),
                "bp": np.reshape(np.array([bp_sys, bp_dia]), (1, 2)),
                # "bp": np.reshape(np.array([bp_sys]), (1, 1))
                "session": session_id
            }
        else:
            item = {
                "ppg_chunk": np.reshape(chunk_filt_norm, (1, len(chunk_filt_norm))),
                "bp": np.reshape(np.array([bp_sys, bp_dia]), (1, 2))
            }

        return item

# class UWBP_train_GPU(Dataset):

#     def __init__(self, config):
#         super().__init__()
#         self.data_list_path = config["train_dataset_path"]
#         self.data_list = self._load_data_list(self.data_list_path)
#         self.bp_gt = pd.read_csv(config["bp_csv_path"])
#         self.chunk_amount = config["chunk_amount"]
#         self.chunk_length = config["chunk_length"]
#         self.LPF = config["filter_lpf"]
#         self.HPF = config["filter_hpf"]
#         self.FS = config["frequency_sample"]
#         self.frequency_field = config["frequency_field"]

#     def _load_data_list(self, data_list_path):
#         with open(data_list_path, "r") as f:
#             lines = f.readlines()
#         data_list = []
#         for line in lines:
#             data_item = line.strip("\n ").split(",")[0]
#             data_list.append(data_item)
#         return data_list
    
#     def _concat_chunks(self, chunks):
#         chunk = np.zeros((self.chunk_length))
#         i = 0
#         for ppg_chunk in chunks:
#             chunk[i : i + len(ppg_chunk)] = np.squeeze(ppg_chunk)
#             i = i + len(ppg_chunk)
#         return chunk

#     def __len__(self):
#         return len(self.data_list)

#     def __getitem__(self, index):
#         mat_path = self.data_list[index]
#         mat_data = np.load(mat_path, allow_pickle=True)
#         chunk_pos = random.randint(0, len(mat_data) - self.chunk_amount - 1)
#         chunks = mat_data[chunk_pos : chunk_pos + self.chunk_amount]
#         chunk = self._concat_chunks(chunks)
#         # chunk_filt = filter_ppg_signal(chunk, self.LPF, self.HPF, self.FS)
#         # chunk_filt_norm = normalize_min_max(chunk_filt)
#         # if self.frequency_field:
#         #     chunk_filt_norm = np.real(np.fft.rfft(chunk_filt_norm))

#         mat_pid = mat_path.split("/")[-1].split(".")[0][:-1]
#         df = self.bp_gt[self.bp_gt["PID"] == mat_pid]
#         bp_sys = max(min((df.iloc[0, 4] - 100) / (200 - 100), 1), 0)
#         bp_dia = max(min((df.iloc[0, 8] - 50) / (100 - 50), 1), 0)

#         item = {
#             "ppg_chunk": np.reshape(chunk, (1, len(chunk))),
#             "bp": np.reshape(np.array([bp_sys, bp_dia]), (1, 2))
#         }

#         return item