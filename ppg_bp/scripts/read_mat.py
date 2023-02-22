import numpy as np
import random, os
from matplotlib import pyplot as plt
from preprocess import normalize_min_max, filter_ppg_signal


def concat_chunks(chunks):
    chunk = np.zeros((512))
    i = 0
    for ppg_chunk in chunks:
        chunk[i : i + len(ppg_chunk)] = np.squeeze(ppg_chunk)
        i = i + len(ppg_chunk)
    return chunk, i


if __name__ == "__main__":
    # mat_path = "/gscratch/ubicomp/xliu0/data3/mnt/Datasets/UWMedicine/ProcessedDataNoVideo/P001b.mat"
    # mat_data = scipy.io.loadmat(mat_path)
    # print(mat_data)
    LPF = 0.7
    HPF = 10
    FS = 60
    folder = "./temp"
    # mat_path = "/gscratch/ubicomp/cm74/bp/bp_preprocess/ppg_chunks_mat_skew_230106/P006a.npy"
    mat_path = "/gscratch/ubicomp/cm74/bp/bp_preprocess/ppg_chunks_mat_face_palm_230124/P006a.npy"
    mat_data = np.load(mat_path, allow_pickle=True)
    mat_data = mat_data[1]
    # chunk_pos = random.randint(0, len(mat_data) - 1)
    for chunk_pos in range(len(mat_data)):
        chunk_ori = mat_data[chunk_pos]

        # chunk = np.zeros((512))
        # ppg_length = len(chunk_ori)
        # chunk[:ppg_length] = chunk_ori
        # chunk[ppg_length:] = np.mean(chunk_ori)
        # chunk_filt = filter_ppg_signal(chunk, LPF, HPF, FS)
        # chunk_filt_norm = normalize_min_max(chunk_filt)

        # chunk, pos = concat_chunks(chunk_ori)
        # chunk[pos:] = np.mean(chunk[:pos])
        chunk = np.zeros((512))
        chunk[:len(chunk_ori)] = chunk_ori
        chunk[len(chunk_ori):] = np.mean(chunk_ori)
        chunk_filt = filter_ppg_signal(chunk, LPF, HPF, FS)
        chunk_filt_norm = normalize_min_max(chunk_filt)

        plt.figure()
        plt.plot(chunk_filt_norm)
        plt.savefig(os.path.join(folder, "face_ppg_skew_input" + str(chunk_pos) + ".png"))