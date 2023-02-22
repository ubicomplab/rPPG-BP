import os
import numpy as np
import scipy
import scipy.io
import random
from matplotlib import pyplot as plt
from preprocess import normalize_min_max, filter_ppg_signal
# mat_path = "/gscratch/ubicomp/cm74/clinical_data/ProcessedDataNoVideo/P036a.mat"
# mat_data = scipy.io.loadmat(mat_path)
# print(mat_data)

def concat_chunks(chunks):
    chunk = np.zeros((512))
    i = 0
    for ppg_chunk in chunks:
        chunk[i : i + len(ppg_chunk)] = np.squeeze(ppg_chunk)
        i = i + len(ppg_chunk)
    return chunk, i


if __name__ == "__main__":
    mat_dir = "/gscratch/ubicomp/cm74/bp/bp_preprocess/ppg_chunks_mat_palm_230110/"
    new_mat_dir = "/gscratch/ubicomp/cm74/bp/bp_preprocess/ppg_chunks_mat_palm_skew_230110"
    chunk_amount = 5
    LPF = 0.7
    HPF = 16
    FS = 60
    empty = dict()

    for mat_path in os.listdir(mat_dir):
        score = dict()
        mat_data = np.load(os.path.join(mat_dir, mat_path), allow_pickle=True)
        # print(mat_data.shape)
        # chunk_pos = random.randint(0, len(mat_data) - chunk_amount - 1)
        # chunks = mat_data[chunk_pos : chunk_pos + chunk_amount]

        for i in range(len(mat_data) - chunk_amount):
            skew1 = scipy.stats.skew(mat_data[i])
            skew2 = scipy.stats.skew(mat_data[i + 1])
            skew3 = scipy.stats.skew(mat_data[i + 2])
            skew4 = scipy.stats.skew(mat_data[i + 3])
            skew5 = scipy.stats.skew(mat_data[i + 4])
            average = (skew1 + skew2 + skew3 + skew4 + skew5) / chunk_amount
            score[i] = average
        
        # chunk, pos = concat_chunks(chunks)
        # # chunk = 1 - chunk
        # chunk[pos:] = np.mean(chunk[:pos])
        # chunk_filt = filter_ppg_signal(chunk, LPF, HPF, FS)
        # chunk_filt_norm = normalize_min_max(chunk_filt)
    # plt.figure()
    # plt.plot(chunk_filt_norm)
    # plt.savefig("finger_ppg_input.png")
        # print(score)
        temp_score = sorted(score.items(), key=lambda item: item[1], reverse=True)
        top3 = temp_score[:3]
        
        top3_chunk_list = list()
        for pos in top3:
            new_chunks = mat_data[int(pos[0]) : int(pos[0]) + chunk_amount]
            if len(new_chunks) != 5:
                print(mat_path)
                # print(len(new_chunks))
                empty[mat_path] = len(new_chunks)
                continue
            top3_chunk_list.append(new_chunks)
        # print(np.array(top3_chunk_list).shape)

        # for i in range(len(mat_data)):
        #     skew1 = scipy.stats.skew(mat_data[i])
        #     score[i] = skew1

        # temp_score = sorted(score.items(), key=lambda item: abs(item[1]))
        # top3 = temp_score[:3]

        # top3_chunk_list = list()
        # for pos in top3:
        #     new_chunks = mat_data[int(pos[0])]
        #     # new_chunks = mat_data[int(pos[0]) : int(pos[0]) + chunk_amount]
        #     # if len(new_chunks) != 5:
        #     #     # print(mat_path)
        #     #     # print(len(new_chunks))
        #     #     empty[mat_path] = len(new_chunks)
        #     #     continue
        #     top3_chunk_list.append(new_chunks)

        if len(top3_chunk_list) < 1:
            print(mat_path)
            continue
        np.save(os.path.join(new_mat_dir, mat_path), np.array(top3_chunk_list))
        # for key, value in score.items():
        #     if value
    
    
    # print(len(empty))
    # print(empty)
        