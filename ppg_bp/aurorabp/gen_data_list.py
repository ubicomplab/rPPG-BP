import sys, os, random
import pandas as pd
from collections import defaultdict

def write_data_list(data_list, file_path):
    all_content = ""
    for data_item in sorted(data_list):
        s = f"{data_item}, \n"
        all_content += s
    with open(file_path, "w") as f:
        f.write(all_content)


if __name__ == "__main__":

    train_filename = "ms_aurorabp_ambulatory_selected_train_fold"
    val_filename = "ms_aurorabp_ambulatory_selected_val_fold"
    train_data_list = list()
    val_data_list = list()

    data_folder = "/gscratch/ubicomp/cm74/MS_aurorabp/measurements_oscillometric_ambulatory_seleceted"
    pids = os.listdir(data_folder)
    fold_len = 0.2 * len(pids)
    # for pid in pids:
    #     print(pid)
    #     measurements = os.listdir(os.path.join(data_folder, pid))
    #     # if not os.path.exists(os.path.join(result_path, pid)):
    #     #     os.makedirs(os.path.join(result_path, pid))
    #     for ms in measurements:
    #         train_data_list.append(os.path.join(data_folder, pid, ms))
    #         tmp = os.path.join(data_folder, pid, ms)
    #         new_tmp = tmp.split("/")[-1].split(".")[0].split("_")[2:-1]
    #         res = " ".join([item for item in new_tmp])
    #         print(res)
    # write_data_list(train_data_list, train_filename + "all" + ".txt")

            # write_data_list(val_data_list, val_filename + str(j + 1) + ".txt")
    ### Random selection
    # fold_len = 0.2 * len(data_folder_list)
    for j in range(5):
        train_data_list = list()
        val_data_list = list()
        for i in range(len(pids)):
            print(pids)
            measurements = os.listdir(os.path.join(data_folder, pids[i]))
            # if not os.path.exists(os.path.join(result_path, pid)):
            #     os.makedirs(os.path.join(result_path, pid))
            for ms in measurements:
                if (i < (j + 1) * fold_len) and (i >= j * fold_len):
                    val_data_list.append(os.path.join(data_folder, pids[i], ms))
                else:
                    train_data_list.append(os.path.join(data_folder, pids[i], ms))
                tmp = os.path.join(data_folder, pids[i], ms)
                new_tmp = tmp.split("/")[-1].split(".")[0].split("_")[2:-1]
                res = " ".join([item for item in new_tmp])
                # print(res)
        write_data_list(train_data_list, train_filename + str(j + 1) + ".txt")
        write_data_list(val_data_list, val_filename + str(j + 1) + ".txt")

    
    ### Sample on bp distribution
    # file_path = "/gscratch/ubicomp/cm74/clinical_data/BPData_230223_demograph.csv"
    # df = pd.read_csv(file_path)
    # map = defaultdict(list)
    # fold = defaultdict(list)
    # fold_path = defaultdict(list)

    # for i in range(len(data_folder_list)):
    #     file_path = data_folder_list[i]
    #     if file_path.split(".")[-1] == "npy":
    #         pid = file_path.split(".")[0][:-1]
    #         # print(pid)
    #         row = df[df["PID"] == pid]
    #         sys_bp =row["SYSBP"].values[0]
    #         # print(type(df["PID"][1]))
    #         if sys_bp <= 111:
    #             sys_bp = 111
    #         elif sys_bp <= 120:
    #             sys_bp = 120
    #         elif sys_bp <= 130:
    #             sys_bp = 130
    #         elif sys_bp <= 140:
    #             sys_bp = 140
    #         elif sys_bp <= 150:
    #             sys_bp = 150
    #         elif sys_bp <= 160:
    #             sys_bp = 160
    #         else:
    #             sys_bp = 170
    #         if sys_bp not in map.keys():
    #             map[sys_bp].append(pid)
    #         elif pid not in map[sys_bp]:
    #             map[sys_bp].append(pid)
    # flag = 0
    # for key, value in map.items():
    #     residual = len(value) % 5
    #     for i in range(5):
        
    #         if i < 4:
    #             pos = len(value) // 5
    #             fold[i] += value[pos * i : pos * (i + 1)]
    #         else:
    #             pos = len(value) // 5
    #             fold[i] += value[pos * i : pos * (i + 1)]
    #             if flag == 0:
    #                 flag = 1
    #                 for j in range(residual):
    #                     fold[j] += [value[pos * (i + 1) + j]]
    #             else:
    #                 flag = 0
    #                 for j in range(residual):
    #                     fold[4 - j] += [value[pos * (i + 1) + j]]
    # for i in range(len(data_folder_list)):
    #     file_path = data_folder_list[i]
    #     if file_path.split(".")[-1] == "npy":
    #         pid = file_path.split(".")[0][:-1]
    #         for key, value in fold.items():
    #             if pid in value:
    #                 fold_path[key].append(os.path.join(data_folder, file_path))
    
    # for i in range(5):
    #     value = fold[i]
    #     print(i, len(value))
    #     print(value)
    #     train_list = list()
    #     val_list = list()
    #     for j in range(len(data_folder_list)):
    #         file_path = data_folder_list[j]
    #         if file_path.split(".")[-1] == "npy":
    #             pid = file_path.split(".")[0][:-1]
    #             if pid in value:
    #                 val_list.append(os.path.join(data_folder, file_path))
    #             else:
    #                 train_list.append(os.path.join(data_folder, file_path))
        
    #     write_data_list(train_list, train_filename + str(i + 1) + ".txt")

    #     write_data_list(val_list, val_filename + str(i + 1) + ".txt")

    # data_folder_list = sorted(os.listdir(data_folder))
    # data_folder_list_palm = sorted(os.listdir(data_folder_palm))
    # palm_dict = dict()

    # for i in range(len(data_folder_list_palm)):
    #     session_name = data_folder_list_palm[i].split(".")[0]
    #     palm_dict[session_name] = data_folder_list_palm[i]

    # fold_len = 0.2 * len(data_folder_list)
    # for j in range(5):
    #     train_data_list = list()
    #     val_data_list = list()
    #     for i in range(len(data_folder_list)):
    #         file_path = data_folder_list[i]
    #         session_name = file_path.split(".")[0]
    #         if file_path.split(".")[-1] == "npy" and session_name in palm_dict:
    #             # if file_path.split(".")[0][-1] != "b":
    #             #     train_data_list.append(os.path.join(data_folder, file_path))
    #             # else:
    #             #     val_data_list.append(os.path.join(data_folder, file_path))
    #             # if int(file_path.split(".")[0][1:4]) < 72:
    #             if (i < (j + 1) * fold_len) and (i >= j * fold_len):
    #                 val_data_list.append(os.path.join(data_folder, file_path) + "," + os.path.join(data_folder_palm, palm_dict[session_name]))
    #             else:
    #                 train_data_list.append(os.path.join(data_folder, file_path) + "," + os.path.join(data_folder_palm, palm_dict[session_name]))

    #     write_data_list(train_data_list, train_filename + str(j + 1) + ".txt")

    #     write_data_list(val_data_list, val_filename + str(j + 1) + ".txt")