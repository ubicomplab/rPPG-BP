import sys, os, random


def write_data_list(data_list, file_path):
    all_content = ""
    for data_item in sorted(data_list):
        s = f"{data_item}, \n"
        all_content += s
    with open(file_path, "w") as f:
        f.write(all_content)


if __name__ == "__main__":

    train_filename = "uwppg_bp_face_palm_train_fold_"
    val_filename = "uwppg_bp_face_palm_val_fold_"
    data_folder_palm = "/gscratch/ubicomp/cm74/bp/bp_preprocess/ppg_chunks_mat_palm_skew_230110"
    data_folder = "/gscratch/ubicomp/cm74/bp/bp_preprocess/ppg_chunks_mat_skew_230106"
    
    # data_folder_list = sorted(os.listdir(data_folder))
    # fold_len = 0.2 * len(data_folder_list)
    # for j in range(5):
    #     train_data_list = list()
    #     val_data_list = list()
    #     for i in range(len(data_folder_list)):
    #         file_path = data_folder_list[i]
    #         if file_path.split(".")[-1] == "npy":
    #             # if file_path.split(".")[0][-1] != "b":
    #             #     train_data_list.append(os.path.join(data_folder, file_path))
    #             # else:
    #             #     val_data_list.append(os.path.join(data_folder, file_path))
    #             # if int(file_path.split(".")[0][1:4]) < 72:
    #             if (i < (j + 1) * fold_len) and (i >= j * fold_len):
    #                 val_data_list.append(os.path.join(data_folder, file_path))
    #             else:
    #                 train_data_list.append(os.path.join(data_folder, file_path))

    #     write_data_list(train_data_list, train_filename + str(j + 1) + ".txt")

    #     write_data_list(val_data_list, val_filename + str(j + 1) + ".txt")
    

    data_folder_list = sorted(os.listdir(data_folder))
    data_folder_list_palm = sorted(os.listdir(data_folder_palm))
    palm_dict = dict()

    for i in range(len(data_folder_list_palm)):
        session_name = data_folder_list_palm[i].split(".")[0]
        palm_dict[session_name] = data_folder_list_palm[i]

    fold_len = 0.2 * len(data_folder_list)
    for j in range(5):
        train_data_list = list()
        val_data_list = list()
        for i in range(len(data_folder_list)):
            file_path = data_folder_list[i]
            session_name = file_path.split(".")[0]
            if file_path.split(".")[-1] == "npy" and session_name in palm_dict:
                # if file_path.split(".")[0][-1] != "b":
                #     train_data_list.append(os.path.join(data_folder, file_path))
                # else:
                #     val_data_list.append(os.path.join(data_folder, file_path))
                # if int(file_path.split(".")[0][1:4]) < 72:
                if (i < (j + 1) * fold_len) and (i >= j * fold_len):
                    val_data_list.append(os.path.join(data_folder, file_path) + "," + os.path.join(data_folder_palm, palm_dict[session_name]))
                else:
                    train_data_list.append(os.path.join(data_folder, file_path) + "," + os.path.join(data_folder_palm, palm_dict[session_name]))

        write_data_list(train_data_list, train_filename + str(j + 1) + ".txt")

        write_data_list(val_data_list, val_filename + str(j + 1) + ".txt")