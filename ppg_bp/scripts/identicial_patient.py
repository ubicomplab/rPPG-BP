from collections import defaultdict

def load_data_list(data_list_path):
    with open(data_list_path, "r") as f:
        lines = f.readlines()
    data_list = []
    for line in lines:
        data_item = line.strip("\n ").split(",")[0]
        # data_item = line.strip("\n ").split(",")
        data_list.append(data_item)
    return data_list

if __name__ == "__main__":
    val_path1 = "/gscratch/ubicomp/cm74/bp/ppg_bp/uwppg_bp_val_fold_1.txt"
    val_path2 = "/gscratch/ubicomp/cm74/bp/ppg_bp/uwppg_bp_val_fold_2.txt"
    val_path3 = "/gscratch/ubicomp/cm74/bp/ppg_bp/uwppg_bp_val_fold_3.txt"
    val_path4 = "/gscratch/ubicomp/cm74/bp/ppg_bp/uwppg_bp_val_fold_4.txt"
    val_path5 = "/gscratch/ubicomp/cm74/bp/ppg_bp/uwppg_bp_val_fold_5.txt"

    data_list1 = load_data_list(val_path1)
    data_list2 = load_data_list(val_path2)
    data_list3 = load_data_list(val_path3)
    data_list4 = load_data_list(val_path4)
    data_list5 = load_data_list(val_path5)

    count = defaultdict(lambda:0)
    for i in range(len(data_list1)):
        count[data_list1[i].split(".")[0][-5:-1]] += 1
    
    for i in range(len(data_list2)):
        count[data_list2[i].split(".")[0][-5:-1]] += 1

    for i in range(len(data_list3)):
        count[data_list3[i].split(".")[0][-5:-1]] += 1

    for i in range(len(data_list4)):
        count[data_list4[i].split(".")[0][-5:-1]] += 1
    
    for i in range(len(data_list5)):
        count[data_list5[i].split(".")[0][-5:-1]] += 1

    print(count)
    print(len(count))
