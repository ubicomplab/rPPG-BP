import os, sys, time, argparse
import numpy as np
import torch
import shutil
import json
import pandas as pd
from torch.nn import functional as F
from matplotlib import pyplot as plt
from model import M5, M5_fusion, M5_fusion_conv, M5_fusion_transformer
from dataset import UWBP_test_manual_demo
from config import DEFAULT_CONFIG, TEST_CONFIG
import BlandAlman



def output_log(log_str, config):
    log_file_path = os.path.join(config["output_dir"], "test_log.txt")
    with open(log_file_path, "a") as f:
        f.write(log_str)

def generate_csv(sys_predict, dia_predict, sys_gt, dia_gt, session, file_name):
    result = dict()
    result["sys_pred"] = sys_predict
    result["dia_pred"] = dia_predict
    result["sys_gt"] = sys_gt
    result["dia_gt"] = dia_gt
    result["session_id"] = session
    df = pd.DataFrame(result)
    df.to_csv(file_name, index=False)

def inference(test_loader):

    # skip_list = ["P070a", "P070b", "P071a", "P071b", "P072a", "P072b", "P073a", "P073b", "P074a", "P074b", "P075a", "P075b", "P076a", "P076b", "P077a", "P077b", "P078a", "P078b", "P079a", "P079b"]
    # Error higher than 20
    # skip_list = ["P003", "P005", "P006", "P009", "P010", "P025", "P035", "P040", "P043", "P050", "P058", "P061", "P062", "P063", "P067", "P068", "P076", "P084", "P086", "P093", "P105", "P114", "P115", "P120"]
    # Error higher than 30
    # skip_list = ["P003", "P010", "P025", "P035", "P061", "P076", "P114", "P120"]
    # Arrythmia
    # skip_list = ["P003", "P011", "P013", "P014", "P015", "P017", "P022", "P023", "P026", "P028", "P032", "P034", "P035", "P038", "P040", "P043", "P046", "P047", "P048", 
    #              "P049", "P050", "P051", "P052", "P055", "P057", "P058", "P061", "P064", "P068", "P072", "P073", "P076", "P078", "P080", "P086", "P090", "P091", "P092", "P093", "P094", "P096", "P100",
    #              "P101", "P102", "P104", "P108", "P110", "P111", "P112", "P116", "P117", "P118", "P120", "P122", "P123", "P124", "P126", "P129", "P130", "P132", "P134", "P137", "P138", "P139", "P141", "P142", "P144"]
    # Atrial fibrillation
    # skip_list = ["P013", "P034", "P046", "P048", "P049", "P064", "P072", "P080", "P093", "P094", "P095", "P104", "P111", "P116", "P118", "P123", "P126",
    #             "P137", "P138"]
    # # # frequent ectopy
    # skip_list = ["P022", "P023", "P032", "P035", "P038", "P050", "P055", "P091", "P092", "P102", "P112", "P124", "P138", "P144"]
    # # # paced rhythm
    # skip_list = ["P040", "P051", "P068", "P073", "P078", "P086", "P090", "P091", "P122", "P129", "P130"]
    # # frequent ectopy + Atrial
    # skip_list = ["P022", "P023", "P032", "P035", "P038", "P050", "P055", "P091", "P092", "P102", "P112", "P124", "P138", "P144", "P013", "P034", "P046", "P048", "P049", "P064", "P072", "P080", "P093", "P094", "P095", "P104", "P111", "P116", "P118", "P123", "P126",
    #             "P137", "P138"]
    # # frequent ectopy + Atrial + paced rhythm
    # skip_list = ["P022", "P023", "P032", "P035", "P038", "P050", "P055", "P091", "P092", "P102", "P112", "P124", "P138", "P144", "P013", "P034", "P046", "P048", "P049", "P064", "P072", "P080", "P093", "P094", "P095", "P104", "P111", "P116", "P118", "P123", "P126",
    #             "P137", "P138", "P040", "P051", "P068", "P073", "P078", "P086", "P090", "P091", "P122", "P129", "P130"]
    for i, data in enumerate(test_loader):
        temp_loss = list()
        tmp_by_sys_gt = list()
        tmp_bp_sys_pred = list()
        tmp_bp_dia_gt = list()
        tmp_bp_dia_pred = list()
        flag = 0
        for j, batch in enumerate(data):
            
            ### skip the selected session
            # if str(batch["session"][0][:-1]) in skip_list:
            #     flag = 1
            #     continue

            chunks = batch["ppg_chunk"].to(device).to(torch.float32)
            bp = batch["bp"].to(device).to(torch.float32)
            age = batch["age"].to(device).to(torch.float32)
            bmi = batch["bmi"].to(device).to(torch.float32)
            # gender = batch["gender"].to(device).to(torch.float32)
            # bp_predict = model(age, bmi)
            # bp_predict = model(chunks, age, bmi, gender)
            bp_predict = model(chunks, age, bmi)
            # bp_predict = model(chunks)
            l2_loss = l2_criterion(bp_predict, bp).item()
            temp_loss.append(l2_loss)
            # tmp_by_sys_gt.append(torch.squeeze(bp).to("cpu").detach().numpy())
            # tmp_bp_sys_pred.append(torch.squeeze(bp_predict).to("cpu").detach().numpy())
            tmp_by_sys_gt.append(torch.squeeze(bp)[0].to("cpu").detach().numpy())
            tmp_bp_sys_pred.append(torch.squeeze(bp_predict)[0].to("cpu").detach().numpy())
            tmp_bp_dia_gt.append(torch.squeeze(bp)[1].to("cpu").detach().numpy())
            tmp_bp_dia_pred.append(torch.squeeze(bp_predict)[1].to("cpu").detach().numpy())

        # if flag == 1:
        #     continue

        if abs(np.mean(tmp_bp_sys_pred) - np.mean(tmp_by_sys_gt)) > 0.3:
            print(batch["session"])
            print(np.mean(tmp_bp_sys_pred), np.mean(tmp_by_sys_gt))

        all_loss.append(np.mean(temp_loss))
        bp_sys_gt.append(np.mean(tmp_by_sys_gt))
        bp_sys_pred.append(np.mean(tmp_bp_sys_pred))
        bp_dia_gt.append(np.mean(tmp_bp_dia_gt))
        bp_dia_pred.append(np.mean(tmp_bp_dia_pred))
        sessions.append(batch["session"])
        sys.stdout.write(f"\rval [{i + 1}/{len(test_loader)}] loss {np.mean(temp_loss):.4f}")
        sys.stdout.flush()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="settings")
    parser.add_argument("-c", "--config", type=str, default="./test_config.json", help="config setting")
    opt = parser.parse_args()

    with open(opt.config, "r") as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = config["model_path1"]
    model_path2 = config["model_path2"]
    model_path3 = config["model_path3"]
    model_path4 = config["model_path4"]
    model_path5 = config["model_path5"]
    figure_savepath = config["output_dir"]
    if not os.path.isdir(figure_savepath):
        os.makedirs(figure_savepath)
    # model = M5()
    if config["derivative_input"]:
        model = M5(n_input=6)
    else:
        # model = M5_fusion_conv(n_input=1, n_output=2)
        model = M5_fusion_transformer(n_input=1, n_output=2)

    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt, strict=True)
    model.eval()
    model.to(device)
     
    l2_criterion = torch.nn.MSELoss().to(device)
    test_set = UWBP_test_manual_demo(config, 1)
    test_loader = torch.utils.data.DataLoader(test_set, 
            batch_size=1, 
            shuffle=False, 
            num_workers=8, 
            drop_last=False, 
            pin_memory=True
            )
    sys_max = 180
    sys_min = 100
    dia_max = 100
    dia_min = 55
    all_loss = []
    bp_sys_gt = []
    bp_sys_pred = []
    bp_dia_gt = []
    bp_dia_pred = []
    sessions = []
    print("------------validating--------------")

    start_time = time.time()
    inference(test_loader)
    end_time = time.time()


    ckpt = torch.load(model_path2, map_location=device)
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    model.to(device)
    test_set = UWBP_test_manual_demo(config, 2)
    test_loader = torch.utils.data.DataLoader(test_set, 
            batch_size=1, 
            shuffle=False, 
            num_workers=8, 
            drop_last=False, 
            pin_memory=True
            )
    print("------------validating--------------")
    inference(test_loader)
    

    ckpt = torch.load(model_path3, map_location=device)
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    model.to(device)
    test_set = UWBP_test_manual_demo(config, 3)
    test_loader = torch.utils.data.DataLoader(test_set, 
            batch_size=1, 
            shuffle=False, 
            num_workers=8, 
            drop_last=False, 
            pin_memory=True
            )
    print("------------validating--------------")
    inference(test_loader)

    
    ckpt = torch.load(model_path4, map_location=device)
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    model.to(device)
    test_set = UWBP_test_manual_demo(config, 4)
    test_loader = torch.utils.data.DataLoader(test_set, 
            batch_size=1, 
            shuffle=False, 
            num_workers=8, 
            drop_last=False, 
            pin_memory=True
            )
    print("------------validating--------------")
    inference(test_loader)


    ckpt = torch.load(model_path5, map_location=device)
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    model.to(device)
    test_set = UWBP_test_manual_demo(config, 5)
    test_loader = torch.utils.data.DataLoader(test_set, 
            batch_size=1, 
            shuffle=False, 
            num_workers=8, 
            drop_last=False, 
            pin_memory=True
            )
    print("------------validating--------------")
    inference(test_loader)


    avg_loss = np.mean(all_loss)
    bp_sys_matrix = torch.vstack((torch.tensor(bp_sys_gt), torch.tensor(bp_sys_pred)))
    sys_r_coef = torch.corrcoef(bp_sys_matrix)[0, 1]
    bp_dia_matrix = torch.vstack((torch.tensor(bp_dia_gt), torch.tensor(bp_dia_pred)))
    dia_r_coef = torch.corrcoef(bp_dia_matrix)[0, 1]
    # sys_var, sys_mean = torch.var_mean(torch.tensor(bp_sys_pred) - torch.tensor(bp_sys_gt))
    # dia_var, dia_mean = torch.var_mean(torch.tensor(bp_dia_pred) - torch.tensor(bp_dia_gt))
    sys_var, sys_mean = torch.var_mean((torch.tensor(bp_sys_pred) - torch.tensor(bp_sys_gt)) * (sys_max - sys_min))
    dia_var, dia_mean = torch.var_mean((torch.tensor(bp_dia_pred) - torch.tensor(bp_dia_gt)) * (dia_max - dia_min))
    sys_mae = torch.mean(torch.abs((torch.tensor(bp_sys_pred) - torch.tensor(bp_sys_gt)) * (sys_max - sys_min)))
    dia_mae = torch.mean(torch.abs((torch.tensor(bp_dia_pred) - torch.tensor(bp_dia_gt)) * (dia_max - dia_min)))

    sys_mmhg = sys_mean.to("cpu").numpy()
    dia_mmhg = dia_mean.to("cpu").numpy()
    sys_mae = sys_mae.to("cpu").numpy()
    dia_mae = dia_mae.to("cpu").numpy()
    sys_mmhg_std = np.sqrt(sys_var.to("cpu").numpy())
    dia_mmhg_std = np.sqrt(dia_var.to("cpu").numpy())

    duration = end_time - start_time
    print(f"\r val loss: {avg_loss:.4f}, sys pearson corr {sys_r_coef:.4f}, dia pearson corr {dia_r_coef:.4f}, duration {duration:.2f}")
    print("===============================================================")
    print("sys mean: " + str(sys_mmhg)[:5] + " std: " + str(sys_mmhg_std)[:5] + " corr: " + str(sys_r_coef.to("cpu").numpy())[:5])
    print("dia mean: " + str(dia_mmhg)[:5] + " std: " + str(dia_mmhg_std)[:5] + " corr: " + str(dia_r_coef.to("cpu").numpy())[:5])
    fig, axs = plt.subplots(1, 2, figsize=(30, 30), dpi=80, linewidth=10)
    axs[0].scatter(torch.tensor(bp_sys_gt).to("cpu").numpy() * (sys_max - sys_min) + sys_min, torch.tensor(bp_sys_pred).to("cpu").numpy() * (sys_max - sys_min) + sys_min)
    axs[1].scatter(torch.tensor(bp_dia_gt).to("cpu").numpy() * (dia_max - dia_min) + dia_min, torch.tensor(bp_dia_pred).to("cpu").numpy() * (dia_max - dia_min) + dia_min)
    axs[0].set_title("sys mean: " + str(sys_mmhg)[:5] + " mmHg std: " + str(sys_mmhg_std)[:5] + " mmHg corr: " + str(sys_r_coef.to("cpu").numpy())[:5], fontsize=30, pad=50)
    axs[1].set_title("dia mean: " + str(dia_mmhg)[:5] + " mmHg std: " + str(dia_mmhg_std)[:5] + " mmHg corr: " + str(dia_r_coef.to("cpu").numpy())[:5], fontsize=30, pad=50)

    axs[0].set_xlim(sys_min, sys_max)
    axs[0].set_ylim(sys_min, sys_max)
    axs[1].set_xlim(dia_min, dia_max)
    axs[1].set_ylim(dia_min, dia_max)
    axs[0].set_xlabel("Ground Truth", fontsize=30)
    axs[0].set_ylabel("Prediction", fontsize=30)
    axs[1].set_xlabel("Ground Truth", fontsize=30)
    axs[1].set_ylabel("Prediction", fontsize=30)
    axs[0].set_box_aspect(1)
    axs[1].set_box_aspect(1)
    axs[0].spines['right'].set_visible(False)
    axs[0].spines['top'].set_visible(False) 
    axs[0].spines['left'].set_visible(False)
    axs[0].spines['bottom'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['top'].set_visible(False) 
    axs[1].spines['left'].set_visible(False)
    axs[1].spines['bottom'].set_visible(False)
    axs[0].tick_params(axis='both', which='major', labelsize=20)
    axs[1].tick_params(axis='both', which='major', labelsize=20)

    plt.savefig(os.path.join(figure_savepath, "bp_corr_plot.png"))

    output_dir = config["output_dir"]
    shutil.copyfile(opt.config, os.path.join(output_dir, "test_config.json"))
    log_str = f"val loss: {avg_loss:.4f} sys pearson corr {sys_r_coef:.4f} dia pearson corr {dia_r_coef:.4f} \n"
    output_log(log_str, config)

    sys_pred_mmhg = torch.tensor(bp_sys_pred).to("cpu").numpy() * (sys_max - sys_min) + sys_min
    sys_gt_mmhg = torch.tensor(bp_sys_gt).to("cpu").numpy() * (sys_max - sys_min) + sys_min
    dia_pred_mmhg = torch.tensor(bp_dia_pred).to("cpu").numpy() * (dia_max - dia_min) + dia_min
    dia_gt_mmhg = torch.tensor(bp_dia_gt).to("cpu").numpy() * (dia_max - dia_min) + dia_min
    generate_csv(sys_pred_mmhg, dia_pred_mmhg, sys_gt_mmhg, dia_gt_mmhg, sessions, os.path.join(output_dir, "bp_results.csv"))


    compare = BlandAlman.BlandAltman(torch.tensor(bp_sys_gt).to("cpu").numpy() * (sys_max - sys_min) + sys_min, torch.tensor(bp_sys_pred).to("cpu").numpy() * (sys_max - sys_min) + sys_min,averaged=True)
    compare.scatter_plot(x_label='Ground Truth Systolic BP [mmHg]',y_label='Prediction Systolic BP [mmHg]', low_range=sys_min, up_range=sys_max, figure_size=(5, 5), the_title="sys mean: " + str(sys_mmhg)[:5] + " sys mae: " + str(sys_mae)[:5] + " mmHg; std: " + str(sys_mmhg_std)[:5] + " mmHg; corr: " + str(sys_r_coef.to("cpu").numpy())[:5], file_name=os.path.join(output_dir, 'ScatterPlot_sys.png'))
    compare.difference_plot(x_label='Ground Truth Systolic BP [mmHg]',y_label='Prediction Systolic BP [mmHg]', figure_size=(5, 5), file_name=os.path.join(output_dir, 'BlandAlman_sys.png'))
    compare.print_stats(round_amount = 3)

    compare = BlandAlman.BlandAltman(torch.tensor(bp_dia_gt).to("cpu").numpy() * (dia_max - dia_min) + dia_min, torch.tensor(bp_dia_pred).to("cpu").numpy() * (dia_max - dia_min) + dia_min,averaged=True)
    compare.scatter_plot(x_label='Ground Truth Diastolic BP [mmHg]',y_label='Prediction Diastolic BP [mmHg]', low_range=dia_min, up_range=dia_max, figure_size=(5, 5), the_title="dia mean: " + str(dia_mmhg)[:5] + " dia mae: " + str(dia_mae)[:5] + " mmHg; std: " + str(dia_mmhg_std)[:5] + " mmHg; corr: " + str(dia_r_coef.to("cpu").numpy())[:5], file_name=os.path.join(output_dir, 'ScatterPlot_dia.png'))
    compare.difference_plot(x_label='Ground Truth Diastolic BP [mmHg]',y_label='Prediction Diastolic BP [mmHg]', figure_size=(5, 5), file_name=os.path.join(output_dir, 'BlandAlman_dia.png'))
    compare.print_stats(round_amount = 3)
