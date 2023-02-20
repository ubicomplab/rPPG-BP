import os, sys, time, argparse
import numpy as np
import torch
import shutil
import json
import pandas as pd
from torch.nn import functional as F
from matplotlib import pyplot as plt
from model import M5, M5_fusion_conv
from dataset import UWBP_test_manual_demo


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
    for i, data in enumerate(test_loader):
        temp_loss = list()
        tmp_by_sys_gt = list()
        tmp_bp_sys_pred = list()
        tmp_bp_dia_gt = list()
        tmp_bp_dia_pred = list()
        for j, batch in enumerate(data):
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

        # index = np.argmin(temp_loss)

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
        model = M5_fusion_conv(n_output=2)
        # model = FC_naive()
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

    end_time = time.time()
    
    avg_loss = np.mean(all_loss)
        # slope, intercept, sys_r_coef, p_values, se = scipy.stats.linregress(bp_sys_gt, bp_sys_pred)
        # slope, intercept, dia_r_coef, p_values, se = scipy.stats.linregress(bp_dia_gt, bp_dia_pred)
    bp_sys_matrix = torch.vstack((torch.tensor(bp_sys_gt), torch.tensor(bp_sys_pred)))
    sys_r_coef = torch.corrcoef(bp_sys_matrix)[0, 1]
    bp_dia_matrix = torch.vstack((torch.tensor(bp_dia_gt), torch.tensor(bp_dia_pred)))
    dia_r_coef = torch.corrcoef(bp_dia_matrix)[0, 1]
    # sys_var, sys_mean = torch.var_mean(torch.tensor(bp_sys_pred) - torch.tensor(bp_sys_gt))
    # dia_var, dia_mean = torch.var_mean(torch.tensor(bp_dia_pred) - torch.tensor(bp_dia_gt))
    sys_var, sys_mean = torch.var_mean((torch.tensor(bp_sys_pred) - torch.tensor(bp_sys_gt)) * (sys_max - sys_min))
    dia_var, dia_mean = torch.var_mean((torch.tensor(bp_dia_pred) - torch.tensor(bp_dia_gt)) * (dia_max - dia_min))
    # sys_mmhg = sys_mean.to("cpu").numpy() * (sys_max - sys_min) + sys_min
    # dia_mmhg = dia_mean.to("cpu").numpy() * (dia_max - dia_min) + dia_min
    # sys_mmhg_std = np.sqrt(sys_var.to("cpu").numpy()) * (sys_max - sys_min) + sys_min
    # dia_mmhg_std = np.sqrt(dia_var.to("cpu").numpy()) * (dia_max - dia_min) + dia_min
    sys_mmhg = sys_mean.to("cpu").numpy()
    dia_mmhg = dia_mean.to("cpu").numpy()
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
    # axs[1].set_title("dia bp" + ": corr: " + str(dia_r_coef.to("cpu").numpy())[:5])
    # axs[0].set_xlim(0, 1)
    # axs[0].set_ylim(0, 1)
    # axs[1].set_xlim(0, 1)
    # axs[1].set_ylim(0, 1)
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
    # axs[0].set_xticks(fontsize=20)
    # axs[1].set_yticks(fontsize=20)
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


    # compare = BlandAlman.BlandAltman(torch.tensor(bp_sys_gt).to("cpu").numpy() * (sys_max - sys_min) + sys_min, torch.tensor(bp_sys_pred).to("cpu").numpy() * (sys_max - sys_min) + sys_min,averaged=True)
    # compare.scatter_plot(x_label='Ground Truth Systolic BP [mmHg]',y_label='Prediction Systolic BP [mmHg]', figure_size=(5, 5), the_title="sys mean: " + str(sys_mmhg)[:5] + " mmHg; std: " + str(sys_mmhg_std)[:5] + " mmHg; corr: " + str(sys_r_coef.to("cpu").numpy())[:5], file_name=f'ScatterPlot_sys.pdf')
    # compare.difference_plot(x_label='Ground Truth Systolic BP [mmHg]',y_label='Prediction Systolic BP [mmHg]', figure_size=(5, 5), file_name=f'BlandAlman_sys.pdf')
    # compare.print_stats(round_amount = 3)

    # compare = BlandAlman.BlandAltman(torch.tensor(bp_dia_gt).to("cpu").numpy() * (dia_max - dia_min) + dia_min, torch.tensor(bp_dia_pred).to("cpu").numpy() * (dia_max - dia_min) + dia_min,averaged=True)
    # compare.scatter_plot(x_label='Ground Truth Systolic BP [mmHg]',y_label='Prediction Systolic BP [mmHg]', low_range=50, up_range=100, figure_size=(5, 5), the_title="dia mean: " + str(dia_mmhg)[:5] + " mmHg; std: " + str(dia_mmhg_std)[:5] + " mmHg; corr: " + str(dia_r_coef.to("cpu").numpy())[:5], file_name=f'ScatterPlot_dia.pdf')
    # compare.difference_plot(x_label='Ground Truth Systolic BP [mmHg]',y_label='Prediction Systolic BP [mmHg]', figure_size=(5, 5), file_name=f'BlandAlman_dia.pdf')
    # compare.print_stats(round_amount = 3)

    # ckpt = torch.load(model_path4, map_location=device)
    # model.load_state_dict(ckpt, strict=False)
    # model.eval()
    # model.to(device)
    # test_set = UWBP_test_manual_skew(config, 4)
    # test_loader = torch.utils.data.DataLoader(test_set, 
    #         batch_size=1, 
    #         shuffle=False, 
    #         num_workers=8, 
    #         drop_last=False, 
    #         pin_memory=True
    #         )
    # print("------------validating--------------")
    # start_time = time.time()
    # for i, batch in enumerate(test_loader):

    #     chunks = batch["ppg_chunk"].to(device).to(torch.float32)
    #     bp = batch["bp"].to(device).to(torch.float32)
    #     bp_predict = model(chunks)
    #     l2_loss = l2_criterion(bp_predict, bp).item()

    #     all_loss.append(l2_loss)
    #     bp_sys_gt.append(torch.squeeze(bp)[0])
    #     bp_sys_pred.append(torch.squeeze(bp_predict)[0])
    #     bp_dia_gt.append(torch.squeeze(bp)[1])
    #     bp_dia_pred.append(torch.squeeze(bp_predict)[1])
    #     sessions.append(batch["session"])

    #     sys.stdout.write(f"\rval [{i + 1}/{len(test_loader)}] loss {l2_loss:.4f}")
    #     sys.stdout.flush()
    # end_time = time.time()
