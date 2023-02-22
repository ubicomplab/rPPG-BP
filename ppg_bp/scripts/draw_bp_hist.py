import os, sys, time, random
import numpy as np
import torch
import scipy
from torch.nn import functional as F
from matplotlib import pyplot as plt
from model import M5
from dataset import UWBP_train, UWBP_test, UWBP_val
from config import DEFAULT_CONFIG, TEST_CONFIG


def output_log(log_str, config):
    log_file_path = os.path.join(config["output_dir"], "test_log.txt")
    with open(log_file_path, "a") as f:
        f.write(log_str)


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    config = TEST_CONFIG
    model_path = "/gscratch/ubicomp/cm74/bp/ppg_bp/output/20221219_v1/step_140.pth"
    figure_savepath = config["output_dir"]
    if not os.path.isdir(figure_savepath):
        os.makedirs(figure_savepath)
    model = M5()
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    model.to(device)
    l2_criterion = torch.nn.MSELoss().to(device)
    test_set = UWBP_val(config)
    test_loader = torch.utils.data.DataLoader(test_set, 
            batch_size=1, 
            shuffle=False, 
            num_workers=8, 
            drop_last=False, 
            pin_memory=True
            )
    
    all_loss = []
    bp_sys_gt = []
    bp_sys_pred = []
    bp_dia_gt = []
    bp_dia_pred = []
    print("------------validating--------------")
    start_time = time.time()
    for i, batch in enumerate(test_loader):

        bp = batch["bp"].to(device).to(torch.float32)
        bp_sys_gt.append(torch.squeeze(bp)[0])

        bp_dia_gt.append(torch.squeeze(bp)[1])

    end_time = time.time()
    fig, axs = plt.subplots(1, 2)
    axs[0].hist(torch.tensor(bp_sys_gt).to("cpu").numpy())
    axs[1].hist(torch.tensor(bp_dia_gt).to("cpu").numpy())
    axs[0].set_title("Sys BP")
    axs[1].set_title("Dia BP")
    plt.savefig(os.path.join(figure_savepath, "bp_distribution_144.png"))
    # avg_loss = np.mean(all_loss)
    #     # slope, intercept, sys_r_coef, p_values, se = scipy.stats.linregress(bp_sys_gt, bp_sys_pred)
    #     # slope, intercept, dia_r_coef, p_values, se = scipy.stats.linregress(bp_dia_gt, bp_dia_pred)
    # bp_sys_matrix = torch.vstack((torch.tensor(bp_sys_gt), torch.tensor(bp_sys_pred)))
    # sys_r_coef = torch.corrcoef(bp_sys_matrix)[0, 1]
    # bp_dia_matrix = torch.vstack((torch.tensor(bp_dia_gt), torch.tensor(bp_dia_pred)))
    # dia_r_coef = torch.corrcoef(bp_dia_matrix)[0, 1]
    # duration = end_time - start_time
    # print(f"\r val loss: {avg_loss:.4f}, sys pearson corr {sys_r_coef:.4f}, dia pearson corr {dia_r_coef:.4f}, duration {duration:.2f}")
    # print("===============================================================")
    
    # fig, axs = plt.subplots(1, 2, figsize=(30, 30), dpi=80)
    # axs[0].scatter(torch.tensor(bp_sys_gt).to("cpu").numpy(), torch.tensor(bp_sys_pred).to("cpu").numpy())
    # axs[1].scatter(torch.tensor(bp_dia_gt).to("cpu").numpy(), torch.tensor(bp_dia_pred).to("cpu").numpy())
    # axs[0].set_title("sys bp" + ": corr: " + str(sys_r_coef.to("cpu").numpy())[:6])
    # axs[1].set_title("dia bp" + ": corr: " + str(dia_r_coef.to("cpu").numpy())[:6])
    # axs[0].set_box_aspect(1)
    # axs[1].set_box_aspect(1)
    # plt.savefig(os.path.join(figure_savepath, "bp_corr_plot.png"))

    # output_dir = config["output_dir"]
    # log_str = f"val loss: {avg_loss:.4f} sys pearson corr {sys_r_coef:.4f} dia pearson corr {dia_r_coef:.4f} \n"
    # output_log(log_str, config)
