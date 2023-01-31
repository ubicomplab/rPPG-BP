import os, sys, time, argparse
import numpy as np
import torch
import shutil
import json
import pandas as pd
from torch.nn import functional as F
from matplotlib import pyplot as plt
from model import M5
from dataset import UWBP_train, UWBP_test, UWBP_val, UWBP_test_skew, UWBP_test_face_palm_skew


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
        model = M5()
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    model.to(device)
    l2_criterion = torch.nn.MSELoss().to(device)
    test_set = UWBP_test_skew(config, 1)
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
    for i, batch in enumerate(test_loader):

        chunks = batch["ppg_chunk"].to(device).to(torch.float32)
        # bp_sys = batch["bp_sys"].to(self.device)
        # bp_dia = batch["bp_dia"].to(self.device)
        bp = batch["bp"].to(device).to(torch.float32)
        temp_loss = 1000
        bp_predict_temp = 0
        # for i in range(len(chunks)):
        bp_predict = model(chunks)
        #     bp_predict = model(chunks[i : i + config["chunk_amount"]])
        l2_loss = l2_criterion(bp_predict, bp).item()
        #     if l2_loss < temp_loss:
        #         bp_predict_temp = bp_predict
        # bp_predict = bp_predict_temp

        all_loss.append(l2_loss)
        bp_sys_gt.append(torch.squeeze(bp)[0])
        bp_sys_pred.append(torch.squeeze(bp_predict)[0])
        bp_dia_gt.append(torch.squeeze(bp)[1])
        bp_dia_pred.append(torch.squeeze(bp_predict)[1])
        sessions.append(batch["session"])
        # bp_sys_gt.append(torch.squeeze(bp)[0].to("cpu").numpy())
        # bp_sys_pred.append(torch.squeeze(bp_predict)[0].to("cpu").numpy())
        # bp_dia_gt.append(torch.squeeze(bp)[1].to("cpu").numpy())
        # bp_dia_pred.append(torch.squeeze(bp_predict)[1].to("cpu").numpy())
        sys.stdout.write(f"\rval [{i + 1}/{len(test_loader)}] loss {l2_loss:.4f}")
        sys.stdout.flush()
    end_time = time.time()


    ckpt = torch.load(model_path2, map_location=device)
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    model.to(device)
    test_set = UWBP_test_skew(config, 2)
    test_loader = torch.utils.data.DataLoader(test_set, 
            batch_size=1, 
            shuffle=False, 
            num_workers=8, 
            drop_last=False, 
            pin_memory=True
            )
    print("------------validating--------------")
    start_time = time.time()
    for i, batch in enumerate(test_loader):

        chunks = batch["ppg_chunk"].to(device).to(torch.float32)
        bp = batch["bp"].to(device).to(torch.float32)

        bp_predict = model(chunks)
        l2_loss = l2_criterion(bp_predict, bp).item()

        all_loss.append(l2_loss)
        bp_sys_gt.append(torch.squeeze(bp)[0])
        bp_sys_pred.append(torch.squeeze(bp_predict)[0])
        bp_dia_gt.append(torch.squeeze(bp)[1])
        bp_dia_pred.append(torch.squeeze(bp_predict)[1])
        sessions.append(batch["session"])

        sys.stdout.write(f"\rval [{i + 1}/{len(test_loader)}] loss {l2_loss:.4f}")
        sys.stdout.flush()
    end_time = time.time()

    
    ckpt = torch.load(model_path3, map_location=device)
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    model.to(device)
    test_set = UWBP_test_skew(config, 3)
    test_loader = torch.utils.data.DataLoader(test_set, 
            batch_size=1, 
            shuffle=False, 
            num_workers=8, 
            drop_last=False, 
            pin_memory=True
            )
    print("------------validating--------------")
    start_time = time.time()
    for i, batch in enumerate(test_loader):

        chunks = batch["ppg_chunk"].to(device).to(torch.float32)
        bp = batch["bp"].to(device).to(torch.float32)
        bp_predict = model(chunks)
        l2_loss = l2_criterion(bp_predict, bp).item()

        all_loss.append(l2_loss)
        bp_sys_gt.append(torch.squeeze(bp)[0])
        bp_sys_pred.append(torch.squeeze(bp_predict)[0])
        bp_dia_gt.append(torch.squeeze(bp)[1])
        bp_dia_pred.append(torch.squeeze(bp_predict)[1])
        sessions.append(batch["session"])

        sys.stdout.write(f"\rval [{i + 1}/{len(test_loader)}] loss {l2_loss:.4f}")
        sys.stdout.flush()
    end_time = time.time()

    
    ckpt = torch.load(model_path4, map_location=device)
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    model.to(device)
    test_set = UWBP_test_skew(config, 4)
    test_loader = torch.utils.data.DataLoader(test_set, 
            batch_size=1, 
            shuffle=False, 
            num_workers=8, 
            drop_last=False, 
            pin_memory=True
            )
    print("------------validating--------------")
    start_time = time.time()
    for i, batch in enumerate(test_loader):

        chunks = batch["ppg_chunk"].to(device).to(torch.float32)
        bp = batch["bp"].to(device).to(torch.float32)
        bp_predict = model(chunks)
        l2_loss = l2_criterion(bp_predict, bp).item()

        all_loss.append(l2_loss)
        bp_sys_gt.append(torch.squeeze(bp)[0])
        bp_sys_pred.append(torch.squeeze(bp_predict)[0])
        bp_dia_gt.append(torch.squeeze(bp)[1])
        bp_dia_pred.append(torch.squeeze(bp_predict)[1])
        sessions.append(batch["session"])

        sys.stdout.write(f"\rval [{i + 1}/{len(test_loader)}] loss {l2_loss:.4f}")
        sys.stdout.flush()
    end_time = time.time()


    ckpt = torch.load(model_path5, map_location=device)
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    model.to(device)
    test_set = UWBP_test_skew(config, 5)
    test_loader = torch.utils.data.DataLoader(test_set, 
            batch_size=1, 
            shuffle=False, 
            num_workers=8, 
            drop_last=False, 
            pin_memory=True
            )
    print("------------validating--------------")
    start_time = time.time()
    for i, batch in enumerate(test_loader):

        chunks = batch["ppg_chunk"].to(device).to(torch.float32)
        bp = batch["bp"].to(device).to(torch.float32)
        bp_predict = model(chunks)
        l2_loss = l2_criterion(bp_predict, bp).item()

        all_loss.append(l2_loss)
        bp_sys_gt.append(torch.squeeze(bp)[0])
        bp_sys_pred.append(torch.squeeze(bp_predict)[0])
        bp_dia_gt.append(torch.squeeze(bp)[1])
        bp_dia_pred.append(torch.squeeze(bp_predict)[1])
        sessions.append(batch["session"])

        sys.stdout.write(f"\rval [{i + 1}/{len(test_loader)}] loss {l2_loss:.4f}")
        sys.stdout.flush()
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
    fig, axs = plt.subplots(1, 2, figsize=(30, 30), dpi=80)
    axs[0].scatter(torch.tensor(bp_sys_gt).to("cpu").numpy() * (sys_max - sys_min) + sys_min, torch.tensor(bp_sys_pred).to("cpu").numpy() * (sys_max - sys_min) + sys_min)
    axs[1].scatter(torch.tensor(bp_dia_gt).to("cpu").numpy() * (dia_max - dia_min) + dia_min, torch.tensor(bp_dia_pred).to("cpu").numpy() * (dia_max - dia_min) + dia_min)
    axs[0].set_title("sys mean: " + str(sys_mmhg)[:5] + " std: " + str(sys_mmhg_std)[:5] + " corr: " + str(sys_r_coef.to("cpu").numpy())[:5], fontsize=20)
    axs[1].set_title("dia mean: " + str(dia_mmhg)[:5] + " std: " + str(dia_mmhg_std)[:5] + " corr: " + str(dia_r_coef.to("cpu").numpy())[:5], fontsize=20)
    # axs[1].set_title("dia bp" + ": corr: " + str(dia_r_coef.to("cpu").numpy())[:5])
    # axs[0].set_xlim(0, 1)
    # axs[0].set_ylim(0, 1)
    # axs[1].set_xlim(0, 1)
    # axs[1].set_ylim(0, 1)
    axs[0].set_xlim(sys_min, sys_max)
    axs[0].set_ylim(sys_min, sys_max)
    axs[1].set_xlim(dia_min, dia_max)
    axs[1].set_ylim(dia_min, dia_max)
    axs[0].set_xlabel("Ground Truth", fontsize=15)
    axs[0].set_ylabel("Prediction", fontsize=15)
    axs[1].set_xlabel("Ground Truth", fontsize=15)
    axs[1].set_ylabel("Prediction", fontsize=15)
    axs[0].set_box_aspect(1)
    axs[1].set_box_aspect(1)
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