
import numpy as np
import pandas as pd
import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="settings")
    parser.add_argument("-i", "--tsv_path", type=str, default="./features.tsv", help="input tsv file")
    parser.add_argument("-o", "--csv_path", type=str, default="./auro_data_ambulatory.csv", help="processed csv file")
    opt = parser.parse_args()
    # tsv_path = "/gscratch/ubicomp/cm74/MS_aurorabp/features.tsv"
    bp_df = pd.read_csv(opt.tsv_path, sep = '\t')
    pid = list()
    measurement = list()
    sbp = list()
    dbp = list()
    baseline_sbp = list()
    baseline_dbp = list()
    threshold = 0.9
    for i, row in bp_df.iterrows():
        if np.isnan(row["sbp"]) or row["quality_optical"] < threshold or row["phase"] != "ambulatory":
            continue
        pid.append(row["pid"])
        measurement.append(row["measurement"])
        sbp.append(row["sbp"])
        dbp.append(row["dbp"])
        baseline_sbp.append(row["baseline_sbp"])
        baseline_dbp.append(row["baseline_dbp"])
    
    df = pd.DataFrame()
    df["pid"] = pid
    df["measurement"] = measurement
    df["sbp"] = sbp
    df["dbp"] = dbp
    df["baseline_sbp"] = baseline_sbp
    df["baseline_dbp"] = baseline_dbp
    df.to_csv(opt.csv_path)