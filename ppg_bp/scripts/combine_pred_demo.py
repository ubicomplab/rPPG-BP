import pandas as pd


if __name__ == "__main__":
    result_path = "/gscratch/ubicomp/cm74/bp/ppg_bp/output/20230116_test_v1/bp_results.csv"
    demo_path = "/gscratch/ubicomp/cm74/clinical_data/ESH_Master.csv"

    df_result = pd.read_csv(result_path)
    df_demo = pd.read_csv(demo_path)

    age = list()
    gender = list()
    height = list()
    weight = list()
    bmi = list()


    for i, row in df_result.iterrows():
        pid = row["session_id"][2:-3]
       
        df = df_demo[df_demo["ID"] == pid]
        if len(df["Age"].values) < 1:
            age.append(0)
        else:
            age.append(df["Age"].values[0])
        if len(df["Gender"].values) < 1:
            gender.append(0)
        else:
            gender.append(df["Gender"].values[0])
        
        if len(df["Ht (m)"].values) < 1:
            height.append(0)
        else:
            height.append(df["Ht (m)"].values[0])
        
        if len(df["Wt (kg)"].values) < 1:
            weight.append(0)
        else:
            weight.append(df["Wt (kg)"].values[0])
        
        if len(df["BMI"].values) < 1:
            bmi.append(0)
        else:
            bmi.append(df["BMI"].values[0])
    

    df_result["age"] = age
    df_result["gender"] = gender
    df_result["height"] = height
    df_result["weight"] = weight
    df_result["bmi"] = bmi

    df_result.to_csv("bp_results_demograph.csv", index=False)