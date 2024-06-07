#!/bin/bash

# The folder path of the original raw ppg data
RAW_DATA_PATH=/YOUR_PATH_TO_RAW_PPG_DATA
# The file path of patient information csv
INFO_CSV_PATH=./INFO.csv
# The folder path to save preprocessed ppg data
SAVED_PREPROCESSED_DATA_PATH=./PREPROCESSED_PPG_SAVED_PATH
# The folder path to save sqi filtered ppg data
SAVED_SQI_DATA_PATH=./SQI_FILTERED_PPG_SAVED_PATH

# preprocess raw ppg data to chunks
python preprocess_ppg.py --mat_path $RAW_DATA_PATH --df_path $INFO_CSV_PATH --save_path $SAVED_PREPROCESSED_DATA_PATH
# filtered ppg data by sqi
python select_ppg_sqi.py --mat_path $SAVED_PREPROCESSED_DATA_PATH --df_path $INFO_CSV_PATH --save_path $SAVED_SQI_DATA_PATH