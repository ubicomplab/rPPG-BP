#!/bin/bash

ORI_TSV_PATH="/gscratch/ubicomp/cm74/MS_aurorabp/features.tsv"
PROCESSED_CSV_PATH="./auro_data_ambulatory.csv"

RAW_DATA_FOLDER="/gscratch/ubicomp/cm74/MS_aurorabp/measurements_oscillometric"
PROCESSED_DATA_FOLDER="./measurements_oscillometric_ambulatory_preprocessed"
SELECETED_DATA_FOLDER="./measurements_oscillometric_ambulatory_seleceted"

# First filter the sessions by class and signal quality
python preprocess_tsv.py --tsv_path $ORI_TSV_PATH --csv_path $PROCESSED_CSV_PATH

# Second filter the signal and resampling
python preprocess_ppg.py --csv_path $PROCESSED_CSV_PATH --raw_data $RAW_DATA_FOLDER --processed_data $PROCESSED_DATA_FOLDER

# Third beats segmentation and template matching
python select_ppg.py --processed_data $PROCESSED_DATA_FOLDER --selected_data $SELECETED_DATA_FOLDER

