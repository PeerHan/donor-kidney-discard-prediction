#!/usr/bin/env python3

"""
Executive CL Skript to make the model framework easy to use
"""

import argparse
import pandas as pd
from .ModelCalibrator import ModelCalibrator

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("-m", "--model", type=str,
                    help="Specifiy which model should be tuned and tested\n" +
                    "Choose from:\n" +
                    "- 'lr' (Logistic Regression)\n" +
                    "- 'dt' (Decision Tree)\n" +
                    "- 'rf' (Random Forest)\n" +
                    "- 'xgb' (XGBoost)" +
                    "- 'mlp' (Multi Layer Perceptron; Deep Learning)",
                    required=True)
parser.add_argument("-p", "--path", type=str,
                    help="Specify path to folder where artifacts are currently saved\n" +
                    "e.g. /mnt/.../name/DecisionTree_Cal/\n" +
                    "The following Artifacts are expected:\n" +
                    "- train_scaler.pkl\n" + 
                    "- model.pkl\n" +
                    "if model = mlp, then also:\n" +
                    "mlp_trial_data.csv\n" +
                    "MLP.pt\n"+
                    "The Following Artifacts are saved:\n" +
                    "- calibrated_model.pkl (study as joblib dump)\n" +
                    "- calibration_plot.png",
                    required=True)
parser.add_argument("-tf", "--train_file", type=str,
                    help="Path to Training Features (csv File)",
                    required=True)
parser.add_argument("-tsf", "--test_file", type=str,
                    help="Path to Test Features (csv File same as Training File)",
                    required=True)

args = parser.parse_args()
algo = args.model
model_dir = args.path

train_df = pd.read_csv(args.train_file)
test_df = pd.read_csv(args.test_file)

if "SPM_ID" in list(train_df.columns):
    train_df = train_df.drop("SPM_ID", axis=1)
train_df.columns = [col.replace("dwh.Spender.Spender_", "") \
                    if "dwh.Spender.Spender_" in col else col for col in list(train_df.columns)]

if "SPM_ID" in list(test_df.columns):
    test_df = test_df.drop("SPM_ID", axis=1)
test_df.columns = [col.replace("dwh.Spender.Spender_", "") \
                   if "dwh.Spender.Spender_" in col else col for col in list(test_df.columns)]

rename_benchmark = {"dwh.Spender.Spender_Geschlecht" : "Geschlecht",
                    "dwh.Spender.Spender_Alter" : "Alter",
                    "dwh.Spender.Spender_BMI" : "BMI"}
train_df.rename(rename_benchmark, axis=1, inplace=True)
test_df.rename(rename_benchmark, axis=1, inplace=True)

train_df.columns = [col.replace("Tod_ICD_10_Code", "ICD") \
                if "Tod_ICD_10_Code" in col else col for col in list(train_df.columns)]
test_df.columns = [col.replace("Tod_ICD_10_Code", "ICD") \
                if "Tod_ICD_10_Code" in col else col for col in list(test_df.columns)]
train_df.columns = [col.replace("dwh.Labor_Virologie.", "") \
            if "dwh.Labor_Virologie." in col else col for col in list(train_df.columns)]
test_df.columns = [col.replace("dwh.Labor_Virologie.", "") \
            if "dwh.Labor_Virologie." in col else col for col in list(test_df.columns)]
train_df.columns = [col.replace("dwh.Labor_KCBU.", "") \
            if "dwh.Labor_KCBU." in col else col for col in list(train_df.columns)]
test_df.columns = [col.replace("dwh.Labor_KCBU.", "") \
            if "dwh.Labor_KCBU." in col else col for col in list(test_df.columns)]
train_df.columns = [col.replace("dwh.Untersuchung_", "") \
            if "dwh.Untersuchung_" in col else col for col in list(train_df.columns)]
test_df.columns = [col.replace("dwh.Untersuchung_", "") \
            if "dwh.Untersuchung_" in col else col for col in list(test_df.columns)]

train_df.columns = [col.replace("KC_", "")
                if "KC_" in col else col for col in list(train_df.columns)]
test_df.columns = [col.replace("KC_", "")
                if "KC_" in col else col for col in list(test_df.columns)]

calibrator = ModelCalibrator(model_dir, train_df, test_df, algorithm=algo)
print(f"Succesfully loaded the {algo}-Model")

calibrator.create_calibration_plots()

print("Files created successfully!")
