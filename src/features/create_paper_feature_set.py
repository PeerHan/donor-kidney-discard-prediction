#!/usr/bin/env python3

"""
Executive CL Skript to make the model framework easy to use
"""
import os
import argparse
from dotenv import load_dotenv
import pandas as pd
from .FeatureSelectionHandler import FeatureSelectionHandler # pylint: disable=E0402


load_dotenv()
parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("-m", "--model", type=str,
                    help="Specifiy which model should be tuned and tested\n" +
                    "Choose from:\n" +
                    "- 'lr' (Logistic Regression)\n" +
                    "- 'dt' (Decision Tree)\n" +
                    "- 'rf' (Random Forest)\n" + 
                    "- 'xgb' (XGBoost)\n" + 
                    "- 'mlp' (Multi Layer Perceptron; Deep Learning)",
                    required=True)
parser.add_argument("-t", "--trials", type=int, default=1000,
                    help="Amount of tuning trials (max and default = 1000)")
parser.add_argument("-j", "--jobs", type=int, default=1,
                    help="Amount of CPU cores being used in parallel (default = 1, all = -1)")
parser.add_argument("-p", "--path", type=str,
                    help="Specify path to folder where artifacts are saved\n" +
                    "e.g. /mnt/.../name/DT/\n" +
                    "The Following Artifacts are saved:\n" +
                    "- exploration_df.csv (Exploration Dataframe)\n" +
                    "- exploration_study.pkl (study as joblib dump)\n",
                    required=True)
parser.add_argument("-tf", "--train_file", type=str,
                    default=f"{os.getenv('FEATURE_DIR')}/paper/train.csv" ,
                    help="Path to Training Features (csv File)",
                    required=False)
parser.add_argument("-y", "--target", type=str,
                    help="Target column Name, default = 'Transplantationsstatus'",
                    default="Transplantationsstatus",
                    required=False)
parser.add_argument("-pen", "--penalty", type=float,
                    help="Target column Name, default = 'Transplantationsstatus'",
                    default=0.0005,
                    required=False)


args = parser.parse_args()
target_col = args.target
path_to_folder = args.path
model = args.model
n_trials = args.trials
n_jobs = args.jobs
penalty = args.penalty

train_df = pd.read_csv(args.train_file)
print(args.train_file)
test_file = args.train_file.replace("train","test")
test_df = pd.read_csv(test_file)

if "dwh.Medikation.heparinoide" in list(train_df.columns):
    print("Removing Hepa Col")
    train_df = train_df.drop(["dwh.Medikation.heparinoide"], axis=1)
if "dwh.Medikation.heparinoide" in list(test_df.columns):
    print("Removing Hepa Col")
    test_df = test_df.drop(["dwh.Medikation.heparinoide"], axis=1)

if "SPM_ID" in train_df.columns:
    train_df = train_df.drop(columns="SPM_ID", axis=1)
if "SPM_ORGAN_ID" in train_df.columns:
    train_df = train_df.drop(columns="SPM_ORGAN_ID", axis=1)
X_train = train_df.drop(target_col, axis=1).reset_index(drop=True)
y_train = train_df[target_col].reset_index(drop=True)

features = list(X_train.columns)


handler = FeatureSelectionHandler(
    model=model,
    features=features,
    x=X_train,
    y=y_train,
    penalty=penalty,
)

best_columns = handler.start_selection(save_dir=path_to_folder,n_trials=n_trials,n_jobs=n_jobs)

best_columns.append(target_col)
print(f"Best columns: {best_columns}")

data_path = f"{os.getenv('FEATURE_DIR')}/FS/{model.upper()}" # pylint: disable=C0103

if not os.path.exists(data_path):
    os.makedirs(data_path)

train_df[best_columns].to_csv(f"{data_path}/train.csv", index=False)
test_df[best_columns].to_csv(f"{data_path}/test.csv", index=False)
