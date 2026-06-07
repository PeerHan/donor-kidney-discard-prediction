#!/usr/bin/env python3
"""
Executive CL Skript to make the model framework easy to use
"""

import argparse
import pandas as pd
from .OptunaHandler import OptunaHandler
from .ShapExplainer import ShapExplainer

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
parser.add_argument("-t", "--trials", type=int, default=100,
                    help="Specify the amount of tuning trials (max and default = 100.)")
parser.add_argument("-p", "--path", type=str,
                    help="Specify path to folder where artifacts are saved\n" +
                    "e.g. /mnt/.../name/DecisionTree_Exp/\n" +
                    "The Following Artifacts are saved:\n" +
                    "- model_trial_data.csv (Information about all trials)\n" +
                    "- model_study.pkl (study as joblib dump)\n" +
                    "- model_cm.png (Test CM as png)\n" +
                    "- model.pkl (Sklearn pkl or PyTorch state dict pt)\n" +
                    "- model_predictions.csv (Preds + Targets + Probs from Testing)\n" +
                    "- model_metrics.csv (random seeded train-eval cycles)\n" +
                    "- SHAP Folder (Containing Explainability Plots) if specified (default off)",
                    required=True)
parser.add_argument("-tf", "--train_file", type=str,
                    help="Path to Training Features (csv File)",
                    required=True)
parser.add_argument("-y", "--target", type=str,
                    help="Target column Name",
                    required=True)
parser.add_argument("-tsf", "--test_file", type=str,
                    help="Path to Test Features (csv File same as Training File)",
                    required=True)
parser.add_argument("-i", "--iterations", type=int, default=30,
                    help="Iterations for amount of random seeded retraining cycles\n" +
                    "Default = 30")
parser.add_argument("-d", "--device", type=str, default="cpu",
                    help="Specify if cuda or cpu should be used for mlp")
parser.add_argument('-s','--shap', type=bool,
                    help='True if SHAP should be used else False (default)',
                    default=False)
parser.add_argument('-fs','--feature_set', type=str,
                    help='Name of the corresponding Feature Set',
                    default=False)

args = parser.parse_args()
target_col = args.target
path_to_folder = args.path
algo = args.model
n_trials = args.trials
iterations = args.iterations
use_shap = args.shap
fs = args.feature_set

train_df = pd.read_csv(args.train_file)
if "SPM_ID" in list(train_df.columns):
    train_df = train_df.drop("SPM_ID", axis=1)
train_df.columns = [col.replace("dwh.Spender.Spender_", "") \
                    if "dwh.Spender.Spender_" in col else col for col in list(train_df.columns)]

test_df = pd.read_csv(args.test_file)
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

train_df.columns = [col.replace("KC_", "") \
            if "KC_" in col else col for col in list(train_df.columns)]
test_df.columns = [col.replace("KC_", "") \
            if "KC_" in col else col for col in list(test_df.columns)]

train_df.columns = [col.replace("dwh.Labor_KCBU.", "") \
            if "dwh.Labor_KCBU." in col else col for col in list(train_df.columns)]
test_df.columns = [col.replace("dwh.Labor_KCBU.", "") \
            if "dwh.Labor_KCBU." in col else col for col in list(test_df.columns)]
train_df.columns = [col.replace("dwh.Untersuchung_", "") \
            if "dwh.Untersuchung_" in col else col for col in list(train_df.columns)]
test_df.columns = [col.replace("dwh.Untersuchung_", "") \
            if "dwh.Untersuchung_" in col else col for col in list(test_df.columns)]

X_train = train_df.drop(target_col, axis=1)
y_train = train_df[target_col]
X_test = test_df.drop(target_col, axis=1)
y_test = test_df[target_col]

handler = OptunaHandler(
    x=X_train,
    y=y_train,
    algorithm=algo
)

print(f"Handler succesfully fitted with {algo}-Model")
print(f"Starting HP-Tuning with {n_trials} trials")
handler.tune_model(
    n_trials=n_trials,
    save_data=f"{path_to_folder}/{algo}_trial_data.csv",
    save_study=f"{path_to_folder}/{algo}_study.pkl",
    device=args.device,
    weight_tag=fs
)

print(f"{algo}-Model tuning finished!")
print(f"Trial data and study saved in {path_to_folder}")
print(f"Starting Retraining + Test-Eval on {iterations} different Random Seeds")

handler.test_evaluation(
    x_test=X_test,
    y_test=y_test,
    path_to_folder=path_to_folder,
    iterations=iterations,
    device=args.device,
    weight_tag=fs
)

print("Retrain + Testing finished")
print(f"Saving Test Evaluation and Model weights to {path_to_folder}")

if use_shap:

    print("Starting Explaining ...")

    se = ShapExplainer(path_to_folder,
                    X_train,
                    X_test,
                    algo,
                    fs)
    print("Model Loaded")
    se.load_model()
    se.create_explainer()
    print("Plotting Results...")
    se.create_grouped_shap_values()
    se.create_beeswarm_plot()
    se.create_featureimportance_plot()
