#!/usr/bin/env python3
"""
Create model comparison calibration plots for paper-free dataset
"""

import os
import pickle
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn.functional import softmax
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
import seaborn as sns
from .optuna_toolkit import create_mlp_from_result

# Set style
sns.set_style("whitegrid", {'axes.grid': False})

class MLPWrapper(BaseEstimator, ClassifierMixin):
    """Wrapper to use MLP from PyTorch in sklearn methods"""
    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = device
        self.is_fitted_ = False
        self.classes_ = np.array([0, 1])

    def fit(self, _, y):
        """Fit method to comply with sklearn interface"""
        self.is_fitted_ = True
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, x):
        """Predict proba method to comply with sklearn interface"""
        check_is_fitted(self, "is_fitted_")
        train_tensor = torch.tensor(x.astype(float), dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            out = softmax(self.model(train_tensor), dim=1)
        return out.numpy()


def clean_df(df):
    """Clean DataFrame by removing SPM_ID and renaming columns"""
    if "SPM_ID" in list(df.columns):
        df = df.drop("SPM_ID", axis=1)
    df.columns = [col.replace("dwh.Spender.Spender_", "")
                  if "dwh.Spender.Spender_" in col else col for col in list(df.columns)]

    rename_benchmark = {"dwh.Spender.Spender_Geschlecht": "Geschlecht",
                        "dwh.Spender.Spender_Alter": "Alter",
                        "dwh.Spender.Spender_BMI": "BMI"}
    df.rename(rename_benchmark, axis=1, inplace=True)
    df.columns = [col.replace("Tod_ICD_10_Code", "ICD")
                  if "Tod_ICD_10_Code" in col else col for col in list(df.columns)]

    df.columns = [col.replace("dwh.Labor_Virologie.", "")
                  if "dwh.Labor_Virologie." in col else col for col in list(df.columns)]

    df.columns = [col.replace("KC_", "")
                  if "KC_" in col else col for col in list(df.columns)]

    df.columns = [col.replace("dwh.Labor_KCBU.", "")
                  if "dwh.Labor_KCBU." in col else col for col in list(df.columns)]
    df.columns = [col.replace("dwh.Untersuchung_", "")
                  if "dwh.Untersuchung_" in col else col for col in list(df.columns)]
    df.columns = [col.replace("dwh.Untersuchung_", "")
                  if "dwh.Untersuchung_" in col else col for col in list(df.columns)]
    return df

def load_and_predict_all(algorithm, base_path="/mnt/nextcloud"): # pylint: disable=R0914
    """
    Load model and make predictions for all calibration methods -
    same preprocessing as CalibrationComparison.py
    """
    model_dir = f"{base_path}/model_results_v3/paper-free/{algorithm.upper()}"
    data_dir = f"{base_path}/data_v11/paper-free/{algorithm.upper()}"

    # Load and prepare data
    train_df = pd.read_csv(f"{data_dir}/train.csv")
    train_df = clean_df(train_df)
    test_df = pd.read_csv(f"{data_dir}/test.csv")
    test_df = clean_df(test_df)

    # Split train into train/val
    train_split, val_split = train_test_split(train_df, test_size=0.1, random_state=42)

    x_train = train_split.drop("Transplantationsstatus", axis=1)
    y_train = train_split["Transplantationsstatus"]
    x_val = val_split.drop("Transplantationsstatus", axis=1)
    y_val = val_split["Transplantationsstatus"]
    x_test = test_df.drop("Transplantationsstatus", axis=1)
    y_test = test_df["Transplantationsstatus"]

    # Reset indices
    x_train = x_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    x_val = x_val.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)
    x_test = x_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # Load scaler
    with open(f"{model_dir}/train_scaler.pkl", "rb") as file:
        scaler = pickle.load(file)

    # Scale data
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    # Load model
    if algorithm.lower() == "mlp":
        mlp_df = pd.read_csv(f"{model_dir}/MLP_trial_data.csv")
        res = create_mlp_from_result(mlp_df, input_feats=x_train.shape[1], output_feats=2)
        model = res["model"]
        model.load_state_dict(torch.load(f"{model_dir}/MLP.pt", weights_only=True))
        # Transform x_train for MLP fitting
        x_train = scaler.transform(x_train)
        model = MLPWrapper(model)
        model.fit(x_train, y_train)  # Fit after wrapping
    else:
        model = joblib.load(f"{model_dir}/{algorithm.upper()}.pkl")

    # Uncalibrated predictions
    pred_uncal = model.predict_proba(x_test)[:, 1]

    # Calibrated models
    cali_model_sigmoid = CalibratedClassifierCV(estimator=model, method="sigmoid", cv="prefit")
    cali_model_isotonic = CalibratedClassifierCV(estimator=model, method="isotonic", cv="prefit")

    cali_model_sigmoid.fit(x_val, y_val)
    cali_model_isotonic.fit(x_val, y_val)

    # Calibrated predictions
    pred_sigmoid = cali_model_sigmoid.predict_proba(x_test)[:, 1]
    pred_isotonic = cali_model_isotonic.predict_proba(x_test)[:, 1]

    # Calculate calibration curves
    prob_true, prob_pred = calibration_curve(y_test, pred_uncal, n_bins=10)
    prob_cali_true, prob_cali_pred = calibration_curve(y_test, pred_sigmoid, n_bins=10)
    prob_cali_true_iso, prob_cali_pred_iso = calibration_curve(y_test, pred_isotonic, n_bins=10)

    # Calculate Brier scores
    brier_uncal = brier_score_loss(y_test, pred_uncal)
    brier_sigmoid = brier_score_loss(y_test, pred_sigmoid)
    brier_isotonic = brier_score_loss(y_test, pred_isotonic)

    return {
        'uncal': (prob_pred, prob_true, brier_uncal),
        'sigmoid': (prob_cali_pred, prob_cali_true, brier_sigmoid),
        'isotonic': (prob_cali_pred_iso, prob_cali_true_iso, brier_isotonic)
    }

def main():
    """Main function to create comparison plots"""

    # Define models
    algorithms = ["DT", "LR", "RF", "XGB", "MLP"]

    # Create output directory
    output_dir = "/mnt/nextcloud/model_results_v3/paper-free/calibration_comparison"
    os.makedirs(output_dir, exist_ok=True)

    # Storage for all results
    results = {}

    # Process each algorithm
    for algorithm in algorithms:
        print(f"Processing {algorithm}...")
        results[algorithm] = load_and_predict_all(algorithm)
        print(f"{algorithm} - Success")


    # Create uncalibrated comparison plot
    plt.figure(figsize=(10, 5), dpi=250)
    for algo in algorithms:
        if algo in results:
            prob_pred, prob_true, brier_score = results[algo]['uncal']
            plt.plot(prob_pred, prob_true, marker='o', label=f'{algo}: {brier_score:.4f}')

    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Empiric Probability')
    plt.title('Calibration Curves\nModel Comparison')
    plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(f"{output_dir}/calibration_plot_comparison.png", dpi=250, bbox_inches="tight")
    plt.close()

    # Create sigmoid calibrated comparison plot
    plt.figure(figsize=(10, 5), dpi=250)
    for algo in algorithms:
        if algo in results:
            prob_cali_pred, prob_cali_true, cali_brier_score = results[algo]['sigmoid']
            plt.plot(prob_cali_pred, prob_cali_true,
                     marker='o', label=f'{algo}: {cali_brier_score:.4f}')

    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Empiric Probability')
    plt.title('Calibration Curves after Sigmoid Calibration\nModel Comparison')
    plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(f"{output_dir}/calibration_plot_comparison_sigmoid.png",
                dpi=250, bbox_inches="tight")
    plt.close()

    # Create isotonic calibrated comparison plot
    plt.figure(figsize=(10, 5), dpi=250)
    for algo in algorithms:
        if algo in results:
            prob_cali_pred_iso, prob_cali_true_iso, cali_brier_score_iso = results[algo]['isotonic'] # pylint: disable=C0301
            plt.plot(prob_cali_pred_iso, prob_cali_true_iso,
                     marker='o', label=f'{algo}: {cali_brier_score_iso:.4f}')

    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Empiric Probability')
    plt.title('Calibration Curves after Isotonic Calibration\nModel Comparison')
    plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(f"{output_dir}/calibration_plot_comparison_isotonic.png",
                dpi=250, bbox_inches="tight")
    plt.close()

    print("\nAll calibration comparison plots created successfully!")
    print(f"Plots saved to: {output_dir}")

if __name__ == "__main__":
    main()
