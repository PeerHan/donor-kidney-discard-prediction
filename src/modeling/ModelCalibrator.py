#!/usr/bin/env python3 pylint: disable=C0103
"""
Calibrate Model for more accurate representation of prediction probabilities.
"""

import os
import pickle

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from torch.nn.functional import softmax
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from .optuna_toolkit import create_mlp_from_result

sns.set_style("whitegrid", {'axes.grid' : False})


class ModelCalibrator: # pylint: disable=R0902
    """
    Class to calibrate models for a more accurate representation of prediction probabilities.    
    
    Attributes:
        model_dir (str): Directory where model files are stored.
        train (pd.DataFrame): Training feature set.
        test (pd.DataFrame): Testing feature set.
        algorithm (str): Type of algorithm ('mlp', 'dt', 'lr', 'rf', 'xgb').
        feature_set (str): Feature set category ('nlp', 'benchmark', 'domain', 'exploration').
        model (Any): Loaded model for prediction.
        scaler (Any): Scaler used for transforming data.
    """
    def __init__(self, model_dir, train, test, algorithm):
        """
        Initializes the ModelCalibrator class.

        Args:
            model_dir (str): Directory where model files are stored.
            train (pd.DataFrame): Training feature set.
            test (pd.DataFrame): Testing feature set.
            algorithm (str): Algorithm type ('mlp', 'dt', 'lr', 'rf', 'xgb').

        Raises:
            AssertionError: If the algorithm or feature set is invalid.
        """

        algorithm = algorithm.strip().lower()
        assert algorithm in ["mlp", "dt", "lr", "rf", "xgb"]
        self.model_dir = model_dir

        self.train, self.val = train_test_split(train, test_size=0.1, random_state=42)
        self.test = test
        self.x_train = self.train.drop("Transplantationsstatus", axis=1)
        self.y_train = self.train.Transplantationsstatus
        self.x_val = self.val.drop("Transplantationsstatus", axis=1)
        self.y_val = self.val.Transplantationsstatus
        self.x_test = self.test.drop("Transplantationsstatus", axis=1)
        self.y_test = self.test.Transplantationsstatus
        self.x_train = self.x_train.reset_index(drop=True)
        self.y_train =  self.y_train.reset_index(drop=True)
        self.x_val = self.x_val.reset_index(drop=True)
        self.y_val =  self.y_val.reset_index(drop=True)
        self.x_test = self.x_test.reset_index(drop=True)
        self.y_test = self.y_test.reset_index(drop=True)
        self.algorithm = algorithm
        self.model = None
        self.scaler = None
        self.test_explainer = None
        # if file exists, load it
        if not os.path.exists(f"{model_dir}/Calibrated"):
            os.makedirs(f"{model_dir}/Calibrated")

    def load_model(self):
        """
        Loads the trained model and scaler.

        If the algorithm is 'mlp', it builds the MLP model structure and loads weights.
        For other algorithms, it loads the serialized model directly.
        """

        with open(f"{self.model_dir}/train_scaler.pkl", "rb") as file:
            self.scaler = pickle.load(file)
        if self.algorithm == "mlp":
            mlp_df = pd.read_csv(f"{self.model_dir}/MLP_trial_data.csv")
            res = create_mlp_from_result(mlp_df, input_feats=self.x_train.shape[1], output_feats=2)
            model = res["model"]
            model.load_state_dict(torch.load(f"{self.model_dir}/MLP.pt", weights_only=True))
            self.model = model
        else:
            self.model = joblib.load(f"{self.model_dir}/{self.algorithm.upper()}.pkl")


    def create_calibration_plots(self):
        """"
        Creates calibration plots and saves calibrated model.
        """
        self.load_model()
        self.x_val = self.scaler.transform(self.x_val)
        self.x_test = self.scaler.transform(self.x_test)

        # Falls MLP, wrappe das Modell direkt nach dem Laden
        if self.algorithm == "mlp":
            self.model = MLPWrapper(self.model)
            self.model.fit(self.x_train,self.y_train)

        # Berechne Vorhersagen vor der Kalibrierung
        pred = self.model.predict_proba(self.x_test)[:, 1]

        # Kalibriertes Modell trainieren
        cali_model = CalibratedClassifierCV(estimator=self.model,
                                            method="sigmoid", cv="prefit")
        cali_model_iso = CalibratedClassifierCV(estimator=self.model,
                                            method="isotonic", cv="prefit")
        cali_model.fit(self.x_val, self.y_val)
        cali_model_iso.fit(self.x_val, self.y_val)

        # Berechne kalibrierte Vorhersagen für MLP & andere Modelle
        cali_pred = cali_model.predict_proba(self.x_test)[:, 1]
        cali_pred_iso = cali_model_iso.predict_proba(self.x_test)[:, 1]

        # Brier Scores berechnen
        brier_score = brier_score_loss(self.y_test, pred)
        cali_brier_score = brier_score_loss(self.y_test, cali_pred)
        cali_brier_score_iso = brier_score_loss(self.y_test, cali_pred_iso)

        # Calibration Curves
        prob_true, prob_pred = calibration_curve(self.y_test, pred, n_bins=10)
        prob_cali_true, prob_cali_pred = calibration_curve(self.y_test, cali_pred, n_bins=10)
        prob_cali_true_iso, prob_cali_pred_iso = calibration_curve(self.y_test,
                                                                   cali_pred_iso, n_bins=10)

        # Plot erstellen
        plt.figure(figsize=(10, 5), dpi=250)
        plt.plot(prob_pred, prob_true, marker='x',
                 label='Pre-Calibration '+f'Brier: {brier_score:.4}')
        plt.plot(prob_cali_pred, prob_cali_true, marker='o',
                 label='Post-Calibration Sigmoid '+f'Brier: {cali_brier_score:.4}')
        plt.plot(prob_cali_pred_iso, prob_cali_true_iso, marker='o',
                 label='Post-Calibration Isotonic ' +f'Brier: {cali_brier_score_iso:.4}')
        plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Empiric Probability')
        plt.title('Calibration Curve with Brier Score\n' +
                  f'Model: {self.algorithm} ' +
                  f'Feature Set: {self.model_dir.split("/")[-2]}')
        plt.legend()
        plt.savefig(f"{self.model_dir}/Calibrated/calibration_plot.png",
                    dpi=250, bbox_inches="tight")
        plt.close()

        # Speichere das kalibrierte Modell
        joblib.dump(cali_model, f"{self.model_dir}/Calibrated/calibrated_model_sigmoid.pkl")
        joblib.dump(cali_model_iso, f"{self.model_dir}/Calibrated/calibrated_model_isotonic.pkl")


class MLPWrapper(BaseEstimator, ClassifierMixin):
    """
    Wrapper to be able to use MLP from PyTorch in sklearn method (CalibratedClassifierCV).
    """
    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = device
        self.is_fitted_ = False
        self.classes_ = np.array([0, 1])

    def fit(self, x, y): # pylint: disable=W0613
        """Sklearn erwartet eine Fit-Methode, aber für vortrainierte Modelle bleibt sie leer."""
        self.is_fitted_ = True
        self.classes_ = np.unique(y)
        return self  # Modell ist bereits trainiert

    def predict_proba(self, x):
        """Gibt die Wahrscheinlichkeiten für die Klassen zurück."""
        check_is_fitted(self, "is_fitted_")
        train_tensor = torch.tensor(x.astype(float), dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            out = softmax(self.model(train_tensor), dim=1)
        return out.numpy()
