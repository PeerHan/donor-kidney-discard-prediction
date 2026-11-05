#!/usr/bin/env python3
# pylint: disable=C0103
"""
Explain Shap
"""

import os
import pickle

import shap # pylint: disable=E0401
import numpy as np
import joblib
import pandas as pd
from torch.nn.functional import softmax
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from .optuna_toolkit import create_mlp_from_result
shap.initjs()

class ShapExplainer: # pylint: disable=R0902
    """
    Class to create Explainable AI with trained models based on SHAP values.

    Attributes:
        model_dir (str): Directory where model files and SHAP outputs are stored.
        x_train (pd.DataFrame): Training feature set.
        x_test (pd.DataFrame): Testing feature set.
        algorithm (str): Type of algorithm ('mlp', 'dt', 'lr', 'rf', 'xgb').
        feature_set (str): Feature set category ('nlp', 'benchmark', 'domain', 'exploration').
        model (Any): Loaded model for prediction.
        scaler (Any): Scaler used for transforming data.
        test_explainer (shap.Explanation): SHAP explanations for test data.
        grouped_shap_values (pd.DataFrame): Grouped SHAP values for features.
        shap_cmap (LinearSegmentedColormap): Custom color map for SHAP plots.
    """

    def __init__(self, model_dir, x_train, x_test, algorithm, feature_set, de=False): # pylint: disable=R0913, R0917
        """
        Initializes the ShapExplainer class.

        Args:
            model_dir (str): Directory where model files are stored.
            x_train (pd.DataFrame): Training feature set.
            x_test (pd.DataFrame): Testing feature set.
            algorithm (str): Algorithm type ('mlp', 'dt', 'lr', 'rf', 'xgb').
            feature_set (str): Feature set category ('nlp', 'benchmark', 'domain', 'exploration').

        Raises:
            AssertionError: If the algorithm or feature set is invalid.
        """

        algorithm = algorithm.strip().lower()
        feature_set = feature_set.strip().lower()
        #assert algorithm in ["mlp", "dt", "lr", "rf", "xgb"]
        #assert feature_set in ["nlp", "benchmark", "domain", "exploration"]
        self.model_dir = model_dir

        if "SPM_ID" in x_train.columns:
            x_train = x_train.drop(columns=["SPM_ID"])
        if "SPM_ID" in x_test.columns:
            x_test = x_test.drop(columns=["SPM_ID"])
        self.x_train = x_train
        self.x_test = x_test
        self.algorithm = algorithm
        self.model = None
        self.scaler = None
        self.test_explainer = None
        # if file exists, load it
        self.grouped_shap_values = None
        self.de = de
        if os.path.exists(f"{model_dir}/SHAP/grouped_shap_values.csv"):
            self.grouped_shap_values = pd.read_csv(f"{model_dir}/SHAP/grouped_shap_values.csv")

        self.feature_set = feature_set
        if not os.path.exists(f"{model_dir}/SHAP"):
            os.makedirs(f"{model_dir}/SHAP")
        self.shap_cmap = mcolors.LinearSegmentedColormap.from_list(
            "custom_RdBu_Lila", 
            ["#008bfb", "#5b5dde", "#8a3cc0", "#b10ea4", "#d7008d", "#ff0052"],
            N=256
        )

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

    def mlp_forward(self, x):
        """
        Performs forward pass and predicts probabilities for MLP models.

        Args:
            X (np.ndarray): Input features.

        Returns:
            np.ndarray: Predicted probabilities.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        train_tensor = torch.tensor(x, dtype=torch.float32).to(device)
        self.model.eval().to(device)
        with torch.no_grad():
            out = softmax(self.model(train_tensor), dim=1)
        out = out.to("cpu")
        return out.numpy()

    def create_explainer(self):
        """
        Creates the SHAP explainer for the loaded model using the training data.

        Sets the `test_explainer` attribute with SHAP values for the test set.
        """
        max_evals = 500\
            if "domain" in self.feature_set else self.x_train.columns.shape[0] * 2 + 1
        explainer = shap.Explainer(self.mlp_forward if self.algorithm == "mlp" \
                                   else self.model.predict_proba,
                                   self.scaler.transform(self.x_train),
                                   max_evals=max_evals,
                                   feature_names=self.x_train.columns,
                                   algorithm="auto")
        test_explainer = explainer(self.scaler.transform(self.x_test))
        self.test_explainer = test_explainer
        self.explainer = explainer # pylint: disable=W0201


    def create_grouped_shap_values(self):
        """
        Groups SHAP values by predefined feature groups based on the feature set.

        For 'benchmark' feature set, groups features like BMI, age, gender, etc.
        Updates `grouped_shap_values` with grouped SHAP values.
        """
        grouped_features = {col : [col] for col in self.x_train.columns}
        shap_values_grouped = pd.DataFrame()
        if self.de:
            shap_values_grouped = pd.DataFrame(self.test_explainer.values[:, :], # pylint: disable=E1101
                                           columns=self.x_train.columns)
        else:
            shap_values_grouped = pd.DataFrame(self.test_explainer.values[:, :, 1], # pylint: disable=E1101
                                           columns=self.x_train.columns)
        grouped_shap_values = pd.DataFrame()
        for group_name, columns in grouped_features.items():
            grouped_shap_values[group_name] = shap_values_grouped[columns].sum(axis=1)

        self.grouped_shap_values = grouped_shap_values
        grouped_shap_values.to_csv(f"{self.model_dir}/SHAP/grouped_shap_values.csv", index=False)


    def create_waterfall_plot(self, test_index, max_display=10):
        """
        Generates and saves a SHAP waterfall plot for a specific test observation.

        Args:
            test_index (int): Index of the test observation to explain.
            max_display (int, optional): Maximum number of features to display.
            Defaults to 5.
        """
        plt.figure(dpi=150)
        shap.plots.waterfall(self.test_explainer[test_index, :],
                                show=0, max_display=max_display)
        plt.title(f"SHAP Werte zur Erklärung einer Test Beobachtung für {self.algorithm.upper()}"+
                  f" auf dem {self.feature_set.title()}-Feature Set")
        plt.xlabel("Vorhersage Wahrscheinlichkeit")
        plt.savefig(f"{self.model_dir}/SHAP/waterfall_plot.png",
                    dpi=250,
                    bbox_inches="tight")
        plt.close()


    def create_beeswarm_plot(self, max_display=10):
        """
        Generates and saves a SHAP beeswarm plot for global explainability.

        Args:
            max_display (int, optional): Maximum number of features to display. Defaults to 5.
        """
        plt.figure(dpi=150)
        if self.de:
            shap.summary_plot(self.test_explainer[:, :],
                          max_display=max_display, show=0)
        else:
            shap.summary_plot(self.test_explainer[:, :, 1],
                          max_display=max_display, show=0)

        plt.title(f"Globale Test Explainability für {self.algorithm.upper()} " +
                  f"auf dem {self.feature_set.title()}-Feature Set")
        plt.xlabel("SHAP Wert")
        plt.yticks(fontsize=10)

        plt.savefig(f"{self.model_dir}/SHAP/Beeswarm_plot.png",
                    dpi=250,
                    bbox_inches="tight")
        plt.close()

    def create_featureimportance_plot(self, max_feats=25):
        """
        Generates and saves a global feature importance plot using grouped SHAP values.

        Highlights the absolute SHAP values for each feature group.
        """
        plt.figure(figsize=(15, 5), dpi=150)
        grouped_values = self.grouped_shap_values.abs()
        s = grouped_values.sum()
        grouped_values = grouped_values[s.sort_values(ascending=False).index[:max_feats]]
        sns.boxplot(
            data=grouped_values,
            notch=True, showcaps=False,
            showfliers=False,
            boxprops={"facecolor": (.3, .5, .7, .5)},
            medianprops={"color": "r", "linewidth": 2},
        )
        plt.title(f"Globale Feature Importance für {self.algorithm.upper()}" +
                  f" auf dem {self.feature_set.title()}-Feature Set")
        plt.ylabel("Absoluter SHAP Wert")
        plt.xticks(fontsize=14, rotation=90)
        plt.savefig(f"{self.model_dir}/SHAP/Featureimportance_plot.png",
                    dpi=250,
                    bbox_inches="tight")
        plt.close()
