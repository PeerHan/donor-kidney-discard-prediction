#!/usr/bin/env python3
"""
DSO Ensemble
"""

import os
import warnings
import joblib
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, matthews_corrcoef, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_class_weight

from .optuna_toolkit import (create_lr_from_result,
                            create_mlp_from_result, create_rf_from_result,
                            create_xgb_from_result)
from .machine_learning_toolkit import (test_dt, test_model, TorchDataSet,
                                      initialize_weights, train_model)
from .ShapExplainer import ShapExplainer

warnings.filterwarnings("ignore")

class DataEnsemble:
    """
    Ensemble Class using the following Models:
    - Decision Tree
    - Random Forest
    - Logistic Regression
    - eXtreme Gradient Boosting
    - Neural Networks

    The Predictions is based on the mean
    value of the Output probabilities of
    the models.

    For each model, the best found model
    through previous experiments is used.

    All Artifacts are stored in:
    path/model_results/DE/
    (DE=DSO Ensemble)
    """

    def __init__(self, path_to_models, path_to_data, feature_set):
        """
        Parameters:
         - feature_set: Used feature_set
         - path_to_models : path/to/models/
         - path_to_data : path/to/data/
        """
        algos = ['RF', "LR", "XGB"]
        self.algos = algos
        self.fs = feature_set
        path_to_models = path_to_models + "/" + feature_set + "/"

        self.path_to_models = path_to_models
        self.feature_set = feature_set

        path_to_data = path_to_data + "/"+ feature_set + "/"
        self.train_dfs = {algo : pd.read_csv(path_to_data + f"{algo}/" + 'train.csv')\
                            for algo in algos + ["MLP"]}
        self.test_dfs = {algo : pd.read_csv(path_to_data + f"{algo}/" + 'test.csv')\
                            for algo in algos + ["MLP"]}
        for algo, _ in self.train_dfs.items():
            train_df = self.train_dfs[algo]
            test_df = self.test_dfs[algo]
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
            self.train_dfs[algo] = train_df
            self.test_dfs[algo] = test_df
        self.mlp_model = ShapExplainer(
        self.path_to_models + "MLP",
        self.train_dfs["MLP"].drop("Transplantationsstatus", axis=1),
        self.test_dfs["MLP"].drop("Transplantationsstatus", axis=1),
        "MLP",
        feature_set
    )
        self.mlp_model.load_model()
        self.mlp_scaler = joblib.load(self.path_to_models + "MLP/train_scaler.pkl")


        self.df_train = train_df.drop("Transplantationsstatus", axis=1)
        self.target_train = train_df.Transplantationsstatus
        self.df_test = test_df.drop("Transplantationsstatus", axis=1)
        self.target_test = test_df.Transplantationsstatus
        self.mlp_model = ShapExplainer(
            self.path_to_models + "MLP",
            self.df_train,
            self.df_test,
            "MLP",
            feature_set
        )
        self.mlp_model.load_model()
        self.mlp_scaler = joblib.load(self.path_to_models + "MLP/train_scaler.pkl")



        if not os.path.exists(f"{path_to_models}DE"):
            os.makedirs(f"{path_to_models}DE")

        self.models = []
        self.scalers = []

        for algo in algos:
            model_path = path_to_models + algo + "/" + algo + ".pkl"
            self.models.append(joblib.load(model_path))
            scaler_path = path_to_models + algo + "/train_scaler.pkl"
            self.scalers.append(joblib.load(scaler_path))


    def predict_proba(self, to_predict=None, ret_dict=False):
        """
        Prediction on a dataset. Predict the probabilities of each model and takes the mean.
        Attributes:
            - to_predict: Dataset to predict on, if empty, it takes the test dataset 
                from the constructor
            - ret_dict: Return the prediction as a dictionary where each prediction gehts 
                a dict with the model name and the probability
        """
        if to_predict is None:
            to_predict = self.df_test

        predictions_list = [{} for _ in range(len(to_predict))]
        probas = []

        for model, scaler in zip(self.models, self.scalers):
            to_predict_scaled = scaler.transform(to_predict)
            probas_model = []
            model_probas = model.predict_proba(to_predict_scaled)
            for i, row in enumerate(model_probas):
                predictions_list[i][model.__class__.__name__] = row[1]
                probas_model.append(row[1])

            probas.append(probas_model)

        # MLP
        probas_model = []
        to_predict_scaled = self.mlp_scaler.transform(to_predict)
        model_probas = self.mlp_model.mlp_forward(to_predict_scaled)
        for i, row in enumerate(model_probas):
            predictions_list[i]["MLP"] = row[1]
            probas_model.append(row[1])
        probas.append(probas_model)

        probas_t = np.array(probas).T
        mean_probas = np.mean(probas_t, axis=1)
        for i, row in enumerate(mean_probas):
            predictions_list[i]["DSOEnsemble"] = row

        if ret_dict:
            return mean_probas, predictions_list
        return mean_probas

    def predict(self, to_predict=None, threshold=0.5):
        """
        Prediction on a dataset. Predict the probabilities of each model and takes the mean.
        Then makes it 1 above the threshold
        Attributes:
        - to_predict: Dataset to predict on, if empty, it takes the test dataset 
                from the constructor
        - threshold: Threshold to decide if the prediction is 1 or 0 above which value
        """
        probas = self.predict_proba(to_predict=to_predict) >= threshold
        return np.array(probas).astype(int)

    def evaluate_ensemble(self, n_predictions=30): 
        """
        Retrain the found hp-set for each model architecture on 
        n different Random Seeds.

        For each Random Seed the mean of the predictions is taken
        to predict on the test data.

        The results are saved under
        path/model_results/DE/DE_metrics.csv and
        path/model_Results/DE/DE_predictions.csv

        Parameters:
         - n_predictions: Amount of random seeds
         - drop_cols: Columns to drop
        """

        model_map = {
            "MLP" : create_mlp_from_result,
            "RF" : create_rf_from_result,
            "XGB" : create_xgb_from_result,
            "LR" : create_lr_from_result
        }

        res_probs = {}

        for algo in ["RF", "LR", "XGB", "MLP"]:
            print(f"Evaluating {algo}")
            res_probs[algo] = {}
            model_funct = model_map[algo]
            model_df = pd.read_csv(f"{self.path_to_models}{algo}/{algo}_trial_data.csv")
            if "exploration" in self.fs or "paper" in self.fs:
                train = self.train_dfs[algo]
                test = self.test_dfs[algo]
                x_train, y_train = train.drop("Transplantationsstatus", axis=1),\
                    train.Transplantationsstatus
                x_test, y_test = test.drop("Transplantationsstatus", axis=1),\
                    test.Transplantationsstatus
            else:
                x_train, y_train = self.df_train.copy(), self.target_train.copy()
                x_test, y_test = self.df_test.copy(), self.target_test.copy()

            if algo != "MLP":

                for state in range(n_predictions):
                    print(f"Random state: {state}")
                    res = model_funct(model_df,
                                      input_feats=x_train.shape[1],
                                      output_feats=len(y_train.unique()))
                    model = res["model"]
                    model.random_state = state
                    scaler = StandardScaler()
                    x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                                    y_train,
                                                                    test_size=0.1,
                                                                    random_state=42)
                    if algo == "XGB":
                        x_train = scaler.fit_transform(x_train)
                        x_val = scaler.transform(x_val)
                        if res["use_class_weights"]:
                            weight = y_train.value_counts()[0] / y_train.value_counts()[1]
                            model.scale_pos_weight = weight
                        model.fit(x_train,y_train,eval_set=[(x_val,y_val)],verbose=False)
                    else:
                        x_train = scaler.fit_transform(x_train)
        
                        if pd.isna(model.class_weight):
                            model.class_weight = None
                        model.fit(x_train, y_train)
                    x_test = scaler.transform(x_test)
                    _, _, _, _, _, _, probs = test_dt(model, x_test, y_test)
                    res_probs[algo][state] = probs
            else:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                                  y_train,
                                                                  test_size=0.1,
                                                                  random_state=42)
                x_train = x_train.reset_index(drop=True)
                y_train = y_train.reset_index(drop=True)
                x_test= x_test.reset_index(drop=True)
                y_test = y_test.reset_index(drop=True)
                x_val = x_val.reset_index(drop=True)
                y_val = y_val.reset_index(drop=True)
                model_funct = model_map[algo]
                res = model_funct(model_df,
                                 input_feats=x_train.shape[1],
                                 output_feats=len(y_train.unique()))

                train_dataset = TorchDataSet(x_train, y_train)
                train_scaler = train_dataset.scaler
                val_dataset = TorchDataSet(x_val, y_val, train_scaler)
                test_dataset = TorchDataSet(x_test, y_test, train_scaler)
                for state in range(n_predictions):
                    print(f"Random state: {state}")
                    model = res["model"]
                    model.apply(initialize_weights)
                    optimizer = torch.optim.AdamW(model.parameters(),
                                                 lr=res["init_lr"],
                                                 weight_decay=res["weight_decay"])
                    train_loader = DataLoader(train_dataset, batch_size=512,
                                              shuffle=True)
                    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
                    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset),
                                             shuffle=False)
                    weights = compute_class_weight(class_weight="balanced",
                                                classes=np.unique(y_train),
                                                y=y_train)
                    criterion = nn.CrossEntropyLoss(torch.tensor(weights, dtype=torch.float32)) if \
                        res["class_weights"] else nn.CrossEntropyLoss()
                    train_model(model,
                                optimizer,
                                train_loader,
                                val_loader,
                                criterion,
                                num_classes=np.unique(y_train).shape[0],
                                scheduler_pat=5,
                                es_pat=5,
                                max_epochs=res["max_epochs"],
                                device=device,
                                es_delta=0,
                                show_every=None)

                    _, _, _, _, _, _, probs = test_model(model,
                                                         test_loader,
                                                         device=device,
                                                         num_classes=np.unique(y_train).shape[0],
                                                         return_softmax=True)
                    res_probs[algo][state] = probs
        seed_prob_dict = {seed : [] for seed in range(len(res_probs["RF"]))}

        for algo, seed_dict in res_probs.items():

            for seed, probs in seed_dict.items():
                prob_df = pd.DataFrame(probs)
                seed_prob_dict[seed].append(prob_df)

        ensemble_res = {}
        best_mcc = 0
        for seed, dfs in seed_prob_dict.items():
            ensemble_res[seed] = {}

            seed_df = sum(dfs) / len(dfs)
            preds = seed_df[1] >= 0.5
            prob_df = seed_df.copy()
            prob_df["Targets"] = y_test
            prob_df["Prediction"] = preds
            ensemble_res[seed]["F1 Test"] = f1_score(y_test, preds)
            ensemble_res[seed]["Normed MCC Test"] = (matthews_corrcoef(y_test, preds) + 1) / 2
            ensemble_res[seed]["AUC Test"] = roc_auc_score(y_true=y_test,
                                                           y_score=seed_df[1])
            ensemble_res[seed]["Model"] = "DE"
            ensemble_res[seed]["Iteration"] = seed
            if ensemble_res[seed]["Normed MCC Test"] > best_mcc:
                best_mcc = ensemble_res[seed]["Normed MCC Test"]
                prob_df = prob_df.rename({0 : "Prob_0", 1 : "Prob_1"}, axis=1)
                prob_df.to_csv(f"{self.path_to_models}DE/DE_predictions.csv", index=False)
                plt.figure(figsize=(8, 8), dpi=250)
                sns.set_style("darkgrid")
                cm_test = confusion_matrix(y_true=y_test, y_pred=preds)
                sns.heatmap(cm_test,
                            cmap="viridis",
                            annot=True,
                            fmt=".0f",
                            xticklabels=["Nicht Transplantiert", "Transplantiert"],
                            yticklabels=["Nicht Transplantiert", "Transplantiert"],
                            annot_kws={"fontsize":30})
                plt.xticks(fontsize=15)
                plt.yticks(fontsize=15)
                plt.xlabel("Vorhersage", fontsize=20)
                plt.ylabel("Echtes Label", fontsize=20)
                plt.title("Ensemble Performance auf dem Testdatensatz\n" +
                          f"F1 Score: {f1_score(y_test, preds):.2} " +
                          f"| MCC: {(matthews_corrcoef(y_test, preds) + 1) / 2:.2}")
                plt.savefig(f"{self.path_to_models}DE/DE_cm.png", bbox_inches="tight", dpi=250)
                plt.close()
        ensemble_df = pd.DataFrame(ensemble_res).T
        ensemble_df.to_csv(f"{self.path_to_models}DE/DE_metrics.csv", index=False)
