#!/usr/bin/env python3
"""
Toolkit for Machine Learning
"""
import os
import pickle
import copy
import joblib

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchmetrics.classification import MulticlassMatthewsCorrCoef, F1Score, Precision, Recall

from sklearn.metrics import f1_score, matthews_corrcoef, precision_score, \
    recall_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from .mlflow_logger import MlFlowLogger


sns.set_style("darkgrid")

class TorchDataSet(Dataset):
    """
    Dataset for PyTorch Models.
    The dataset uses a standard scaler for training
    """

    def __init__(self, x, y, scaler=None):
        """
        Parameter:
            - x: Feature which should be used for training
            - y: Target
            - scaler: None for training set else get the train scaler
            and use it for val/test
        """
        self.original_x = x
        self.y = y
        if scaler:
            self.scaler = scaler
            self.scaled_x = scaler.transform(x)
        else:
            self.scaler = StandardScaler()
            self.scaled_x = self.scaler.fit_transform(x)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        y = self.y[idx]
        x = self.scaled_x[idx]
        y = torch.tensor(y, dtype=torch.long)
        x = torch.tensor(x, dtype=torch.float32)
        return x, y

    def get_scaler(self):
        """
        Return the scaler with its parameters
        """
        return self.scaler

class EarlyStopping(): # pylint: disable=R0903
    """
    EarlyStopping Module for PyTorch training
    """

    def __init__(self, patience, min_delta=0.0, direction="minimize"):
        """
        Parameters:
            - patience: Determine how many scores should be tolerated til breaking
            - min_delta: Determine a delta value which can be considered to detect differences
            - direction: Minimize for loss/errors else Maximize for Scores

        """

        if patience < 1:
            raise ValueError("Argument patience should be positive integer.")
        if min_delta < 0.0:
            raise ValueError("Argument min_delta should not be a negative number.")
        assert direction in ["minimize", "maximize"], "Direction must be 'minimize' or 'maximize'"

        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.direction = direction
        self.best_score = float("inf") if direction == "minimize" else float("-inf")

    def stop_training(self, loss):
        """
        Returns a bool:
        - True: if the training should be stopped
        - False: the counter is < patience value
        """
        threshold_score = self.best_score + self.min_delta
        minimize = self.direction == "minimize"
        count_up = loss >= threshold_score if minimize else loss <= threshold_score
        if count_up:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_score = loss
            self.counter = 0
        return False

def initialize_weights(m):
    """
    Using Xavier Init For MLPs
    https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
    """
    if isinstance(m, nn.Linear) and hasattr(m, "weight"):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def train_one_epoch(model, optimizer, train_loader, criterion, device="cpu"):
    """
    Function to train one epoch with a PyTorch Model.
    Returning the average loss of an epoch.

    Parameters:
        - model: PyTorch Model
        - optimizer: PyTorch Optimizer
        - train_loader: PyTorch trainloader
        - criterion: PyTorch loss function
        - device: Compute on cuda or cpu
    """

    model.to(device)
    criterion.to(device)
    average_loss = 0.0
    model.train()

    if len(train_loader) == 0:
        return -1


    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()

        outputs = model(x)
        loss = criterion(outputs, y)

        loss.backward()
        optimizer.step()
        average_loss += loss.item()

    average_loss /= len(train_loader)

    return average_loss


def val_one_epoch(model, val_loader, criterion, device="cpu", num_classes=3): # pylint: disable=R0914
    """
    Function to validate on a validation dataset for one epoch.
    Returning the average val loss, average f1 score, average mcc on the val dataset

    Parameter:
        - model: PyTorch Model
        - val_loader: PyTorch Valdataset
        - criterion: PyTorch Lossfunction
        - device: Train on CPU or Cuda
        - num_classes: Amount of classes (for metrics)
    """
    average_loss = 0.0
    model.to(device)
    criterion.to(device)
    model.eval()
    f1 = F1Score(task="multiclass", num_classes=num_classes, average="weighted").to(device)
    mcc = MulticlassMatthewsCorrCoef(num_classes=num_classes).to(device)

    average_mcc = 0.
    average_f1 = 0.

    for x, y in val_loader:
        x = x.to(device)
        y = y.to(device)

        outputs = model(x)
        loss = criterion(outputs, y)
        f1_batch = f1(outputs, y)
        mcc_batch = mcc(outputs, y)

        average_loss += loss.item()
        average_mcc += mcc_batch.item()
        average_f1 += f1_batch.item()

    average_loss /= len(val_loader)
    average_mcc /= len(val_loader)
    average_f1 /= len(val_loader)

    return average_loss, average_f1, average_mcc

def train_model( # pylint: disable=R0913,R0914, R0917
        model,
        optimizer,
        train_loader,
        val_loader,
        criterion,
        scheduler_pat=5,
        es_pat=5,
        es_delta=0.0,
        show_every=5,
        device="cpu",
        max_epochs=1000,
        num_classes=3,
        temp_weight_tag="Tmp"
        ):
    """
    Train a PyTorch Model for a given amount of epochs +
    Validate on a Validationset per Epoch

    Parameters:
        - model: PyTorch Model which should be trained
        - optimizer: Used Optimizer
        - train_loader: Dataloader for training
        - val_loader: Dataloader for validation
        - criterion: Loss function
        - scheduler_pat: Patience for lr scheduler
        - es_pat: Patience for EarlyStop
        - es_delta: Delta value for EarlyStop
        - show_every: Logging - Print statement for current epoch
        - device: Cpu or Cuda 
        - max_epochs: Maximum amount of epochs to train
        - num_classes: Amount of Targets
    """
    stopper = EarlyStopping(es_pat, es_delta, direction="minimize")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           patience=scheduler_pat, mode="min")
    min_val = float("inf")
    weight_file = f"{temp_weight_tag}_Best_Val_Weights.pt"
    for epoch in range(max_epochs):
        train_loss = train_one_epoch(model, optimizer, train_loader, criterion, device=device)
        val_loss, val_f1, val_mcc = val_one_epoch(model,
                                                  val_loader,
                                                  criterion,
                                                  device=device,
                                                  num_classes=num_classes)
        if val_loss < min_val:
            min_val = val_loss
            model.to("cpu")
            torch.save(model.state_dict(), weight_file)
            model.to(device)
        if show_every and epoch % show_every == 0:
            print(f"Epoch: {epoch} | Train Loss: {train_loss:.3} | " +
                   f"Val Loss: {val_loss:.3} | Val F1: {val_f1:.3} | Val MCC: {val_mcc:.3}")     
        if stopper.stop_training(val_loss):
            model.to("cpu")
            model.load_state_dict(torch.load(weight_file, weights_only=True))
            model.to(device)
            os.remove(weight_file)
            break
        scheduler.step(val_loss)


def test_model(model, test_loader, device="cpu", # pylint: disable=R0913, R0914, R0917
               num_classes=3, return_softmax=False, mc_iter=100):
    """
    PyTorch model for testing on a test dataset.
    Returning: f1 score, mcc, y, y predictions, [softmax_scores]
    Parameters:
        - model: PyTorch Model
        - test_loader: Dataloader with testdata
        - device: CPU or Cuda
        - num_classes: Amount of unique classes to predict
        - return_softmax: Return softmax scores
    """

    f1 = F1Score(task="multiclass", num_classes=num_classes, average="weighted").to(device)
    mcc = MulticlassMatthewsCorrCoef(num_classes=num_classes).to(device)
    precision = Precision(task="multiclass", num_classes=num_classes).to(device)
    recall = Recall(task="multiclass", num_classes=num_classes).to(device)
    preds = []
    targets = []
    probs = []
    model.to(device)

    with torch.no_grad():
        model.train()

        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            all_softmax_outputs = []

            for _ in range(mc_iter):
                for module in model.modules():
                    if isinstance(module, torch.nn.BatchNorm1d):
                        module.eval()
                outputs = model(x)
                outputs = outputs.to(device)
                softmax_output = nn.functional.softmax(outputs, dim=1)
                all_softmax_outputs.append(softmax_output)
            avg_probs = torch.mean(torch.stack(all_softmax_outputs), dim=0)
            outputs = torch.argmax(avg_probs, dim=1)
            probs.append(avg_probs)
            targets.append(y)
            preds.append(outputs)

    targets = torch.cat(targets)
    preds = torch.cat(preds)
    probs = torch.cat(probs)
    f1_test = f1(preds, targets).item()
    mcc_test = mcc(preds, targets).item()
    precision_test = precision(preds, targets).item()
    recall_test = recall(preds, targets).item()
    model.eval()
    if return_softmax:
        return f1_test, mcc_test, precision_test, recall_test, \
            targets.cpu().detach().numpy(), preds.cpu().detach().numpy(), probs.cpu()
    return f1_test, mcc_test, precision_test, recall_test, \
        targets.cpu().detach().numpy(), preds.cpu().detach().numpy()

def test_dt(model, x_test, y_test):
    """
    DesicionTreeClassifier for testing on a test dataset.
    Returning: f1 score, mcc, y, y predictions
    Parameters:
        - model: DecisionTreeClassifier
        - test_loader: DataLoader-like object or tuple (X_test, y_test)
    """
    y_pred = model.predict(x_test)
    probs = model.predict_proba(x_test)
    f1_test = f1_score(y_test, y_pred, average='weighted')
    mcc_test = matthews_corrcoef(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    return f1_test, mcc_test, precision, recall, y_test, y_pred, probs


def test_xgb(model, x_test, y_test):
    """
    XGBClassifier for testing on a test dataset.
    Returning: f1 score, mcc, y, y predictions
    Parameters:
        - model: XGBClassifier
        - test_loader: DataLoader-like object or tuple (X_test, y_test)
    """
    return test_dt(model, x_test, y_test)

def test_rf(model, x_test, y_test):
    """
    RandomForestClassifier for testing on a test dataset.
    Returning: f1 score, mcc, y, y predictions
    Parameters:
        - model: RandomForestClassifier
        - test_loader: DataLoader-like object or tuple (X_test, y_test)
    """
    return test_dt(model, x_test, y_test)


def test_lr(model, x_test, y_test):
    """
    LogisticRegression for testing on a test dataset.
    Returning: f1 score, mcc, y, y predictions
    Parameters:
        - model: XGBClassifier
        - test_loader: DataLoader-like object or tuple (X_test, y_test)
    """
    return test_dt(model, x_test, y_test)

def fit_and_eval_sklearn(res, # pylint: disable=R0913,R0914,R0917, R0912, R0915
                         x_train,
                         y_train,
                         x_test,
                         y_test,
                         path_to_folder,
                         n_predictions=30,
                         logging_active=False,
                         **kwargs):
    """
    Retrain with different Random Seeds to generate Test-Eval Distribution
    """
    best_preds = None
    best_mcc = -1
    best_f1 = 0
    best_probs = None

    mcc_scores = []
    f1_scores = []
    auc_scores = []

    if not isinstance(x_train, pd.DataFrame):
        x_train = pd.DataFrame(x_train, columns=[f"feature_{i}" for i in range(x_train.shape[1])])
    if not isinstance(x_test, pd.DataFrame):
        x_test = pd.DataFrame(x_test, columns=[f"feature_{i}" for i in range(x_test.shape[1])])

    if not isinstance(y_train, pd.Series):
        y_train = pd.Series(y_train, name="target")
    if not isinstance(y_test, pd.Series):
        y_test = pd.Series(y_test, name="target")

    df_train = x_train.copy()
    df_train[y_train.name] = y_train

    df_test = x_test.copy()
    df_test[y_test.name] = y_test

    if logging_active:
        mfl = MlFlowLogger(kwargs["model"])
    best_cm = None
    for state in range(n_predictions):
        model = res["model"]
        model.random_state = state
        scaler = StandardScaler()
        x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                          y_train,
                                                          test_size=0.15,
                                                          stratify=y_train,
                                                          random_state=42)
        if kwargs["model"] == "XGB":

            x_train_scaled = scaler.fit_transform(x_train)
            x_val_scaled = scaler.transform(x_val)
            if res["use_class_weights"]:
                weight = y_train.value_counts()[0] / y_train.value_counts()[1]
                model.scale_pos_weight = weight
            model.fit(x_train_scaled,y_train,eval_set=[(x_val_scaled,y_val)],verbose=False)
        else:
            x_train_scaled = scaler.fit_transform(x_train)
            x_val_scaled = scaler.transform(x_val)
            model.fit(x_train_scaled, y_train)
        x_test_scaled = scaler.transform(x_test)
        f1_test, mcc_test, precision, recall, targets, preds, probs = \
            test_dt(model, x_test_scaled, y_test)

        cm = confusion_matrix(targets, preds)
        auc_value = roc_auc_score(y_true=targets,
                                  y_score=probs[:, 1])
        tp = cm[1][1]
        tn = cm[0][0]
        fp = cm[0][1]
        fn = cm[1][0]

        if mcc_test > best_mcc:
            best_mcc = mcc_test
            best_preds = preds
            best_f1 = f1_test
            best_probs = probs
            best_cm = cm
            with open(f"{path_to_folder}/train_scaler.pkl", "wb") as file:
                pickle.dump(scaler, file)
            os.makedirs(path_to_folder, exist_ok=True)
            joblib.dump(model, f"{path_to_folder}/{kwargs['model']}.pkl")
            _, _, _, _, val_targets, val_preds, val_probs = test_dt(model, x_val_scaled, y_val)
            val_prob_df = pd.DataFrame(val_probs, columns=[f"Prob_{cat}" for cat in range(len(y_test.unique()))])
            val_prob_df["Targets"] = val_targets.values
            val_prob_df["Prediction"] = val_preds

        if logging_active:
            mfl.start_run("model_test")
            log_dict = copy.deepcopy(res)
            del log_dict['model']
            mfl.log_dataset(df_train, "basic", y_train.name, "train")
            mfl.log_dataset(df_test, "basic", y_test.name, "test")
            mfl.log_params(log_dict)
            mfl.log_metrics(
                {"MCC": mcc_test,
                    "F1": f1_test,
                    "Precision": precision,
                    "Recall": recall,
                    "TP": tp,
                    "TN": tn,
                    "FP": fp,
                    "FN": fn
                }
            )
            mfl.log_model(res['model'])
            mfl.end_run()

        mcc_scores.append(mcc_test)
        f1_scores.append(f1_test)
        auc_scores.append(auc_value)

    metric_df = pd.DataFrame({"F1 Test" : f1_scores,
                              "Normed MCC Test" : mcc_scores,
                              "AUC Test" : auc_scores})
    metric_df["Normed MCC Test"] = (metric_df["Normed MCC Test"] + 1) / 2

    metric_df["Model"] = kwargs["model"]
    metric_df["Iteration"] = list(range(1, len(metric_df) + 1))
    cats = len(y_test.unique())
    prob_df = pd.DataFrame(best_probs, columns=[f"Prob_{cat}" for cat in range(cats)])
    prob_df["Targets"] = targets
    prob_df["Prediction"] = best_preds
    prob_df.to_csv(f"{path_to_folder}/{kwargs['model']}_predictions.csv", index=False)
    metric_df.to_csv(f"{path_to_folder}/{kwargs['model']}_metrics.csv", index=False)
    val_prob_df.to_csv(f"{path_to_folder}/{kwargs['model']}_val_predictions.csv", index=False)

    best_x_features = 10

    if kwargs['model'] == "LR":
        coefficients = res['model'].coef_[0]
        feature_names = x_train.columns
        importance = np.abs(coefficients)
        logreg_importance = pd.Series(importance, index=feature_names)
        logreg_importance = logreg_importance.sort_values(ascending=False)
        logreg_importance = logreg_importance[:best_x_features]

        fig, ax = plt.subplots()
        logreg_importance.plot.bar(ax=ax)
        ax.set_title("Feature importances")
        fig.tight_layout()
        fig.savefig(f"{path_to_folder}/{kwargs['model']}_feature_importance.png", dpi=300)

    elif kwargs['model'] == 'RF':
        importances = res['model'].feature_importances_
        std = np.std([tree.feature_importances_ for tree in res['model'].estimators_], axis=0)
        feature_names = x_train.columns

        feature_data = pd.DataFrame({
            'importance': importances,
            'std': std
        }, index=feature_names)

        feature_data = feature_data.sort_values(by='importance', ascending=False)
        feature_data = feature_data.head(best_x_features)
        sorted_importances = feature_data['importance']
        sorted_std = feature_data['std']

        fig, ax = plt.subplots()
        sorted_importances.plot.bar(yerr=sorted_std, ax=ax)
        ax.set_title("Feature importances")
        fig.tight_layout()
        fig.savefig(f"{path_to_folder}/{kwargs['model']}_feature_importance.png", dpi=300)

    elif kwargs['model'] == "DT" or kwargs['model'] == "XGB":
        importance = res['model'].feature_importances_
        print("Feature Importance: ", importance)
        feature_names = x_train.columns
        dt_xgb_importance = pd.Series(importance, index=feature_names)
        dt_xgb_importance = dt_xgb_importance.sort_values(ascending=False)
        dt_xgb_importance = dt_xgb_importance[:best_x_features]

        fig, ax = plt.subplots()
        dt_xgb_importance.plot.bar(ax=ax)
        ax.set_title("Feature importances")
        fig.tight_layout()
        fig.savefig(f"{path_to_folder}/{kwargs['model']}_feature_importance.png", dpi=300)


    return best_f1, best_mcc, precision, recall, best_cm

def fit_and_eval_mlp(res, # pylint: disable=R0913,R0914,R0917, R0912, R0915
                     x_train,
                     y_train,
                     x_test,
                     y_test,
                     path_to_folder,
                     n_predictions=30,
                     logging_active=False,
                     **kwargs):
    """
    Retrain with different Random Seeds to generate Test-Eval Distribution
    """
    device = kwargs["device"] if "device" in kwargs else "cpu"

    if isinstance(x_train, np.ndarray):
        x_train = pd.DataFrame(x_train)

    if isinstance(x_test, np.ndarray):
        x_test = pd.DataFrame(x_test)

    if isinstance(y_train, np.ndarray):
        y_train = pd.Series(y_train)

    if isinstance(y_test, np.ndarray):
        y_test = pd.Series(y_test)

    x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                      y_train,
                                                      stratify=y_train,
                                                      test_size=0.15,
                                                      random_state=42)
    x_train = x_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    x_test= x_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    x_val = x_val.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)
    train_dataset = TorchDataSet(x_train, y_train)
    train_scaler = train_dataset.scaler
    val_dataset = TorchDataSet(x_val, y_val, train_scaler)
    test_dataset = TorchDataSet(x_test, y_test, train_scaler)

    best_preds = None
    best_mcc = -1
    best_f1 = 0
    best_probs = None

    mcc_scores = []
    f1_scores = []
    auc_scores = []

    if logging_active:
        mfl = MlFlowLogger("mlp")

    df_train = x_train.copy()
    df_test = x_test.copy()
    df_train[y_train.name] = y_train
    df_test[y_test.name] = y_test
    best_cm = None
    for seed in range(n_predictions):
        torch.manual_seed(seed)
        model = res["model"]
        model.apply(initialize_weights)
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=res["init_lr"],
                                      weight_decay=res["weight_decay"])
        train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
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
                    #scheduler_pat=res["sched_patience"],
                    #es_pat=res["es_patience"],
                    max_epochs=res["max_epochs"],
                    device=device,
                    #es_delta=0,
                    show_every=None,
                    temp_weight_tag=kwargs["weight_tag"])

        f1_test, mcc_test, precision, recall, targets, preds, probs = test_model(model,
                                                        test_loader,
                                                        device=device,
                                                        num_classes=np.unique(y_train).shape[0],
                                                        return_softmax=True)
        cm = confusion_matrix(targets, preds)

        auc_value = roc_auc_score(y_true=targets,
                                  y_score=probs[:, 1])
        f1_scores.append(f1_test)
        mcc_scores.append(mcc_test)
        auc_scores.append(auc_value)

        if mcc_test > best_mcc:
            best_mcc = mcc_test
            best_preds = preds
            best_f1 = f1_test
            best_probs = probs
            best_cm = cm
            with open(f"{path_to_folder}/train_scaler.pkl", "wb") as file:
                pickle.dump(train_scaler, file)
            os.makedirs(path_to_folder, exist_ok=True)
            model = model.to("cpu")
            torch.save(model.state_dict(), f"{path_to_folder}/MLP.pt")
            _, _, _, _, val_targets, val_preds, val_probs = test_model(model,
                                                           val_loader,
                                                           device=device,
                                                           num_classes=np.unique(y_train).shape[0],
                                                           return_softmax=True)
            val_prob_df = pd.DataFrame(val_probs, columns=[f"Prob_{cat}" for cat in range(len(y_test.unique()))])
            val_prob_df["Targets"] = val_targets
            val_prob_df["Prediction"] = val_preds

        if logging_active:
            mfl.start_run("model_test")
            log_dict = copy.deepcopy(res)
            del log_dict['model']
            mfl.log_dataset(df_train, "basic", y_train.name, "train")
            mfl.log_dataset(df_test, "basic", y_test.name, "test")
            mfl.log_params(log_dict)

            mfl.end_run()


    metric_df = pd.DataFrame({"F1 Test" : f1_scores,
                              "Normed MCC Test" : mcc_scores,
                              "AUC Test" : auc_scores
                              })
    metric_df["Normed MCC Test"] = (metric_df["Normed MCC Test"] + 1) / 2
    metric_df["Model"] = "MLP"
    metric_df["Iteration"] = list(range(1, len(metric_df) + 1))
    cats = len(y_test.unique())
    prob_df = pd.DataFrame(best_probs, columns=[f"Prob_{cat}" for cat in range(cats)])
    prob_df["Targets"] = targets
    prob_df["Prediction"] = best_preds
    prob_df.to_csv(f"{path_to_folder}/MLP_predictions.csv", index=False)
    metric_df.to_csv(f"{path_to_folder}/MLP_metrics.csv", index=False)
    val_prob_df.to_csv(f"{path_to_folder}/MLP_val_predictions.csv", index=False)

    return best_f1, mcc_test, precision, recall, best_cm
