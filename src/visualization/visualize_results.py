#!/usr/bin/env python3

"""
Executive Script to visualize all results
"""

import os
from evaluation_plots import tukey_plots, roc_auc_plot_setwise, \
    mcc_performance_with_seeds, cohen_d_featurewise, global_metric_barplot 

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

PARENT_FOLDER = os.getenv("MODEL_RESULTS")

# Adjust when Feature Sets are set
feature_sets = ["paper-free"]

models =["DT", "RF", "XGB", "MLP", "DE", "LR"] #, "DE"

for metric in ("F1 Test", "Normed MCC Test", "AUC Test"):
    mcc_performance_with_seeds(parent_folder=PARENT_FOLDER,
                               feature_sets=feature_sets,
                               models=models,
                               metric=metric)
    mcc_performance_with_seeds(parent_folder=PARENT_FOLDER,
                               feature_sets=feature_sets,
                               kind="violin",
                               models=models,
                               metric=metric)

for algo in models:
    roc_auc_plot_setwise(algo, parent_folder=PARENT_FOLDER, feature_sets=feature_sets)

cohen_d_featurewise(
    feature_sets,
    models,
    PARENT_FOLDER
)


if not os.path.exists(f"{PARENT_FOLDER}/Imgs/Tukey"):
    os.makedirs(f"{PARENT_FOLDER}/Imgs/Tukey")

tukey_plots(
    models=models,
    feature_sets=feature_sets,
    save_path=f"{PARENT_FOLDER}/Imgs/Tukey/",
    parent_folder=PARENT_FOLDER
)

tukey_plots(
    models=models,
    feature_sets=feature_sets,
    save_path=f"{PARENT_FOLDER}/Imgs/Tukey/",
    parent_folder=PARENT_FOLDER,
    tukey_plot=True
)

global_metric_barplot(path=PARENT_FOLDER, metric="F1 Test", feature_sets=feature_sets)
global_metric_barplot(path=PARENT_FOLDER, metric="F1 Test", hue="Modell", feature_sets=feature_sets)
global_metric_barplot(path=PARENT_FOLDER, metric="Normed MCC Test", feature_sets=feature_sets)
global_metric_barplot(path=PARENT_FOLDER, metric="Normed MCC Test",
                      hue="Modell", feature_sets=feature_sets)
global_metric_barplot(path=PARENT_FOLDER, metric="AUC Test", feature_sets=feature_sets)
global_metric_barplot(path=PARENT_FOLDER, metric="AUC Test",
                      hue="Modell", feature_sets=feature_sets)
