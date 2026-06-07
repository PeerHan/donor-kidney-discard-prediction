#!/usr/bin/env python3
"""
Module with Plotting Functions after the training/testing is finished
"""

import os

import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.stats.anova import AnovaRM
from scipy.stats import tukey_hsd
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import f_oneway
from matplotlib.colors import ListedColormap

def mcc_performance_with_seeds(parent_folder, metric, feature_sets, models, kind="strip"):
    """
    Plotting a Distribution of MCC values based on
    a model on a specific feature set.

    The plot is a swarm plot and based on the best
    hp set on a 5 fold cv.

    Parameters:
    - parent_folder e.g. /shared/model_results/
    - feature-sets : List of feature-sets, Those feature-sets
    must be available as subdir in parent_folder. 
    Additionally there should be a folder for each model
    with a x_metrics.csv where x is the name of the model.

    The img will be saved to parent_dir/Imgs
    """
    if not os.path.exists(f"{parent_folder}/Imgs"):
        os.makedirs(f"{parent_folder}/Imgs")
    if not os.path.exists(f"{parent_folder}/Imgs/Summary_Plots"):
        os.makedirs(f"{parent_folder}/Imgs/Summary_Plots")
    dfs = []

    for parent_dir in feature_sets:
        label = parent_dir.title().replace("_", " ")
        for model in models:
            df = pd.read_csv(f"{parent_folder}/{parent_dir}/{model}/{model}_metrics.csv")
            df["Feature Set"] = label
            dfs.append(df)

    eval_df = pd.concat(dfs, ignore_index=True)
    sns.catplot(eval_df,
                x="Model",
                y=metric,
                kind=kind,
                palette="tab10",
                hue="Feature Set",
                alpha=1,
                height=6,
                aspect=1.5)
    plt.title(f"Distribution of Test {metric} based on Random Seeds of\n" +
              "different Model-Feature set Combinations")    
    plt.ylabel(metric, fontsize=18)
    plt.xlabel("Model Typ", fontsize=18)
    plt.xticks(fontsize=15)
    plt.savefig(f"{parent_folder}/Imgs/Summary_Plots/{kind}_{metric}_plot.png",
                dpi=300, bbox_inches="tight")



def roc_auc_plot_modelwise(parent_folder, models, save_folder): # pylint: disable=R0914
    """
    Function to create a Subplot:
    - ax1 a ROC Plot with all 5 Models
    - ax2 a AUC Barplot for all 5 Models

    Input:
    - parent_folder : Folder to results/feature_set/
    e.g. 'mnt/shared/results/benchmark'

    The Plot will be saved to 
    results/feature_set/Imgs/ROC_AUC_Testplot.png
    """
    if not os.path.exists(f"{save_folder}/Imgs/ROC_AUC"):
        os.makedirs(f"{save_folder}/Imgs/ROC_AUC")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), dpi=150)
    #plt.figure(figsize=(8, 5), dpi=150)
    ax1.plot([0, 1], [0, 1], linestyle="--", color="red", label="Random")
    auc_values = {}
    feature_set = parent_folder.split("/")[-1]
    for algo in models:
        df = pd.read_csv(f"{parent_folder}/{algo}/{algo}_predictions.csv")
        probs = df.loc[:, "Prob_1"]
        y_true = df.loc[:, "Targets"]
        fpr, tpr, _ = roc_curve(y_true=y_true,
                            y_score=probs)
        auc = roc_auc_score(y_true=y_true,
                            y_score=probs)
        auc_values[algo] = auc
        ax1.plot(fpr, tpr, marker=".", label=f"{algo}")


    ax1.legend()
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title(f"ROC Curve of the {feature_set.title()} Feature Set")

    auc_values =  dict(sorted(auc_values.items(), key=lambda item: item[1], reverse=True))
    sns.barplot(auc_values, ax=ax2)
    ax2.set(xlabel="Model",
            ylabel="AUC Value",
            title="AUC Values")
    ax2.axhline(0.5, color="red", linestyle="--")
    fig.suptitle(f"ROC Curves and AUC Values of the {feature_set.title()} Feature Set")
    fig.tight_layout()

    plt.savefig(f"{save_folder}/Imgs/ROC_AUC/{feature_set}_ROC_AUC_Testplot.png",
                bbox_inches="tight",
                dpi=250)
    plt.close()

def roc_auc_plot_setwise(algo, parent_folder, feature_sets): # pylint: disable=R0914
    """
    Function to create a Subplot:
    - ax1 a ROC Plot for one Model for all Feature Sets
    - ax2 a AUC Barplot for one Model for all Feature Sets

    Input:
    - parent_folder : Folder to results/
    e.g. 'mnt/shared/results/'

    The Plot will be saved to 
    results/Imgs/ROC_AUC_Testplot.png
    """
    if not os.path.exists(f"{parent_folder}/Imgs"):
        os.makedirs(f"{parent_folder}/Imgs")
    if not os.path.exists(f"{parent_folder}/Imgs/ROC_AUC"):
        os.makedirs(f"{parent_folder}/Imgs/ROC_AUC")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), dpi=150)
    ax1.plot([0, 1], [0, 1], linestyle="--", color="red", label="Random")
    auc_values = {}
    for feature_set in feature_sets:
        df = pd.read_csv(f"{parent_folder}/{feature_set}/{algo}/{algo}_predictions.csv")
        probs = df.loc[:, "Prob_1"]
        y_true = df.loc[:, "Targets"]
        fpr, tpr, _ = roc_curve(y_true=y_true,
                            y_score=probs)
        auc = roc_auc_score(y_true=y_true,
                            y_score=probs)
        auc_values[feature_set] = auc
        ax1.plot(fpr, tpr, marker=".", label=f"{feature_set}")
    ax1.legend()
    ax1.set(xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            title="ROC Kurve")
    auc_values =  dict(sorted(auc_values.items(), key=lambda item: item[1], reverse=True))
    auc_values = {key.replace("_", "\n") : value\
                  for key, value in auc_values.items()}
    sns.barplot(auc_values, ax=ax2)
    ax2.set(xlabel="Feature Sets",
            ylabel="AUC Value",
            title="AUC Values")
    ax2.axhline(0.5, color="red", linestyle="--")
    ax2.tick_params(axis='x', which='minor', labelsize=5)
    fig.suptitle(f"ROC Curve and AUC Values of {algo.title()} dependent on the Feature Set")
    fig.tight_layout()

    plt.savefig(f"{parent_folder}/Imgs/ROC_AUC/{algo}_ROC_AUC_Testplot.png",
                bbox_inches="tight",
                dpi=250)
    plt.close()

def cohens_d(vector1, vector2):
    """
    Calculate Cohens d for measureing the effect size
    based on:
    Practically significant method comparison protocols
    for machine learning in small molecule drug discovery.

    Parameters:
     - vector1 Metric vector of model1
     - vector2 Metric vector of model2
    """
    mu1 = vector1.mean()
    mu2 = vector2.mean()
    std1 = vector1.std()
    std2 = vector2.std()
    d = (mu1 - mu2) / (0.5 * (std1 + std2))
    return d

def cohen_matrix(label_vector_dict):
    """
    Calculate a n x n matrix for multiple models
    Parameters:
     - label_vector_dict: Dict with {Modelname : Vector}
    """
    n = len(label_vector_dict)
    d_matrix = np.zeros((n, n))
    for num1, (_, vector1) in enumerate(label_vector_dict.items()):
        for num2, (_, vector2) in enumerate(label_vector_dict.items()):
            d_matrix[num1, num2] = cohens_d(vector1, vector2)
    return d_matrix

def cohen_d_plot(title, vec_dict, parent_folder, save_file):
    """
    1 x 2 Subplot Showing:
     - KDE Distribution of Metric vectors per model
     - Heatmap of the cohen_d matrix
    Function can be used modelwise or featurewise

    Parameters:
     - title: Title of the plot
     - vec_dict: Dict with {x : metric_vector}
     - parent_folder: Folder to 
     /parent_folder/feature_set/model/model_metrics.csv
     - save_file: Name of the file
    """
    if not os.path.exists(f"{parent_folder}/Imgs/Cohen"):
        os.makedirs(f"{parent_folder}/Imgs/Cohen")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), dpi=150)
    d_matrix = cohen_matrix(vec_dict)
    for label, vec in vec_dict.items():
        sns.kdeplot(vec, bw_adjust=2, ax=ax1, label=label)

    ax1.legend(fontsize=15, loc="upper left")
    sns.move_legend(
        ax1, "lower center",
        bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False,
    )
    sns.heatmap(d_matrix,
                cmap="flare",
                ax=ax2,
                xticklabels=[col.replace("_", "\n") for col in list(vec_dict.keys())],
                yticklabels=vec_dict.keys(),
                annot=True,
                annot_kws={"fontsize":15})
    plt.suptitle(title, size=20)
    fig.tight_layout()
    fig.savefig(f"{parent_folder}/Imgs/Cohen/{save_file}", bbox_inches="tight", dpi=150)
    plt.close()

def cohen_d_featurewise(features, models, parent_folder):
    """
    Function to use cohen_d_plot to generate Featurewise plots
    Parameters:
     - features: All Featuresets as list like
     ['benchmark', ..., 'all']
     - models: All models as list like
     ['DT', ..., 'MLP']
    - parent_folder: Parent folder to access
    the metric files. Must have structure
    /parent_folder/featureset/model/model_metrics.csv
    """
    for feature in features:
        vec_dict = {}
        for model in models:
            metric_vector = pd.read_csv(f"{parent_folder}/{feature}/{model}/{model}_metrics.csv")
            vec_dict[model] = metric_vector.iloc[:, :]["Normed MCC Test"]
        cohen_d_plot(
            title=f"Normed MCC Distribution and Cohen D Matrix\nFor the {feature} Feature set",
            vec_dict=vec_dict,
            parent_folder=parent_folder,
            save_file=f"cohen_d_featurewise_{feature}.png"
        )


def cohen_d_modelwise(features, models, parent_folder):
    """
    Function to use cohen_d_plot to generate Featurewise plots
    Parameters:
     - features: All Featuresets as list like
     ['benchmark', ..., 'all']
     - models: All models as list like
     ['DT', ..., 'MLP']
    - parent_folder: Parent folder to access
    the metric files. Must have structure
    /parent_folder/featureset/model/model_metrics.csv
    """
    for model in models:
        vec_dict = {}
        for feature in features:
            metric_vector = pd.read_csv(f"{parent_folder}/{feature}/{model}/{model}_metrics.csv")
            vec_dict[feature] = metric_vector.iloc[:, :]["Normed MCC Test"]
        cohen_d_plot(
            title=f"Normed MCC Distribution and Cohen D Matrix\nFor {model}",
            vec_dict=vec_dict,
            parent_folder=parent_folder,
            save_file=f"cohen_d_modelwise_{model}.png"
        )

def tukey_plots_featurewise(models, feature_sets, title, save_path, parent_folder): # pylint: disable=R0914
    """
    2 x 3 Subplot with each subplot containing a Heatmap
    where the p-values of a tukey hsd posthoc test are contained.
    The ticks of the heatmaps corresponds to the feature_sets to
    capture significant differences between feature_sets for certain
    models.

    Parameters:
     - models: list of models used like
     ['LR', 'DT', 'RF', 'XGB', 'MLP', 'Ensemble']
     - feature_sets: list of feature_sets used
     ['benchmark', 'domain', ..., 'all']
     - title: Title of the Plot
     - save_path: Path with name to save
     e.g. /dir/cohen_d_plot.png
     - parent_folder: Folder where the feature_sets and model_metrics
     are saved. Must have structure: parent_folder/feature_set/model/model_metrics.csv
    """
    fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(15, 5), dpi=150)
    i = j = 0
    for model in models:
        tukey_dict = {}
        for feature_set in feature_sets:
            # Adjust when everything is collected
            domain_scores = pd.read_csv(f"{parent_folder}/{feature_set}/{model}/{model}_metrics.csv")["Normed MCC Test"] # pylint: disable=C0301
            tukey_dict[feature_set] = domain_scores
        df = pd.DataFrame(tukey_dict)
        df = pd.DataFrame(tukey_dict)
        df = df.melt(var_name="Model", value_name="Normed MCC Test")
        unique_models = df.Model.unique().shape[0]
        amount_models = df.Model.value_counts().mean().astype(int)
        df["Model_idx"] = list(range(1, amount_models + 1)) * unique_models
        anova = AnovaRM(data=df, depvar='Normed MCC Test',
                        subject='Model_idx', within=["Model"]).fit()
        pval = anova.anova_table["Pr > F"].values[0]
        test = tukey_hsd(*tukey_dict.values())
        hm = test.pvalue
        sns.heatmap(hm,
                    cmap="flare",
                    annot=True,
                    vmax=0.05,
                    vmin=0,
                    cbar=False,
                    ax=axs[i, j],
                    xticklabels=tukey_dict.keys(),
                    yticklabels=tukey_dict.keys())
        axs[i, j].title.set_text(f"{model}\nAnovaRM P: {pval:.2}")
        j += 1
        if j > 2:
            j = 0
            i += 1

    fig.tight_layout()
    fig.suptitle(title, y=1.05)

    cbar_ax = fig.add_axes([1, 0.3, 0.02, 0.4])
    cbar = fig.colorbar(axs[0, 0].collections[0], cax=cbar_ax)
    cbar.set_label('P-Wert', fontsize=12)
    fig.suptitle(title, y=1.05)
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()

def tukey_plots(models, feature_sets, save_path, parent_folder, tukey_plot=False):

    for metric in ["F1 Test", "Normed MCC Test", "AUC Test"]:
        fig, axs = plt.subplots(1, 2, figsize=(15, 5), dpi=250)
        i = j = 0
        for dataset in feature_sets:
            if j > 1:
                j = 0
                i += 1
            dfs = []
            anova_input = []
            for model in models:
                domain_scores = f"{parent_folder}/{dataset}/{model}/{model}_metrics.csv"
                df = pd.read_csv(domain_scores)
                df["Model"] = model
                df["Dataset"] = dataset
                dfs.append(df)
                metric_val = df[metric]
                anova_input.append(metric_val)
            pval = f_oneway(*anova_input)[1]
            ds_df = pd.concat(dfs)
            tukey = pairwise_tukeyhsd(endog=ds_df[metric], groups=ds_df["Model"], alpha=0.05)
            if tukey_plot:
                best_group = ds_df.groupby("Model")[metric].mean().sort_values(ascending=False).index[0]
                tukey.plot_simultaneous(comparison_name=best_group, figsize=(15, 5), ax=axs[0])
            else:
                sns.boxplot(
                    ds_df,
                    y=metric,
                    x="Model",
                    notch=0,
                    showcaps=False,
                    flierprops={"marker": "x"},
                    boxprops={"facecolor": (.3, .5, .7, .5)},
                    medianprops={"color": "r", "linewidth": 2},
                    ax=axs[0]
                )
            fig.suptitle(f"Comparison of Discriminative Test Metric Distributions ({metric} Metric | ANOVA p-val={pval:.3})")
            levels = tukey.groupsunique
            n = len(levels)
            order = ["DT", "LR", "RF", "XGB", "MLP", "DE"]
            pmat = pd.DataFrame(np.nan, index=levels, columns=levels, dtype=float)
            pmat = pmat.reindex(index=order, columns=order)

            for row in tukey._results_table.data[1:]:  # skip header
                g1, g2, meandiff, p_adj, lower, upper, reject = row[:7]
                p = float(p_adj)
                # We’ll store in both [g2, g1] and [g1, g2], but later mask the upper triangle
                pmat.loc[g2, g1] = p
                pmat.loc[g1, g2] = p
            
            mask = np.triu(np.ones(pmat.shape, dtype=bool))
            pmat_masked = pmat.mask(mask)
            
            annot_vals = pmat_masked.copy()
            annot_vals = annot_vals.applymap(lambda x: "<0.001" if pd.notnull(x) and x < 0.001 else (f"{x:.3f}" if pd.notnull(x) else ""))
            
            sig_mask = pmat_masked.copy()
            sig_mask = sig_mask.applymap(lambda x: 1 if pd.notnull(x) and x < 0.05 else 0)
            
            cmap = ListedColormap(["white", "#ffcccc"])
            sns.heatmap(
                sig_mask,
                ax=axs[1],
                linecolor="black",
                annot=annot_vals,
                fmt="",
                cmap=cmap,
                mask=mask,
                cbar=False  # no colorbar needed for binary significance
            )
            plt.title(f"P-values heatmap for {metric}")
            j += 1
        fig.tight_layout()
        fig.savefig(f"{save_path}/{metric}_tukey_{int(tukey_plot)}.png", bbox_inches="tight", dpi=250)
        fig.show()

def global_metric_barplot(path, metric, feature_sets, hue=None):
    """
    Global Alternative to Stripplot
    """
    all_fs_df = []
    for fs in feature_sets:
        fs_folder = f"{path}/{fs}"
        fs_dfs = []
        for model in ["DT", "LR", "RF", "XGB", "MLP", "DE"]: #, "DE"
            model_csv = f"{fs_folder}/{model}/{model}_metrics.csv"
            model_df = pd.read_csv(model_csv)
            model_df["Modell"] = model
            fs_dfs.append(model_df)
        fs_df = pd.concat(fs_dfs, ignore_index=True)
        fs_df["Feature Set"] = fs.replace("_", "\n")
        all_fs_df.append(fs_df)
    all_fs_df = pd.concat(all_fs_df, ignore_index=True)
    plt.figure(figsize=(15, 5))
    ax = sns.barplot(all_fs_df, x="Feature Set", y=metric, hue=hue, errorbar="sd", #, "DE"
                     hue_order=["LR", "DT", "RF", "XGB", "MLP", "DE"]
                     if hue else None)
    if hue:
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    tag = hue if hue else "no_hue"
    metric_tag = metric.replace(' ', '_')
    plt.title(f"Mean {metric} values with Standard Deviation for every Model-Feature set Combinations") # pylint: disable=C0301
    plt.savefig(f"{path}/Imgs/Summary_Plots/grouped_{metric_tag}_{tag}.png",
                    bbox_inches="tight", dpi=150)
    plt.close()
