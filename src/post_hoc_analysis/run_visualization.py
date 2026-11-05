from bootstrap_eval import bootstrap_eval, load_predictions
from plots import shap_top_features, forest_plots, confusion_matrices, hpo_history, jaccard_heatmap, feature_selection_study, pr_curves
from hpo import search_space_latex, best_trials_latex
from marsilea.upset import Upset, UpsetData
import pandas as pd 
import json
from pathlib import Path


thresholds = json.loads(Path("results/thresholds/thresholds.json").read_text())

table_path = "results/tables"
fig_path = "results/figures"

from pathlib import Path

from bootstrap_eval import load_predictions
from plots import pr_curves

CLASS_NAMES = ("Discard", "Transplanted")  # class 0, class 1

y_test, p_test = load_predictions({m: f"predictions/{m}_predictions.csv" for m in ["DT", "MLP", "RF", "XGB", "LR", "DE"]})
pr_curves(y_test, p_test, class_names=CLASS_NAMES, out_file="results/figures/pr_curves.png")

# 0 Feature Selection
fig, summary = feature_selection_study(
    {m: f"feature_selection_trials/{m}_feature_selection_fs_df.csv" for m in ["DT", "RF", "MLP", "LR", "XGB"]},
    out_file=f"{fig_path}/feature_selection.png"
)

y, probs = load_predictions(
    {m: f"predictions/{m}_predictions.csv" for m in ["DT", "MLP", "RF", "XGB", "LR", "DE"]}
)

kw = dict(pos_label=1, class_names=("Discard", "Transplanted"))
ci_res  = bootstrap_eval(y, probs, thresholds=thresholds, **kw)
ci_res_lr  = bootstrap_eval(y, probs, reference="LR", thresholds=thresholds, **kw)

ci_res.to_csv(f"{table_path}/performance_cis.csv", index=False)
ci_res_lr.to_csv(f"{table_path}/performance_cis_lr_baseline.csv", index=False)

# 1 Performance Forest Subplots

## Discard vs Transplant
metrics = ["F1 (Transplanted)", "F1 (Discard)", "PR-AUC (Transplanted)", "PR-AUC (Discard)"]
forest_plots(ci_res[ci_res.metric.isin(metrics)], ncols=2, out_file=f"{fig_path}/forest_f1_prauc.png")

## CM Metrics 
metrics = ["Sensitivity", "PPV", "Specificity", "NPV"]
forest_plots(ci_res[ci_res.metric.isin(metrics)], ncols=2, out_file=f"{fig_path}/forest_cm.png")

## MCC, AUC, Balanced Brier, PRAUC
metrics = ["MCC", "AUC", "PR-AUC (Discard)", "PR-AUC (Transplanted)", "Balanced Brier"]
forest_plots(ci_res[ci_res.metric.isin(metrics)], ncols=3, out_file=f"{fig_path}/forest_main.png")


# 2 Performance vs LR Subplots

## Discard vs Transplant
metrics = ["F1 (Transplanted)", "F1 (Discard)", "PR-AUC (Transplanted)", "PR-AUC (Discard)"]
forest_plots(ci_res_lr[ci_res_lr.metric.isin(metrics)], ncols=2, out_file=f"{fig_path}/forest_f1_prauc_lr.png")

## CM Metrics 
metrics = ["Sensitivity", "PPV", "Specificity", "NPV"]
forest_plots(ci_res_lr[ci_res_lr.metric.isin(metrics)], ncols=2, out_file=f"{fig_path}/forest_cm_lr.png")

## MCC, AUC, Balanced Brier, PRAUC
metrics = ["MCC", "AUC", "PR-AUC (Discard)", "PR-AUC (Transplanted)", "Balanced Brier"]
forest_plots(ci_res_lr[ci_res_lr.metric.isin(metrics)], ncols=3, out_file=f"{fig_path}/forest_main_lr.png")


# 3 CMs
_ = confusion_matrices(
    y, probs, class_names=("Transplanted", "Discarded"), thresholds=thresholds, out_file=f"{fig_path}/cm_matrices.png"
)

# 4 HPO

trials = {m: f"trial_data/{m}_trial_data.csv" for m in ["LR", "DT", "RF", "XGB", "MLP"]}
Path(f"{table_path}/hpo_best_trials.tex").write_text(best_trials_latex(trials, n=5))

trials = {m: f"trial_data/{m}_trial_data.csv" for m in ["DT", "MLP", "RF", "XGB", "LR"]}
hpo_history(trials, window=10, out_file=f"{fig_path}/hpo_history.png")
open(f"{table_path}/hpo_search_space.tex", "w").write(search_space_latex(trials))

# 5 SHAP
# Post hoc mapping from german to english
feature_name_map = {
    "Alter": "Age",
    "Anti_Toxoplasmose": "Toxoplasma antibodies",
    "Kreatinin_first_value": "Creatinine first value",
    "Kreatinin_last_value": "Creatinine last value",
    "Kreatinin_min_value": "Creatinine min value",
    "Kreatinin_max_value": "Creatinine max value",
    "Kreatinin_intercept": "Creatinine intercept",
    "Kreatinin_std_deviation": "Creatinine std deviation",
    "Harnstoff_min_value": "Urea min value",
    "Harnstoff_max_value": "Urea max value",
    "Harnstoff_intercept": "Urea intercept",
    "Kalium_intercept": "Potassium intercept",
    "Chlorid_max_value": "Chloride max value",
    "Gesamtprotein_first_value": "Total protein first value",
    "Haemoglobin_max_value": "Hemoglobin max value",
    "Blutgase_FIO2_std_deviation": "Blood gas FiO2 std deviation",
    "Blutgase100.Ph" : "Blood gas pH",
    "Urin_HB_first_value_Negative": "Urine hemoglobin first value negative",
    "Urin_Glukose_first_value_Negative": "Urine glucose first value negative",
    "Urin_Sediment_Leukozyten_first_value Negative": "Urine sediment leukocytes first value negative",
    "Urin_Protein_qualitativ_first_value": "Urine protein (qualitative) first value",
    "Urin_Protein_qualitativ_last_value": "Urine protein (qualitative) last value",
    "Diurese_24h_kgKG": "Urine output 24h per kg body weight",
    "Diurese_letzte_Stunde_stand": "Urine output last hour (reading)",
    "Echokardio.Aorten_Annulus_Wurzel": "Echo: aortic annulus/root",
    "Echokardio.Stenose_Aortenklappe_keine": "Echo: no aortic valve stenosis",
    "Echokardio.Morphologie_Tricuspidalklappe_unauffällig": "Echo: tricuspid valve morphology unremarkable",
    "Abdomen.PankreasParenchymFk_unauffällig": "Abdomen: pancreatic parenchyma unremarkable",
    "Abdomen.ReNiereMorphologieFk_unauffällig": "Abdomen: right kidney morphology unremarkable",
    "Abdomen.LiNiereMorphologieFk_unauffällig": "Abdomen: left kidney morphology unremarkable",
}


members = ["RF", "LR", "XGB", "MLP"]
selected = {m: pd.read_csv(f"shap_data/{m}_grouped_shap_values.csv").rename(feature_name_map, axis=1)  for m in ["DT", "MLP", "RF", "XGB", "LR"]}
selected["DE"] = pd.concat([selected[m] for m in members]).groupby(level=0).sum() / len(members)
 
fig = shap_top_features(
    selected,
    out_file=f"{fig_path}/shap_plot.png"
)

selected = {m: pd.read_csv(f"shap_data/{m}_grouped_shap_values.csv") for m in ["DT", "MLP", "RF", "XGB", "LR"]}
ACCENT="#2a78d6"
data = UpsetData.from_sets([set(f) for f in selected.values()], sets_names=list(selected))
upset = Upset(data, sets_order=list(selected), sort_subsets="-cardinality", color=ACCENT)
upset.render()
upset.save(f"{fig_path}/upset_plot.png", dpi=300)

fig = jaccard_heatmap(selected, out_file=f"{fig_path}/jaccard.png")

