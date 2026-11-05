# Necessary Folders + Data
- feature_selection_trials: Optuna trials of the feature selection per model (one row per candidate subset:
  MCC, MCC loss incl. feature penalty, number of features, selected features as True/False, NSGA-II generation)
- predictions: predicted probabilities per model, `<MODEL>_predictions.csv` (test set) and
  `<MODEL>_val_predictions.csv` (validation split), columns Prob_0, Prob_1, Targets, Prediction
- shap_data: SHAP values per model on the test set (rows = test cases, columns = features)
- trial_data: Optuna trials of the hyperparameter tuning per model (hyperparameters and 5-fold CV MCC per trial)
- results:
    - calibrated: calibrated test predictions per model and calibration method (same format as predictions)
    - dev_vs_test: MCC during development (feature selection CV, tuning CV) vs validation and test (CSV, LaTeX, figure)
    - figures: all figures (calibration curves, forest plots, SHAP, confusion matrices, HPO and feature selection)
    - tables: CSV and LaTeX tables (test metrics, calibration metrics, Brier scores, HPO search spaces)
    - thresholds: classification threshold per model chosen on the validation split (JSON) with report (CSV, LaTeX)

# Script Overview
- bootstrap_eval.py: test metrics (sensitivity, specificity, PPV, NPV, F1 and PR-AUC per class, MCC, AUC,
  Brier) with 95% BCa bootstrap CIs, optionally as paired differences to a reference model
- calibration.py: post-hoc calibration from predicted probabilities (sigmoid, isotonic, temperature),
  calibration curves, Brier score, calibration intercept and slope
- hp.py: reads Optuna trials and builds the search-space and best-trial tables (LaTeX)
- plots.py: figure functions (forest plots, SHAP bars and beeswarms, confusion matrices, HPO history,
  feature selection study, feature overlap and Jaccard heatmap)
- thresholds.py: threshold selection on validation predictions (0.5 kept unless the rule improves the MCC)
- run_calibration.py: fits the calibrators on the validation split, evaluates them on the test set
  -> results/calibrated, figures, tables
- run_optimism_comp.py: compares cross-validated MCC from feature selection and tuning with validation
  and test MCC -> results/dev_vs_test
- run_thresholds.py: selects one threshold per model on the validation split -> results/thresholds
- run_visualization.py: creates the remaining figures and tables from predictions, shap_data and trial data
- run_all_analysis.sh: runs the complete analysis in order (thresholds before the test evaluation that uses them)