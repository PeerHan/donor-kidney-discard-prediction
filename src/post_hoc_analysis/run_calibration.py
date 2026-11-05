import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from bootstrap_eval import load_predictions
from calibration import (balanced_brier, calibrate_all, calibration_curves, calibration_intercept,
                         calibration_overlay, calibration_slope, score_with_ci, scores_latex)
from plots import forest_plots

MODELS = ["LR", "DT", "RF", "XGB", "MLP", "DE"]
METHODS = ("sigmoid", "isotonic", "temperature")
MAIN_METHOD = "Sigmoid"
CLASS_NAMES = ("Discard", "Transplanted")
POS_LABEL = 1
N_BOOT = 2000
NCOLS = 2
SCORES = {"brier": ("Brier score", brier_score_loss, None, True, 3),
          "balanced_brier": ("Balanced Brier score", balanced_brier, None, True, 3),
          "intercept": ("Calibration intercept", calibration_intercept, 0, None, 2),
          "slope": ("Calibration slope", calibration_slope, 1, None, 2)}
DATA, OUT = Path("predictions"), Path("results")

y_val, p_val = load_predictions({m: DATA / f"{m}_val_predictions.csv" for m in MODELS})
y_test, p_test = load_predictions({m: DATA / f"{m}_predictions.csv" for m in MODELS})

for m in MODELS:
    val_values, test_values = np.unique(p_val[m].round(12)), np.unique(p_test[m].round(12))
    shared = np.isin(test_values, val_values).mean()
    if len(test_values) < 100 and shared < 0.5:
        warnings.warn(f"{m}: only {shared:.0%} of the test prediction values occur in the validation "
                      "predictions - the files probably come from different model fits")

for sub in ("figures", "tables", "calibrated"):
    (OUT / sub).mkdir(parents=True, exist_ok=True)

calibrated = calibrate_all(y_val, p_val, p_test, METHODS)
_, metrics = calibration_curves(y_test, calibrated, pos_label=POS_LABEL, class_names=CLASS_NAMES,
                                out_file=OUT / "figures" / "calibration.png")
calibration_overlay(y_test, calibrated, methods=("Uncalibrated", MAIN_METHOD), pos_label=POS_LABEL,
                    class_names=CLASS_NAMES, out_file=OUT / "figures" / "calibration_main.png")
metrics.to_csv(OUT / "tables" / "calibration_metrics.csv", index=False)

for key, (name, score, ideal, lower_is_better, decimals) in SCORES.items():
    scores = score_with_ci(y_test, calibrated, score=score, n_boot=N_BOOT)
    scores.to_csv(OUT / "tables" / f"{key}.csv", index=False)
    rule = ("Lower is better, and the lowest value per model is in bold." if lower_is_better else
            f"The ideal value is {ideal}. The {name.lower()} refers to P({CLASS_NAMES[1]}).")
    (OUT / "tables" / f"{key}.tex").write_text(scores_latex(
        scores, caption=f"{name} on the test set (n = {len(y_test)}) with 95\\% BCa bootstrap confidence "
                        f"intervals ({N_BOOT} resamples). Calibrators were fitted on the validation split "
                        f"(n = {len(y_val)}). {rule}",
        label=f"tab:{key.replace('_', '-')}", decimals=decimals, lower_is_better=lower_is_better))
    forest_plots(scores, by="model", label="method", ncols=NCOLS, null=ideal, decimals=decimals,
                 out_file=OUT / "figures" / f"forest_{key}.png",
                 xlabel=f"{name} (lower is better)" if lower_is_better else f"{name} (ideal {ideal})")
    print(f"\n{name}:\n{scores.round(3).to_string(index=False)}")

for m, preds in calibrated.items():
    for method, p in preds.items():
        if method != "Uncalibrated":
            pd.DataFrame({"Prob_0": 1 - p, "Prob_1": p, "Targets": y_test,
                          "Prediction": (p > 0.5).astype(int)}
                         ).to_csv(OUT / "calibrated" / f"{m}_{method.lower()}_predictions.csv", index=False)

print(f"\nCalibration: n_val={len(y_val)} (class 0: {(y_val == 0).sum()}), "
      f"n_test={len(y_test)} (class 0: {(y_test == 0).sum()})")
print(metrics.round(3).to_string(index=False))