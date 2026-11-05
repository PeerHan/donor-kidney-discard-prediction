import json
from pathlib import Path

import pandas as pd

from bootstrap_eval import load_predictions, predict
from thresholds import select_threshold, threshold_report, thresholds_latex

MODELS = ["DT", "RF", "MLP", "XGB", "LR", "DE"]
RULE = "prevalence"
TARGET = 0.8
POS_LABEL = 1
CLASS_NAMES = ("Discard", "Transplanted")
N_BOOT = 2000
DATA, OUT = Path("predictions"), Path("results") / "thresholds"

OUT.mkdir(parents=True, exist_ok=True)
y_val, p_val = load_predictions({m: DATA / f"{m}_val_predictions.csv" for m in MODELS})
choice = {m: select_threshold(y_val, p_val[m], RULE, POS_LABEL, TARGET) for m in MODELS}
thresholds = {m: selected for m, (selected, _) in choice.items()}
(OUT / "thresholds.json").write_text(json.dumps(thresholds, indent=2))

report = pd.concat([threshold_report(y_val, p_val[m], candidate, POS_LABEL, N_BOOT)
                    .assign(model=m, candidate=candidate, selected=selected, rule=RULE,
                            share_predicted_default=(predict(p_val[m], 0.5) == POS_LABEL).mean(),
                            share_predicted_candidate=(predict(p_val[m], candidate) == POS_LABEL).mean())
                    for m, (selected, candidate) in choice.items()], ignore_index=True)
report.to_csv(OUT / "thresholds.csv", index=False)

pos, cls1 = CLASS_NAMES[POS_LABEL], CLASS_NAMES[1]
rule_text = {"prevalence": f"share of predicted {pos} equal to its prevalence",
             "sensitivity": f"highest specificity with a sensitivity of at least {TARGET}",
             "youden": "maximal Youden index"}[RULE]
caption = (f"Classification thresholds selected on the validation split (n = {len(y_val)}; "
           f"{pos}: {(y_val == POS_LABEL).sum()}; rule: {rule_text}). Thresholds refer to the predicted "
           f"probability of {cls1}; {cls1} is predicted if it exceeds the threshold. Sensitivity and "
           f"specificity refer to {pos} as the positive class. The rule threshold was used only if it "
           f"increased the MCC on the validation data; otherwise the default of 0.5 was kept (Selected). "
           f"Differences are rule threshold minus 0.5 on the validation data, with 95\\% BCa bootstrap "
           f"confidence intervals ({N_BOOT} resamples).")
(OUT / "thresholds.tex").write_text(thresholds_latex(report, caption))

print(json.dumps(thresholds, indent=2))
print(report[["model", "candidate", "selected", "metric", "at_default", "at_threshold", "difference",
              "ci_lower", "ci_upper"]].round(3).to_string(index=False))