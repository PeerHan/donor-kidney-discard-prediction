from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef

from bootstrap_eval import bootstrap_ci, load_predictions, predict
from calibration import scores_latex
from hpo import read_trials
from plots import forest_plots

MODELS = ["LR", "DT", "RF", "XGB", "MLP", "DE"]
FILES = {"fs": "feature_selection_trials/{m}_feature_selection_fs_df.csv",
         "hpo": "trial_data/{m}_trial_data.csv",
         "val": "predictions/{m}_val_predictions.csv",
         "test": "predictions/{m}_predictions.csv"}
FOLDS = {"fs": 3, "hpo": 5}
STAGES = {"fs": f"Feature selection ({FOLDS['fs']}-fold CV)", "hpo": f"Tuning ({FOLDS['hpo']}-fold CV)",
          "val": "Validation", "test": "Test"}
N_BOOT = 2000
NCOLS = 2
OUT = Path("results") / "dev_vs_test"


def mcc_with_ci(file):
    y, p = load_predictions({"model": file})
    pred = predict(p["model"], 0.5)
    lo, hi = bootstrap_ci(lambda r: np.array([matthews_corrcoef(y[r], pred[r])]), len(y), n_boot=N_BOOT)
    return matthews_corrcoef(y, pred), lo[0], hi[0], len(y)


rows, n, n_trials = [], {}, {"fs": set(), "hpo": set()}
for m in MODELS:
    path = {key: Path(pattern.format(m=m)) for key, pattern in FILES.items()}
    for key, best in (("fs", lambda t: t.loc[t["MCC Loss"].idxmin(), "MCC"]), ("hpo", lambda t: t["MCC"].max())):
        if path[key].exists():
            trials = read_trials(path[key])
            n_trials[key].add(len(trials))
            rows.append((m, STAGES[key], best(trials), np.nan, np.nan))
    for key in ("val", "test"):
        est, lo, hi, n[key] = mcc_with_ci(path[key])
        rows.append((m, STAGES[key], est, lo, hi))
table = pd.DataFrame(rows, columns=["model", "stage", "estimate", "ci_lower", "ci_upper"])
stages = [s for s in STAGES.values() if s in set(table.stage)]

best_of = " and ".join(f"{min(c)}" + (f"–{max(c)}" if len(c) > 1 else "") + f" ({name})"
                       for c, name in ((n_trials["fs"], "feature selection"), (n_trials["hpo"], "tuning")) if c)
caption = ("MCC during model development and on held-out data. "
           + (f"Cross-validated values (training set) belong to the selected candidate, i.e., the best of "
              f"{best_of} trials, and are therefore optimistic. " if best_of else "")
           + f"Validation (n = {n['val']}) and test (n = {n['test']}) MCC at a threshold of 0.5 with 95\\% BCa "
             f"bootstrap confidence intervals ({N_BOOT} resamples).")

OUT.mkdir(parents=True, exist_ok=True)
table.to_csv(OUT / "dev_vs_test.csv", index=False)
(OUT / "dev_vs_test.tex").write_text(scores_latex(
    table.rename(columns={"stage": "method"}), caption, label="tab:dev-vs-test", lower_is_better=None,
    methods=stages))
grid = pd.MultiIndex.from_product([MODELS, stages], names=["model", "stage"])
forest_plots(table.set_index(["model", "stage"]).reindex(grid).reset_index(), by="model", label="stage",
             ncols=NCOLS, sharey=True, header="MCC (95% CI)", xlabel="MCC", out_file=OUT / "dev_vs_test.png")
print(caption)
print(table.round(3).to_string(index=False))