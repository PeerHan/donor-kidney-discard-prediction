import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, recall_score

from bootstrap_eval import bootstrap_ci, predict
from hpo import tex_escape

RULES = ("prevalence", "sensitivity", "youden")


def choose_threshold(y, p, rule="prevalence", pos_label=1, target=0.8):
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    u = np.unique(p)
    cand = (u[:-1] + u[1:]) / 2
    pred = (p[None, :] > cand[:, None]).astype(int)
    recall = {c: (pred[:, y == c] == c).mean(axis=1) for c in (0, 1)}
    sens, spec = recall[pos_label], recall[1 - pos_label]
    if rule == "prevalence":
        score = -np.abs(pred.mean(axis=1) - y.mean())
    elif rule == "sensitivity":
        if not (sens >= target).any():
            raise ValueError(f"no threshold reaches a sensitivity of {target}")
        score = np.where(sens >= target, spec, -np.inf)
    elif rule == "youden":
        score = sens + spec - 1
    else:
        raise ValueError(f"unknown rule {rule!r}; use one of {RULES}")
    return float(cand[np.argmax(score)])


def select_threshold(y, p, rule="prevalence", pos_label=1, target=0.8):
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    candidate = choose_threshold(y, p, rule, pos_label, target)
    better = matthews_corrcoef(y, predict(p, candidate)) > matthews_corrcoef(y, predict(p, 0.5))
    return (candidate if better else 0.5), candidate


def threshold_report(y, p, threshold, pos_label=1, n_boot=2000, alpha=0.05, seed=42):
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)

    def metrics(rows, t):
        pred = predict(p[rows], t)
        return np.array([matthews_corrcoef(y[rows], pred),
                         recall_score(y[rows], pred, pos_label=pos_label, zero_division=np.nan),
                         recall_score(y[rows], pred, pos_label=1 - pos_label, zero_division=np.nan)])

    rows = np.arange(len(y))
    default, chosen = metrics(rows, 0.5), metrics(rows, threshold)
    lo, hi = bootstrap_ci(lambda r: metrics(r, threshold) - metrics(r, 0.5), len(y),
                          n_boot=n_boot, alpha=alpha, seed=seed)
    return pd.DataFrame({"metric": ["MCC", "Sensitivity", "Specificity"], "at_default": default,
                         "at_threshold": chosen, "difference": chosen - default,
                         "ci_lower": lo, "ci_upper": hi})


def thresholds_latex(report, caption, label="tab:thresholds", decimals=3):
    def num(v):
        return f"{v:.{decimals}f}".replace("-", "$-$")

    blocks = []
    for model, d in report.groupby("model", sort=False):
        lines = []
        for i, r in enumerate(d.itertuples()):
            first = i == 0
            lines.append(" & ".join([
                tex_escape(model) if first else "", num(r.candidate) if first else "",
                num(r.selected) if first else "", tex_escape(r.metric), num(r.at_default),
                num(r.at_threshold), f"{num(r.difference)} ({num(r.ci_lower)} to {num(r.ci_upper)})"]) + r" \\")
        blocks.append("\n".join(lines))
    return ("\\begin{table}[ht]\n\\centering\n"
            f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
            "\\begin{tabular}{lcclccc}\n\\toprule\n"
            "Model & Rule threshold & Selected & Metric & At 0.5 & At rule threshold & "
            "Difference (95\\% CI) \\\\\n\\midrule\n"
            + "\n\\midrule\n".join(blocks) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")