import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.utils.class_weight import compute_sample_weight

from bootstrap_eval import bootstrap_ci
from hpo import tex_escape
from plots import ACCENT, AXIS, INK, INK_2, MUTED, _style

METHODS = ("sigmoid", "isotonic", "temperature")


class _Probabilities(ClassifierMixin, BaseEstimator):

    def fit(self, X=None, y=None):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = np.asarray(X, dtype=float)[:, 0]
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


def calibrate(p_cal, y_cal, p_new, method="sigmoid"):
    cal = CalibratedClassifierCV(FrozenEstimator(_Probabilities().fit()), method=method, ensemble=False)
    cal.fit(np.asarray(p_cal, dtype=float)[:, None], np.asarray(y_cal, dtype=int))
    return cal.predict_proba(np.asarray(p_new, dtype=float)[:, None])[:, 1]


def calibrate_all(y_cal, probs_cal, probs_test, methods=METHODS):
    return {m: {"Uncalibrated": np.asarray(p, dtype=float),
                **{k.capitalize(): calibrate(probs_cal[m], y_cal, p, k) for k in methods}}
            for m, p in probs_test.items()}


def balanced_brier(y, p):
    return brier_score_loss(y, p, sample_weight=compute_sample_weight("balanced", y))


def _clipped_logit(p, eps):
    return logit(np.clip(np.asarray(p, dtype=float), eps, 1 - eps))  # trees and isotonic can return 0 or 1


def calibration_intercept(y, p, eps=1e-6):
    y, lp = np.asarray(y, dtype=int), _clipped_logit(p, eps)
    return brentq(lambda a: np.sum(y - expit(a + lp)), -20, 20)


def calibration_slope(y, p, eps=1e-6):
    lp = _clipped_logit(p, eps)
    return LogisticRegression(C=np.inf).fit(lp[:, None], np.asarray(y, dtype=int)).coef_[0, 0]


def calibration_stats(y, p, eps=1e-6):
    return {"Brier": brier_score_loss(y, p), "Balanced Brier": balanced_brier(y, p),
            "Intercept": calibration_intercept(y, p, eps), "Slope": calibration_slope(y, p, eps)}


def calibration_curves(y_test, calibrated, pos_label=1, class_names=("class 0", "class 1"),
                       n_bins=10, strategy="quantile", out_file=None):
    y_test = np.asarray(y_test, dtype=int)
    columns = list(next(iter(calibrated.values())))
    fig, axes = plt.subplots(len(calibrated), len(columns), sharex=True, sharey=True, squeeze=False,
                             figsize=(2.8 * len(columns) + 0.4, 2.9 * len(calibrated)))
    rows = []
    for r, (model, preds) in enumerate(calibrated.items()):
        for c, method in enumerate(columns):
            yy, pp = (y_test, preds[method]) if pos_label == 1 else (1 - y_test, 1 - preds[method])
            stats = calibration_stats(yy, pp)
            rows.append({"model": model, "method": method, "probability": f"P({class_names[pos_label]})",
                         **stats})  # intercept and slope refer to this probability
            observed, predicted = calibration_curve(yy, pp, n_bins=n_bins, strategy=strategy)

            ax = axes[r, c]
            _style(ax, method if r == 0 else "", grid="both")
            ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, zorder=1)  # perfect calibration
            ax.plot(predicted, observed, "-o", color=ACCENT, linewidth=2, markersize=6,
                    markeredgecolor="white", markeredgewidth=1.2, zorder=3)
            label = (f"Brier {stats['Brier']:.3f}\nBal. Brier {stats['Balanced Brier']:.3f}\n"
                     f"Intercept {stats['Intercept']:.2f}\nSlope {stats['Slope']:.2f}").replace("-", "\u2212")
            ax.text(0.96, 0.04, label, transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                    color=INK_2, linespacing=1.4)
            ax.set(xlim=(0, 1), ylim=(0, 1), xticks=[0, 0.5, 1], yticks=[0, 0.5, 1], aspect="equal")
        axes[r, 0].set_ylabel(f"Observed fraction ({class_names[pos_label]})", color=INK_2, fontsize=9)
        axes[r, 0].annotate(model, xy=(-0.42, 0.5), xycoords="axes fraction", rotation=90, ha="center",
                            va="center", color=INK, fontsize=11, fontweight="bold")
    for ax in axes[-1]:
        ax.set_xlabel(f"Predicted probability ({class_names[pos_label]})", color=INK_2, fontsize=9)
    fig.tight_layout()
    if out_file is not None:
        fig.savefig(out_file, dpi=300, bbox_inches="tight")
    return fig, pd.DataFrame(rows)


def calibration_overlay(y_test, calibrated, methods=("Uncalibrated", "Sigmoid"), pos_label=1,
                        class_names=("class 0", "class 1"), n_bins=10, strategy="quantile", ncols=3,
                        out_file=None):
    y_test = np.asarray(y_test, dtype=int)
    styles = (dict(color=MUTED, linewidth=1.5, markersize=5), dict(color=ACCENT, linewidth=2, markersize=6))
    ncols = min(ncols, len(calibrated))
    nrows = -(-len(calibrated) // ncols)
    fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True, squeeze=False,
                             figsize=(2.9 * ncols + 0.3, 3.0 * nrows + 0.3))
    for ax, (model, preds) in zip(axes.flat, calibrated.items()):
        _style(ax, model, grid="both")
        ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1, zorder=1)  # perfect calibration
        s = []
        for method, style in zip(methods, styles):
            yy, pp = (y_test, preds[method]) if pos_label == 1 else (1 - y_test, 1 - preds[method])
            observed, predicted = calibration_curve(yy, pp, n_bins=n_bins, strategy=strategy)
            ax.plot(predicted, observed, "-o", markeredgecolor="white", markeredgewidth=1.2, zorder=3,
                    label=method, **style)
            s.append(calibration_stats(yy, pp))
        ax.text(0.03, 0.97, (f"slope {s[0]['Slope']:.2f} → {s[1]['Slope']:.2f}\n"  # first -> second method
                             f"intercept {s[0]['Intercept']:.2f} → {s[1]['Intercept']:.2f}").replace("-", "−"),
                transform=ax.transAxes, ha="left", va="top", fontsize=8, color=INK_2, linespacing=1.4)
        ax.set(xlim=(0, 1), ylim=(0, 1), xticks=[0, 0.5, 1], yticks=[0, 0.5, 1], aspect="equal")
    for ax in axes.flat[len(calibrated):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel(f"Observed fraction ({class_names[pos_label]})", color=INK_2, fontsize=9)
    for ax in axes.flat[max(0, len(calibrated) - ncols):len(calibrated)]:  # lowest panel of each column
        ax.set_xlabel(f"Predicted probability ({class_names[pos_label]})", color=INK_2, fontsize=9)
        ax.tick_params(labelbottom=True)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(methods), frameon=False, fontsize=9,
               labelcolor=INK_2, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.3 / fig.get_figheight()))
    if out_file is not None:
        fig.savefig(out_file, dpi=300, bbox_inches="tight")
    return fig


def score_with_ci(y_test, calibrated, score=balanced_brier, groups=None, n_boot=2000, alpha=0.05, seed=42):
    y = np.asarray(y_test, dtype=int)
    rows = []
    for model, preds in calibrated.items():
        methods = list(preds)
        stat = lambda r: np.array([score(y[r], preds[k][r]) for k in methods])  # noqa: E731
        lo, hi = bootstrap_ci(stat, len(y), groups, n_boot, alpha, seed)
        rows += [(model, k, e, a, b) for k, e, a, b in zip(methods, stat(np.arange(len(y))), lo, hi)]
    return pd.DataFrame(rows, columns=["model", "method", "estimate", "ci_lower", "ci_upper"])


def scores_latex(scores, caption, label="tab:calibration", decimals=3, lower_is_better=True, methods=None):
    methods = methods or list(dict.fromkeys(scores["method"]))
    num = lambda v: f"{v:.{decimals}f}".replace("-", "$-$")  # noqa: E731
    sep = " to " if (scores[["ci_lower", "ci_upper"]] < 0).any().any() else "--"
    lines = []
    for model, d in scores.groupby("model", sort=False):
        d = d.set_index("method")
        best = None if lower_is_better is None else d["estimate"].idxmin() if lower_is_better else d["estimate"].idxmax()
        cells = []
        for k in methods:
            if k not in d.index:
                cells.append("")
                continue
            e, lo, hi = d.loc[k, ["estimate", "ci_lower", "ci_upper"]]
            text = num(e) + ("" if np.isnan(lo) else f" ({num(lo)}{sep}{num(hi)})")
            cells.append(f"\\textbf{{{text}}}" if k == best else text)
        lines.append(" & ".join([tex_escape(model), *cells]) + r" \\")
    return ("\\begin{table}[ht]\n\\centering\n"
            f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
            f"\\begin{{tabular}}{{l{'c' * len(methods)}}}\n\\toprule\n"
            + " & ".join(["Model", *map(tex_escape, methods)]) + " \\\\\n\\midrule\n"
            + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")