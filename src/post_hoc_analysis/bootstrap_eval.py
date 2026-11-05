import warnings

import numpy as np
import pandas as pd
from scipy.stats import DegenerateDataWarning, bootstrap
from sklearn.metrics import (average_precision_score, brier_score_loss, matthews_corrcoef,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.utils.class_weight import compute_sample_weight


def load_predictions(files, prob_col="Prob_1", target_col="Targets"):
    tables = {m: pd.read_csv(f) for m, f in files.items()}
    for m, t in tables.items():
        if t[target_col].isna().any():
            raise ValueError(f"{m}: {t[target_col].isna().sum()} of {len(t)} targets are missing - "
                             "check the export (pandas index alignment?)")
    y = next(iter(tables.values()))[target_col].to_numpy().astype(int)
    for m, t in tables.items():
        if not np.array_equal(t[target_col].to_numpy(), y):
            raise ValueError(f"{m}: targets differ from the first file - rows are not aligned")
    return y, {m: t[prob_col].to_numpy() for m, t in tables.items()}


def predict(p, thr):
    return (np.asarray(p) > thr).astype(int)


def _metrics(y, p, thr, pos, class_names):
    neg = 1 - pos
    pred = predict(p, thr)
    prec, rec, f1, _ = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=np.nan)
    prob = {1: p, 0: 1 - p} 
    both = 0 < y.sum() < len(y)
    pos_name, neg_name = class_names[pos], class_names[neg]
    return {
        "Sensitivity": rec[pos],
        "Specificity": rec[neg],
        "PPV": prec[pos],
        "NPV": prec[neg],
        f"F1 ({pos_name})": f1[pos],
        f"F1 ({neg_name})": f1[neg],
        "MCC": matthews_corrcoef(y, pred),
        "AUC": roc_auc_score(y, p) if both else np.nan,
        f"PR-AUC ({pos_name})": average_precision_score(y, prob[pos], pos_label=pos) if both else np.nan,
        f"PR-AUC ({neg_name})": average_precision_score(y, prob[neg], pos_label=neg) if both else np.nan,
        "Brier": brier_score_loss(y, p),
        "Balanced Brier": brier_score_loss(y, p, sample_weight=compute_sample_weight("balanced", y)),
    }


def bootstrap_ci(statistic, n, groups=None, n_boot=2000, alpha=0.05, seed=42, method="BCa"):
    rows = np.arange(n)
    units = list(pd.Series(rows).groupby(rows if groups is None else np.asarray(groups))
                 .indices.values())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DegenerateDataWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        res = bootstrap((np.arange(len(units)),),
                        lambda u: statistic(np.concatenate([units[i] for i in u])),
                        n_resamples=n_boot, vectorized=False, confidence_level=1 - alpha,
                        method=method, rng=seed)
    dist = res.bootstrap_distribution
    constant = np.ptp(dist, axis=-1) == 0
    low = np.where(constant, dist[..., 0], res.confidence_interval.low)
    high = np.where(constant, dist[..., 0], res.confidence_interval.high)
    if np.isnan(low).any() or np.isnan(high).any():
        warnings.warn(f"No bootstrap CI for statistic(s) {np.flatnonzero(np.isnan(low) | np.isnan(high)).tolist()} "
                      "(undefined in some resamples?)")
    return low, high


def bootstrap_eval(y, probs, thresholds=0.5, pos_label=1, class_names=("class 0", "class 1"),
                   groups=None, reference=None, n_boot=2000, alpha=0.05, seed=42, method="BCa"):
    y = np.asarray(y, dtype=int)
    probs = {m: np.asarray(p, dtype=float) for m, p in probs.items()}
    thr = thresholds if isinstance(thresholds, dict) else dict.fromkeys(probs, thresholds)

    def table(rows):
        t = pd.DataFrame({m: _metrics(y[rows], p[rows], thr[m], pos_label, class_names)
                          for m, p in probs.items()}).T
        if reference is not None:
            t = (t - t.loc[reference]).drop(index=reference).rename(index=lambda m: f"{m} - {reference}")
        return t

    point = table(np.arange(len(y)))
    lo, hi = bootstrap_ci(lambda rows: table(rows).to_numpy(dtype=float).ravel(),
                          len(y), groups, n_boot, alpha, seed, method)
    n_models, n_metrics = point.shape
    out = pd.DataFrame({"model": np.repeat(point.index, n_metrics),
                        "metric": np.tile(point.columns, n_models),
                        "positive_class": class_names[pos_label],
                        "estimate": point.to_numpy(dtype=float).ravel(),
                        "ci_lower": lo, "ci_upper": hi})
    if reference is not None:
        out.insert(1, "reference", reference)
    return out