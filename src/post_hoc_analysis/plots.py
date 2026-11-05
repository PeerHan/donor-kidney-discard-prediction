import itertools
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import PrecisionRecallDisplay, confusion_matrix

from bootstrap_eval import bootstrap_ci, predict
from hpo import read_trials

ACCENT, INK, INK_2, MUTED, GRID, AXIS = "#2a78d6", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
BLUES_DOTS = LinearSegmentedColormap.from_list(
    "blues_dots", ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"])
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def _frame(source):
    return pd.read_csv(source) if isinstance(source, (str, Path)) else source


def _style(ax, title, grid="x"):
    ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold")
    ax.grid(axis=grid, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.tick_params(axis="y", labelcolor=INK)


def _num(v, decimals):
    return f"{v:.{decimals}f}".replace("-", "−")


def _forest(ax, d, title, label, null, decimals, header, xlabel):
    y = np.arange(len(d))[::-1]
    _style(ax, title)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    if null is not None:
        ax.axvline(null, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
    ax.hlines(y, d.ci_lower, d.ci_upper, color=ACCENT, linewidth=2, capstyle="round", zorder=2)
    ax.plot(d.estimate, y, "o", color=ACCENT, markersize=8, markeredgecolor="white",
            markeredgewidth=1.5, zorder=3)
    ax.set_yticks(y, d[label])
    ax.set_ylim(-0.6, len(d) - 0.4)
    xs = [d.ci_lower.min(), d.ci_upper.max(), d.estimate.min(), d.estimate.max()]
    lo, hi = np.nanmin(xs + ([null] if null is not None else [])), np.nanmax(xs + ([null] if null is not None else []))
    ax.set_xlim(lo - 0.08 * (hi - lo), hi + 0.08 * (hi - lo))

    ax.annotate(header, xy=(1.04, 1), xycoords="axes fraction", xytext=(0, 6),
                textcoords="offset points", va="baseline", color=INK_2, fontsize=9)
    for yi, r in zip(y, d.itertuples()):
        if np.isnan(r.estimate): 
            continue
        ci = "" if np.isnan(r.ci_lower) else f" ({_num(r.ci_lower, decimals)} to {_num(r.ci_upper, decimals)})"
        ax.text(1.04, yi, _num(r.estimate, decimals) + ci, transform=ax.get_yaxis_transform(),
                va="center", color=INK, fontsize=9)


def forest_plots(res, out_dir=None, null=None, decimals=3, header="Estimate (95% CI)", fmt="png",
                 by="metric", label="model", xlabel=None, ncols=None, out_file=None, sharey=False):
    if null is None and "reference" in res.columns:
        null = 0
    panels = list(res.groupby(by, sort=False))
    args = (label, null, decimals, header, xlabel)
    if ncols:
        ncols = min(ncols, len(panels))
        nrows = -(-len(panels) // ncols)
        rows = max(len(d) for _, d in panels)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, (1.0 + 0.38 * rows) * nrows),
                                 squeeze=False, sharey=sharey)
        for ax, (value, d) in zip(axes.flat, panels):
            _forest(ax, d, value, *args)
        for ax in axes.flat[len(panels):]:
            ax.set_visible(False)
        fig.tight_layout()
        if out_file is not None:
            fig.savefig(out_file, dpi=300, bbox_inches="tight")
        return fig

    figs = {}
    for value, d in panels:
        fig, ax = plt.subplots(figsize=(6.5, 0.8 + 0.38 * len(d)))
        _forest(ax, d, value, *args)
        if out_dir is not None:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            fig.savefig(Path(out_dir) / f"{value.replace(' ', '_')}.{fmt}", dpi=300, bbox_inches="tight")
            plt.close(fig)
        figs[value] = fig
    return figs


def shap_top_features(shap, n=10, labels=None, ncols=2, ci=True, groups=None, sharex=False,
                      xlabel="Mean |SHAP value|", out_file=None):
    labels = labels or {}
    tables = {m: _frame(s) for m, s in shap.items()}
    nrows = -(-len(tables) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, (0.8 + 0.34 * n) * nrows),
                             sharex=sharex, squeeze=False)
    for ax, (model, df) in zip(axes.flat, tables.items()):
        _style(ax, model)
        importance = df.abs().mean()
        top = importance[importance > 0].nlargest(n)
        y = np.arange(len(top))[::-1]
        ax.barh(y, top.to_numpy(), height=0.6, color=ACCENT, zorder=2)
        if ci:
            a = df[top.index].abs().to_numpy()
            lo, hi = bootstrap_ci(lambda rows: a[rows].mean(axis=0), len(a), groups)
            ax.hlines(y, lo, hi, color=INK_2, linewidth=1.25, zorder=3)
        ax.axvline(0, color=AXIS, linewidth=0.8, zorder=1)
        ax.set_xlim(left=0)
        ax.set_yticks(y, [textwrap.fill(labels.get(f, f.replace("_", " ")), 42) for f in top.index],
                      fontsize=8.5)

    for ax in axes.flat[len(tables):]:
        ax.set_visible(False)
    for ax in axes.flat[max(0, len(tables) - ncols):len(tables)]:
        ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
        ax.tick_params(labelbottom=True)
    fig.tight_layout()
    if out_file is not None:
        fig.savefig(out_file, dpi=300, bbox_inches="tight")
    return fig

def confusion_matrices(y, probs, thresholds=0.5, pos_label=1, class_names=("class 0", "class 1"),
                       ncols=3, out_file=None):
    y = np.asarray(y, dtype=int)
    probs = {m: np.asarray(p, dtype=float) for m, p in probs.items()}
    thr = thresholds if isinstance(thresholds, dict) else dict.fromkeys(probs, thresholds)
    order = [pos_label, 1 - pos_label]
    names = [class_names[c] for c in order]
    nrows = -(-len(probs) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 3.1 * nrows), squeeze=False)
    for ax, (model, p) in zip(axes.flat, probs.items()):
        cm = confusion_matrix(y, predict(p, thr[model]), labels=order)
        share = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(share, cmap=BLUES, vmin=0, vmax=1)
        for (i, j), count in np.ndenumerate(cm):
            ink = "white" if share[i, j] > 0.6 else INK
            ax.text(j, i - 0.1, f"{count}", ha="center", va="center", color=ink, fontsize=13,
                    fontweight="bold")
            ax.text(j, i + 0.18, f"{share[i, j]:.0%}", ha="center", va="center", color=ink, fontsize=9)
        ax.set_title(model, loc="left", color=INK, fontsize=11, fontweight="bold")
        ax.set_xticks([0, 1], names)
        ax.set_yticks([0, 1], names)
        ax.tick_params(length=0, labelsize=9, colors=INK)
        for spine in ax.spines.values():
            spine.set_visible(False)

    for ax in axes.flat[len(probs):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("True class", color=INK_2, fontsize=9)
    for ax in axes.flat[max(0, len(probs) - ncols):len(probs)]:  # lowest panel of each column
        ax.set_xlabel("Predicted class", color=INK_2, fontsize=9)
    fig.tight_layout()
    if out_file is not None:
        fig.savefig(out_file, dpi=300, bbox_inches="tight")
    return fig


def hpo_history(trials, objective="MCC", window=20, direction="maximize", ncols=2, out_file=None):
    tables = {m: read_trials(s) for m, s in trials.items()}
    nrows = -(-len(tables) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.0 * nrows), sharey=True,
                             squeeze=False)
    for ax, (model, d) in zip(axes.flat, tables.items()):
        _style(ax, model, grid="y")
        v = d[objective]
        best = v.cummax() if direction == "maximize" else v.cummin()
        ax.scatter(d["number"], v, s=9, color=MUTED, alpha=0.45, linewidths=0, zorder=2, label="Trial")
        ax.step(d["number"], best, where="post", color=INK_2, linewidth=1.25, zorder=3,
                label="Best so far")
        ax.plot(d["number"], v.rolling(window, min_periods=1).mean(), color=ACCENT, linewidth=2,
                zorder=4, label=f"Moving average ({window} trials)")

    for ax in axes.flat[len(tables):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel(objective, color=INK_2, fontsize=9)
    for ax in axes.flat[max(0, len(tables) - ncols):len(tables)]:  # lowest panel of each column
        ax.set_xlabel("Trial", color=INK_2, fontsize=9)
        ax.tick_params(labelbottom=True)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=9,
               labelcolor=INK_2, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.3 / fig.get_figheight()))  # 0.3 in for the legend
    if out_file is not None:
        fig.savefig(out_file, dpi=300, bbox_inches="tight")
    return fig


def _mean_jaccard(S):
    S = S.astype(int)
    a = S.sum(axis=1)
    inter, iu, pair = S @ S.T, np.triu_indices(len(S), 1), a[:, None] + a[None, :]
    expected = np.outer(a, a) / S.shape[1]
    return (inter / (pair - inter))[iu].mean(), (expected / (pair - expected))[iu].mean()


def feature_selection_study(studies, objective="MCC Loss", metric="MCC", top=50, n_random=500, seed=0,
                            out_file=None):
    gen = "system_attrs_nsga2:generation"
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(len(studies), 3, figsize=(13, 3.4 * len(studies)), squeeze=False)
    rows = []
    for (a, b, c), (model, source) in zip(axes, studies.items()):
        d = read_trials(source)
        X = d.filter(like="params_").astype(bool).to_numpy()
        n, y, best = X.sum(axis=1), d[metric], d[objective].idxmin()
        full = n == X.shape[1]
        base = y[full].iloc[0] if full.any() else None

        title = f"{model}: {len(d)} trials" + (f" in {d[gen].nunique()} generations" if gen in d else "")
        _style(a, title, grid="y")
        a.scatter(d.number, y, s=8, color=MUTED, alpha=0.5, linewidths=0, zorder=2, label="Trial")
        if gen in d:
            g = d.groupby(gen)
            a.plot(g.number.mean(), g[metric].mean(), color=ACCENT, linewidth=2, marker="o", markersize=4,
                   zorder=3, label="Generation mean")
            for start in g.number.min().iloc[1:]:
                a.axvline(start - 0.5, color=GRID, linewidth=0.8, zorder=1)
        a.set_xlabel("Trial", color=INK_2, fontsize=9)
        a.set_ylabel(metric, color=INK_2, fontsize=9)

        _style(b, "Number of selected features", grid="y")
        b.sharey(a)
        b.tick_params(labelleft=False)
        b.scatter(n[~full], y[~full], s=8, color=MUTED, alpha=0.5, linewidths=0, zorder=2)
        b.set_xlabel("Selected features", color=INK_2, fontsize=9)
        b.annotate(f"{n[best]} features", (n[best], y[best]), xytext=(6, -7), textcoords="offset points",
                   va="top", color=INK_2, fontsize=8.5)
        penalty = np.median((d[objective] + y) / n)
        if np.allclose(d[objective], -y + penalty * n):
            xs = np.array([n[~full].min(), n[~full].max()])
            b.plot(xs, y[best] + penalty * (xs - n[best]), color=ACCENT, linewidth=1.2, alpha=0.6, zorder=3,
                   label=f"Same objective as the selected trial ({penalty:g} per feature)")
        for ax, x in ((a, d.number[best]), (b, n[best])):
            ax.plot(x, y[best], "o", color=ACCENT, markersize=8, markeredgecolor="white", markeredgewidth=1.5,
                    zorder=4, label="Selected trial")
            if base is not None:
                ax.axhline(base, color=INK_2, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1,
                           label="Trial with all features")

        def profile(M):
            return np.sort(M.mean(axis=0))[::-1]

        S = X[d[objective].nsmallest(top).index]
        draws = [X[~full][rng.choice((~full).sum(), top, replace=False)] for _ in range(n_random)]
        lo, hi = np.quantile([profile(M) for M in draws], [0.025, 0.975], axis=0)
        jac, jac_sets = _mean_jaccard(S)
        jac_trials = np.median([_mean_jaccard(M)[0] for M in draws])
        rank = np.arange(1, X.shape[1] + 1)
        k = int(round(S.sum(axis=1).mean()))
        _style(c, f"Agreement between the {top} best trials", grid="y")
        c.plot(rank, profile(S), color=ACCENT, linewidth=2, zorder=3, label=f"{top} best trials")
        c.fill_between(rank, lo, hi, color=GRID, linewidth=0, zorder=1, label=f"{top} random trials (95% range)")
        c.plot([1, k, k, X.shape[1]], [1, 1, 0, 0], color=INK_2, linewidth=1.5, linestyle=(0, (1, 2)),
               zorder=2, label="Perfect agreement (identical feature sets)")
        c.set(xlim=(1, X.shape[1]), ylim=(-0.02, 1.02))
        c.set_xlabel("Features, sorted from most to least often used", color=INK_2, fontsize=9)
        c.set_ylabel(f"Share of the {top} trials using the feature", color=INK_2, fontsize=9)
        c.text(0.03, 0.05, f"Mean Jaccard index\n{top} best trials: {jac:.2f}\nrandom trials: {jac_trials:.2f}\n"
               f"random sets: {jac_sets:.2f}", transform=c.transAxes, ha="left", va="bottom",
               color=INK_2, fontsize=8.5)
        c.set_xlabel("Features, sorted by inclusion frequency", color=INK_2, fontsize=9)
        c.set_ylabel("Inclusion frequency", color=INK_2, fontsize=9)

        rows.append({"model": model, "trials": len(d), "generations": d[gen].nunique() if gen in d else None,
                     "features_per_trial": f"{n[~full].min()}–{n[~full].max()} (median {np.median(n[~full]):.0f})",
                     "selected_trial": int(d.number[best]), "selected_features": int(n[best]),
                     f"selected_{metric}": y[best], f"all_features_{metric}": base,
                     "share_of_trials_above_all_features": (y[~full] > base).mean() if base is not None else None,
                     f"jaccard_top{top}": jac, "jaccard_random_trials": jac_trials, "jaccard_random_sets": jac_sets})

    entries = {}
    for ax in axes[0]:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            entries.setdefault(label, handle)
    fig.legend(entries.values(), entries.keys(), loc="upper center", ncol=4, frameon=False, fontsize=9,
               labelcolor=INK_2, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.6 / fig.get_figheight()))
    if out_file is not None:
        fig.savefig(out_file, dpi=300, bbox_inches="tight")
    return fig, pd.DataFrame(rows)


def feature_overlap(selected, out_file=None, sort_subsets="-cardinality", **kwargs):
    from marsilea.upset import Upset, UpsetData  # pip install marsilea

    data = UpsetData.from_sets([set(f) for f in selected.values()], sets_names=list(selected))
    upset = Upset(data, sets_order=list(selected), sort_subsets=sort_subsets, color=ACCENT, **kwargs)
    upset.render()
    if out_file is not None:
        upset.save(out_file, dpi=300)
    return data


def pairwise_overlap(selected):
    sets = {m: set(f) for m, f in selected.items()}
    return pd.DataFrame([{"model_a": a, "model_b": b, "shared": len(sets[a] & sets[b]),
                          "jaccard": len(sets[a] & sets[b]) / len(sets[a] | sets[b])}
                         for a, b in itertools.combinations(sets, 2)])


def jaccard_heatmap(selected, out_file=None):
    models, pairs = list(selected), pairwise_overlap(selected)

    def grid(col):
        return pairs.pivot(index="model_b", columns="model_a", values=col).reindex(
            index=models[1:], columns=models[:-1])

    labels = grid("jaccard").map("{:.2f}".format) + "\n" + grid("shared").map("{:.0f}".format)
    fig, ax = plt.subplots(figsize=(0.9 * len(models) + 1, 0.9 * len(models)))
    sns.heatmap(grid("jaccard"), annot=labels, fmt="", cmap=BLUES, vmin=0, vmax=1, square=True,
                linewidths=2, cbar_kws={"label": "Jaccard index", "ticks": [0, 0.5, 1]}, ax=ax)
    ax.set_title("Jaccard index (shared features)", loc="left", color=INK, fontsize=11, fontweight="bold")
    ax.set(xlabel="", ylabel="")
    ax.tick_params(length=0, labelrotation=0)
    if out_file is not None:
        fig.savefig(out_file, dpi=300, bbox_inches="tight")
    return fig


def pr_curves(y, probs, class_names=("class 0", "class 1"), colors=None, out_file=None):
    y = np.asarray(y, dtype=int)
    colors = colors or dict(zip(probs, CATEGORICAL))
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, pos in zip(axes, (0, 1)):
        for i, (model, p) in enumerate(probs.items()):
            p = np.asarray(p, dtype=float)
            last = i == len(probs) - 1
            disp = PrecisionRecallDisplay.from_predictions(y, p if pos == 1 else 1 - p, pos_label=pos,
                                                           name=model, ax=ax, plot_chance_level=last)
            disp.line_.set(color=colors[model], linewidth=2)
            if last:
                disp.chance_level_.set(color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)))
        _style(ax, f"{class_names[pos]} as positive class", grid="both")
        ax.set(xlim=(0, 1), ylim=(0, 1.02))
        ax.set_xlabel("Recall", color=INK_2, fontsize=9)
        ax.set_ylabel("Precision", color=INK_2, fontsize=9)
        ax.legend(loc="lower left", frameon=False, fontsize=8, labelcolor=INK_2)
    fig.tight_layout()
    if out_file is not None:
        fig.savefig(out_file, dpi=300, bbox_inches="tight")
    return fig