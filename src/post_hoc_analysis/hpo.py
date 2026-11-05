import re
from pathlib import Path

import numpy as np
import pandas as pd

_TEX = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
        "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "–": "--"}


def tex_escape(text):
    return re.sub(r"[\\&%$#_{}~^–]", lambda m: _TEX[m.group()], str(text))


def read_trials(source):
    df = pd.read_csv(source) if isinstance(source, (str, Path)) else source
    return df[df["state"] == "COMPLETE"].sort_values("number").reset_index(drop=True)


def _kind(s):
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        v = s.dropna()
        return "integer" if np.all(v == np.round(v)) else "float"
    return "categorical"


def _fmt(x, kind):
    if pd.isna(x):
        return "None"
    if kind == "integer":
        return f"{int(x)}"
    return f"{x:.3g}" if kind == "float" else str(x)


def _ranked(df, objective, direction):
    return df.sort_values([objective, "number"], ascending=[direction == "minimize", True])


def _table(model, caption, label, header, rows, columns):
    body = "\n".join("\\midrule" if r is None else " & ".join(tex_escape(c) for c in r) + r" \\"
                     for r in rows)
    return ("\\begin{table}[ht]\n\\centering\n"
            f"\\caption{{{caption}}}\n"
            f"\\label{{tab:{label}-{re.sub(r'[^a-z0-9]+', '-', model.lower())}}}\n"
            f"\\begin{{tabular}}{{{columns}}}\n\\toprule\n"
            + " & ".join(tex_escape(h) for h in header) + " \\\\\n\\midrule\n"
            f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}")


def _summary(df, objective, direction):
    best = _ranked(df, objective, direction).iloc[0]
    rows = []
    for col in [c for c in df.columns if c.startswith("params_")]:
        s, kind = df[col], _kind(df[col])
        if kind == "categorical":
            space = "{" + ", ".join(dict.fromkeys(_fmt(x, kind) for x in s)) + "}"
        else:
            space = f"{_fmt(s.min(), kind)}–{_fmt(s.max(), kind)}"
        rows.append((col.removeprefix("params_"), kind, space, _fmt(best[col], kind)))
    return best, rows


def search_space_table(trials, objective="MCC", direction="maximize"):
    rows = [(model, *r) for model, source in trials.items()
            for r in _summary(read_trials(source), objective, direction)[1]]
    return pd.DataFrame(rows, columns=["model", "hyperparameter", "type", "search_space", "selected"])


def search_space_latex(trials, objective="MCC", direction="maximize"):
    blocks = []
    for model, source in trials.items():
        df = read_trials(source)
        best, rows = _summary(df, objective, direction)
        blocks.append(_table(
            model, f"Hyperparameter search space for {tex_escape(model)} ({len(df)} trials; best "
                   f"{tex_escape(objective)} = {best[objective]:.3f} in trial {int(best['number'])}).",
            "hpo", ["Hyperparameter", "Type", "Search space", "Selected"], rows, "llll"))
    return "\n\n".join(blocks)


def best_trials(trials, n=5, objective="MCC", direction="maximize", decimals=3):
    tables = {}
    for model, source in trials.items():
        df = read_trials(source)
        top = _ranked(df, objective, direction).head(n)
        rows = {objective: [f"{v:.{decimals}f}" for v in top[objective]]}
        rows.update({c.removeprefix("params_"): [_fmt(v, _kind(df[c])) for v in top[c]]
                     for c in df.columns if c.startswith("params_")})
        table = pd.DataFrame(rows, index=[f"Trial {int(t)}" for t in top["number"]]).T
        table.attrs["n_trials"] = len(df)
        tables[model] = table
    return tables


def best_trials_latex(trials, n=5, objective="MCC", direction="maximize", decimals=3):
    blocks = []
    for model, t in best_trials(trials, n, objective, direction, decimals).items():
        rows = [[name, *values] for name, values in zip(t.index, t.to_numpy())]
        blocks.append(_table(
            model, f"The {t.shape[1]} best of {t.attrs['n_trials']} completed hyperparameter optimization "
                   f"trials for {tex_escape(model)}, ranked by {tex_escape(objective)}; the first "
                   "column is the selected configuration.",
            "hpo-best", ["Hyperparameter", *t.columns], [rows[0], None, *rows[1:]], "l" + "c" * t.shape[1]))
    return "\n\n".join(blocks)