"""Score prediction files against LEWS 1.0.

A prediction file is JSON: {"model": str, "predictions": {task_id: prob-or-null}}.
A null (or missing) prediction means the model did not answer that case. Unanswered
cases are EXCLUDED from scoring and reported as coverage, never imputed: imputing a
non-answer as 0.5 launders a failure into a mid-range prediction.

  python3 evaluation/evaluate.py results/paper_v1/predictions_*.json
  python3 evaluation/evaluate.py preds.json --compare results/paper_v1/predictions_claude-sonnet-4.6.json
  python3 evaluation/evaluate.py preds.json --temporal
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from stats import auroc_ci, boot_p, delong_p

TASKS = os.path.join(os.path.dirname(__file__), "..", "tasks", "lews_v1.jsonl")


def load_tasks():
    rows = [json.loads(l) for l in open(TASKS)]
    return rows


def load_preds(path, ids):
    d = json.load(open(path))
    p = d["predictions"]
    return d.get("model", os.path.basename(path)), np.array(
        [np.nan if p.get(i) is None else float(p[i]) for i in ids], dtype=float)


def brier_ece(y, s, bins=10):
    brier = float(np.mean((s - y) ** 2))
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (s >= lo) & (s < hi) if hi < 1 else (s >= lo) & (s <= hi)
        if m.sum():
            ece += m.mean() * abs(s[m].mean() - y[m].mean())
    return brier, ece


def score(name, y, s, ref=None, refname=""):
    m = ~np.isnan(s)
    ya, sa = y[m], s[m]
    auc, lo, hi = auroc_ci(ya, sa)
    brier, ece = brier_ece(ya, sa)
    line = (f"{name:28} AUROC {auc:.3f} [{lo:.3f}, {hi:.3f}]  "
            f"coverage {int(m.sum())}/{len(y)}  Brier {brier:.3f}  ECE {ece:.3f}")
    if ref is not None:
        ra = ref[m]
        d, pd_ = delong_p(ya, ra, sa)
        pb, _ = boot_p(ya, ra, sa)
        line += f"  | vs {refname}: d={d:+.3f} DeLong p={pd_:.3f} boot p={pb:.3f}"
    print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pred_files", nargs="+")
    ap.add_argument("--compare", help="reference prediction file for paired tests")
    ap.add_argument("--temporal", action="store_true",
                    help="score only the temporal holdout (latest 25%% of cases by cutoff)")
    ap.add_argument("--petitions-only", action="store_true",
                    help="score only the 97 real petitions (grant vs deny); excludes "
                         "constructed negatives. Much harder; expect wide CIs (19 negatives).")
    args = ap.parse_args()

    tasks = load_tasks()
    ids = [t["id"] for t in tasks]
    y = np.array([t["label"] for t in tasks], dtype=float)

    keep = np.ones(len(tasks), dtype=bool)
    if args.petitions_only:
        keep = np.array([t["outcome"] in ("grant", "deny") for t in tasks])
        print(f"petitions only: n={int(keep.sum())} "
              f"({int(y[keep].sum())} granted / {int((1-y[keep]).sum())} denied)\n")
    if args.temporal:
        order = np.argsort([t["cutoff"] for t in tasks], kind="stable")
        n_test = int(len(tasks) * 0.25)   # canonical split: matches the paper exactly
        test = order[-n_test:]
        keep[:] = False
        keep[test] = True
        cut = [tasks[i]["cutoff"] for i in test]
        print(f"temporal holdout: n={keep.sum()} ({min(cut)} .. {max(cut)}); "
              f"models must not have trained on these petitions\n")

    ref = refname = None
    if args.compare:
        refname, ref_full = load_preds(args.compare, ids)
        ref = ref_full[keep]

    for path in args.pred_files:
        name, s = load_preds(path, ids)
        score(name, y[keep], s[keep], ref=ref, refname=refname)


if __name__ == "__main__":
    main()
