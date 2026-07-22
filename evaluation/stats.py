"""Statistical tests for comparing models on LEWS 1.0.

DeLong test: Sun & Xu (2014) fast implementation of DeLong et al. (1988).
Paired bootstrap: Efron & Tibshirani (1993), 10,000 resamples, fixed seed.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score


def _midrank(x):
    order = np.argsort(x)
    ranks = np.empty(len(x))
    i = 0
    xs = x[order]
    while i < len(x):
        j = i
        while j < len(x) and xs[j] == xs[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1
        i = j
    return ranks


def delong_p(y, s1, s2):
    """Two-sided DeLong p-value for AUROC(s1) vs AUROC(s2). Returns (delta, p)."""
    y = np.asarray(y)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    m, n = len(pos), len(neg)
    aucs, v01s, v10s = [], [], []
    for s in (np.asarray(s1), np.asarray(s2)):
        tx, ty = s[pos], s[neg]
        tz = np.concatenate([tx, ty])
        rz, rx, ry = _midrank(tz), _midrank(tx), _midrank(ty)
        auc = (rz[:m].sum() - m * (m + 1) / 2) / (m * n)
        aucs.append(auc)
        v01s.append((rz[:m] - rx) / n)
        v10s.append(1.0 - (rz[m:] - ry) / m)
    v01, v10 = np.array(v01s), np.array(v10s)
    S = np.cov(v01) / m + np.cov(v10) / n
    d = aucs[0] - aucs[1]
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    if var <= 0:
        return d, 1.0
    z = d / np.sqrt(var)
    return d, 2 * stats.norm.sf(abs(z))


def boot_p(y, s1, s2, B=10000, seed=42):
    """Paired bootstrap two-sided p and 95% CI for AUROC(s1) - AUROC(s2)."""
    y, s1, s2 = np.asarray(y), np.asarray(s1), np.asarray(s2)
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        d.append(roc_auc_score(y[i], s1[i]) - roc_auc_score(y[i], s2[i]))
    d = np.array(d)
    return 2 * min((d <= 0).mean(), (d >= 0).mean()), np.percentile(d, [2.5, 97.5])


def auroc_ci(y, s, B=10000, seed=42):
    """AUROC with 95% bootstrap CI."""
    y, s = np.asarray(y), np.asarray(s)
    rng = np.random.default_rng(seed)
    a = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        a.append(roc_auc_score(y[i], s[i]))
    lo, hi = np.percentile(a, [2.5, 97.5])
    return roc_auc_score(y, s), lo, hi
