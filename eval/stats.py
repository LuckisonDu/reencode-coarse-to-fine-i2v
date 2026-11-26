#!/usr/bin/env python3
"""Statistics used in the paper.

  paired_cluster_bootstrap  paired difference A - B over clip-seed pairs, 95% interval from a bootstrap
                            over first frames (input clips), keeping both seeds of a clip together
                            (5000 resamples, numpy default_rng(0)).
  sign_test                 exact two-sided sign test over clips (ties dropped).
  net_preference            blind pairwise study: per clip, the mean over raters of +1 (prefers A),
                            -1 (prefers B), 0 (similar); reported with a clip bootstrap interval.
  recovery                  share of a gap closed by a variant: (variant - worse) / (better - worse).

Inputs are dicts keyed by (seed, clip) -> value.
"""
import math

import numpy as np


def paired_cluster_bootstrap(a, b, n_boot=5000, seed=0):
    keys = sorted(set(a) & set(b))
    diff = np.array([a[k] - b[k] for k in keys])
    clips = sorted({k[1] for k in keys}); idx = {c: i for i, c in enumerate(clips)}
    sums, cnts = np.zeros(len(clips)), np.zeros(len(clips))
    for k, d in zip(keys, diff):
        sums[idx[k[1]]] += d; cnts[idx[k[1]]] += 1
    rng = np.random.default_rng(seed); bs = []
    for _ in range(n_boot):
        j = rng.integers(0, len(clips), len(clips)); bs.append(sums[j].sum() / cnts[j].sum())
    return {"n_pairs": len(keys), "n_clips": len(clips), "mean": float(diff.mean()),
            "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "a_higher": int((diff > 0).sum())}


def sign_test(wins, losses):
    n, k = wins + losses, min(wins, losses)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def net_preference(votes_by_clip, n_boot=5000, seed=0):
    """votes_by_clip: {clip: [vote of rater 1, vote of rater 2, ...]} with votes in {+1, 0, -1}."""
    clips = sorted(votes_by_clip)
    per = np.array([np.mean(votes_by_clip[c]) for c in clips])
    summed = np.array([np.sum(votes_by_clip[c]) for c in clips])
    rng = np.random.default_rng(seed)
    bs = [per[rng.integers(0, len(per), len(per))].mean() for _ in range(n_boot)]
    wins, ties, losses = int((summed > 0).sum()), int((summed == 0).sum()), int((summed < 0).sum())
    return {"clips": len(clips), "preferred": wins, "tied": ties, "dispreferred": losses, "p_sign": sign_test(wins, losses),
            "net": float(per.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}


def recovery(variant, worse, better):
    return (variant - worse) / (better - worse)
