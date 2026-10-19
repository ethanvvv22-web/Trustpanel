"""Compare predicted rating distributions with what real consumers said.

Metrics map to decisions an innovation team actually makes:
- mean_abs_error: how far off is the predicted average star rating?
- wasserstein: how far off is the whole rating distribution (incl. the share of unhappy buyers)?
- negative_share_error: is the share of 1-2 star reviews (the "this failed me" group) right?
- pairwise_accuracy: given two products in the same category whose real averages differ by at
  least `min_gap`, does the predictor pick the one consumers liked more?
- spearman: how well does it rank products overall?
- bias / spread: is it systematically too positive, and too unanimous?
All with 95% bootstrap confidence intervals over products.
"""

import random
from itertools import combinations
from statistics import mean
from typing import Callable, Dict, List, Sequence

from scipy.stats import spearmanr

STARS = (1, 2, 3, 4, 5)
PREDICTORS = ("persona_panel", "persona_calibrated", "generic_llm", "generic_calibrated", "topic_prior")


def expected(dist: Sequence[float]) -> float:
    return sum(s * p for s, p in zip(STARS, dist))


def spread(dist: Sequence[float]) -> float:
    m = expected(dist)
    return sum(p * (s - m) ** 2 for s, p in zip(STARS, dist)) ** 0.5


def wasserstein(a: Sequence[float], b: Sequence[float]) -> float:
    """Earth mover's distance between two distributions over 1..5 stars (in stars)."""
    ca = cb = total = 0.0
    for pa, pb in zip(a[:-1], b[:-1]):
        ca += pa
        cb += pb
        total += abs(ca - cb)
    return total


def negative_share(dist: Sequence[float]) -> float:
    return dist[0] + dist[1]


def pairwise_accuracy(records: List[Dict], key: str, min_gap: float = 0.3) -> float:
    correct = total = 0
    for a, b in combinations(records, 2):
        if a["topic"] != b["topic"]:
            continue
        gap = expected(a["truth"]) - expected(b["truth"])
        if abs(gap) < min_gap:
            continue
        pred_gap = expected(a[key]) - expected(b[key])
        total += 1
        correct += (pred_gap > 0) == (gap > 0) if pred_gap != 0 else 0.5
    return correct / total if total else float("nan")


def metrics(records: List[Dict], key: str) -> Dict[str, float]:
    truth_means = [expected(r["truth"]) for r in records]
    pred_means = [expected(r[key]) for r in records]
    rho = spearmanr(pred_means, truth_means).statistic if len(set(pred_means)) > 1 else float("nan")
    return {
        "mean_abs_error": mean(abs(p - t) for p, t in zip(pred_means, truth_means)),
        "wasserstein": mean(wasserstein(r[key], r["truth"]) for r in records),
        "negative_share_error": mean(abs(negative_share(r[key]) - negative_share(r["truth"])) for r in records),
        "pairwise_accuracy": pairwise_accuracy(records, key),
        "spearman": rho,
        "bias": mean(p - t for p, t in zip(pred_means, truth_means)),
        "spread_ratio": mean(spread(r[key]) for r in records) / max(mean(spread(r["truth"]) for r in records), 1e-9),
    }


def bootstrap(records: List[Dict], fn: Callable[[List[Dict]], Dict[str, float]], n: int = 1000,
              seed: int = 0) -> Dict[str, List[float]]:
    """95% CI per metric by resampling products with replacement."""
    rng = random.Random(seed)
    draws: Dict[str, List[float]] = {}
    for _ in range(n):
        sample = [rng.choice(records) for _ in records]
        for k, v in fn(sample).items():
            if v == v:  # skip NaN (e.g. no qualifying pairs in a resample)
                draws.setdefault(k, []).append(v)
    ci = {}
    for k, vals in draws.items():
        vals.sort()
        ci[k] = [vals[int(0.025 * (len(vals) - 1))], vals[int(0.975 * (len(vals) - 1))]]
    return ci


def summarize(records: List[Dict], n_boot: int = 1000) -> Dict[str, Dict]:
    out = {}
    keys = [k for k in PREDICTORS if k in records[0]]
    for key in keys:
        out[key] = {"metrics": metrics(records, key),
                    "ci95": bootstrap(records, lambda rs, k=key: metrics(rs, k), n=n_boot)}
    out["by_topic"] = {
        topic: {key: metrics([r for r in records if r["topic"] == topic], key)["mean_abs_error"] for key in keys}
        for topic in sorted({r["topic"] for r in records})
    }
    out["n_products"] = len(records)
    out["predictors"] = keys
    return out
