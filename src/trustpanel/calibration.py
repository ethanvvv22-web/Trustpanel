"""Calibrate a panel's predicted mean rating against real ratings on separate products.

LLM panels can rank products sensibly yet be off in level (too harsh, too unanimous).
A linear map from predicted to real mean star rating is fitted on calibration products
(never used for personas or testing). The calibrated distribution is the topic's prior
rating distribution, tilted to have the calibrated mean: the panel supplies the signal,
the prior supplies the shape that real reviews have.
"""

import math
from typing import Dict, List, Sequence, Tuple

from .validation import STARS, expected


def fit_linear(pred: Sequence[float], real: Sequence[float]) -> Tuple[float, float]:
    """Least-squares intercept and slope of real ~ pred."""
    n = len(pred)
    mp, mr = sum(pred) / n, sum(real) / n
    var = sum((p - mp) ** 2 for p in pred)
    slope = sum((p - mp) * (r - mr) for p, r in zip(pred, real)) / var if var > 1e-12 else 0.0
    return mr - slope * mp, slope


def tilt(prior: Sequence[float], target_mean: float) -> List[float]:
    """The distribution closest to `prior` (exponential tilting) whose mean is `target_mean`."""
    target = min(max(target_mean, 1.01), 4.99)

    def tilted(lam: float) -> List[float]:
        w = [p * math.exp(lam * s) for p, s in zip(prior, STARS)]
        z = sum(w)
        return [x / z for x in w]

    lo, hi = -20.0, 20.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if expected(tilted(mid)) < target:
            lo = mid
        else:
            hi = mid
    return tilted((lo + hi) / 2)


def calibrate(records: List[Dict], calibration_records: List[Dict], key: str) -> Dict[str, float]:
    """Add `<key prefix>_calibrated` distributions to `records`; return the fitted map."""
    intercept, slope = fit_linear([expected(r[key]) for r in calibration_records],
                                  [expected(r["truth"]) for r in calibration_records])
    out_key = key.replace("_panel", "").replace("_llm", "") + "_calibrated"
    for r in records:
        r[out_key] = tilt(r["topic_prior"], intercept + slope * expected(r[key]))
    return {"intercept": intercept, "slope": slope}
