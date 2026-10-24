"""Per-task null rate of a valid split-conformal predictor (Section 3.3).

For calibration size n and per-task test size m, the conditional coverage of split conformal
at level alpha is Beta(k, n+1-k) with k = ceil((n+1)(1-alpha)); the realised task coverage is
Binomial(m, C)/m. We average P(task coverage >= 1-alpha) over the ten tasks and five class orders.
"""
import numpy as np


def p_meet(n, m, alpha=0.10, sims=40000, rng=None):
    rng = rng or np.random.default_rng(0)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    C = rng.beta(k, n + 1 - k, sims)
    return float((rng.binomial(m, C) / m >= 1 - alpha - 1e-12).mean())
