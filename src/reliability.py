"""Reliability measurement instrument for continual learning.

Standard CL evaluation reports only accuracy-based quantities (ACC / BWT / forgetting).
This module adds the *validity* axis: after each incremental step we ask whether the
model's uncertainty is still trustworthy on earlier tasks, using split conformal
prediction (distribution-free, finite-sample) plus calibration and selective-risk views.

Conventions
-----------
probs : (n, C) row-stochastic array of predictive probabilities.
labels: (n,) int array of ground-truth class indices.
alpha : miscoverage level; nominal coverage is 1 - alpha.

Conformal scores follow the two standard classification choices:
  * "lac"  — Least Ambiguous set-valued Classifier (a.k.a. THR): s = 1 - p_y.
             Smallest average set size; coverage is marginal, not class-conditional.
  * "aps"  — Adaptive Prediction Sets: s = cumulative mass of classes ranked above y,
             plus (optionally randomized) part of p_y. Better conditional coverage,
             larger sets.

References for the estimators implemented here: Vovk et al. (2005) split conformal;
Sadinle et al. (2019) LAC/THR; Romano et al. (2020) APS; Geifman & El-Yaniv (2017)
risk-coverage / AURC; Naeini et al. (2015) ECE.
"""
from __future__ import annotations
import numpy as np

EPS = 1e-12


# ----------------------------------------------------------------------------- scores
def lac_score(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Non-conformity of the TRUE label: s = 1 - p_y.  Lower = better conformity."""
    return 1.0 - probs[np.arange(len(labels)), labels]


def aps_score(probs: np.ndarray, labels: np.ndarray, rng: np.random.Generator | None = None,
              randomized: bool = True) -> np.ndarray:
    """APS non-conformity of the true label.

    s_i = sum of probabilities of classes ranked strictly above y_i, plus u * p_{y_i}
    with u ~ U(0,1) when randomized (Romano et al. 2020), else plus p_{y_i}.
    """
    order = np.argsort(-probs, axis=1)                       # descending
    sorted_p = np.take_along_axis(probs, order, axis=1)
    csum = np.cumsum(sorted_p, axis=1)
    rank = np.argmax(order == labels[:, None], axis=1)       # position of true label
    n = len(labels)
    cum_incl = csum[np.arange(n), rank]                      # includes p_y
    p_y = sorted_p[np.arange(n), rank]
    if randomized:
        rng = rng or np.random.default_rng(0)
        u = rng.uniform(size=n)
        return cum_incl - p_y + u * p_y
    return cum_incl


def _aps_set_scores(probs: np.ndarray, rng: np.random.Generator | None = None,
                    randomized: bool = True) -> np.ndarray:
    """Per-class APS scores for every candidate label (used at prediction time)."""
    order = np.argsort(-probs, axis=1)
    sorted_p = np.take_along_axis(probs, order, axis=1)
    csum = np.cumsum(sorted_p, axis=1)
    if randomized:
        rng = rng or np.random.default_rng(0)
        u = rng.uniform(size=(probs.shape[0], 1))
        sc_sorted = csum - sorted_p + u * sorted_p
    else:
        sc_sorted = csum
    out = np.empty_like(sc_sorted)
    np.put_along_axis(out, order, sc_sorted, axis=1)
    return out


# ------------------------------------------------------------------- split conformal
def conformal_quantile(cal_scores: np.ndarray, alpha: float) -> float:
    """Finite-sample corrected (1-alpha) quantile: level ceil((n+1)(1-alpha))/n.

    With this correction split conformal guarantees marginal coverage >= 1-alpha
    for exchangeable data. Returns +inf when n is too small for the level to exist
    (i.e. the only valid set is the full label space).
    """
    n = len(cal_scores)
    if n == 0:
        return np.inf
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return np.inf
    return float(np.sort(cal_scores)[k - 1])


def predict_sets(probs: np.ndarray, qhat: float, method: str = "lac",
                 rng: np.random.Generator | None = None, randomized: bool = True) -> np.ndarray:
    """Boolean (n, C) membership matrix {y : score(x, y) <= qhat}."""
    if method == "lac":
        return (1.0 - probs) <= qhat
    if method == "aps":
        return _aps_set_scores(probs, rng=rng, randomized=randomized) <= qhat
    raise ValueError(f"unknown method {method!r}")


def score_fn(method: str):
    return {"lac": lac_score, "aps": aps_score}[method]


# ------------------------------------------------------------------- validity metrics
def coverage(sets: np.ndarray, labels: np.ndarray) -> float:
    """Empirical marginal coverage: fraction of points whose set contains the truth."""
    return float(sets[np.arange(len(labels)), labels].mean())


def avg_set_size(sets: np.ndarray) -> float:
    return float(sets.sum(axis=1).mean())


def size_stratified_coverage(sets: np.ndarray, labels: np.ndarray,
                             bins=(1, 2, 3, 6, 11, 1 << 30)) -> dict:
    """Coverage within strata of set size (SSCV diagnostic, Angelopoulos et al. 2021)."""
    sz = sets.sum(axis=1)
    hit = sets[np.arange(len(labels)), labels]
    out, lo = {}, 0
    for hi in bins:
        m = (sz >= lo) & (sz < hi)
        if m.sum() > 0:
            out[f"[{lo},{hi if hi < (1 << 30) else 'inf'})"] = float(hit[m].mean())
        lo = hi
    return out


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error on top-1 confidence (equal-width bins)."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    acc = (pred == labels).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        total += (m.mean()) * abs(acc[m].mean() - conf[m].mean())
    return float(total)


def selective_risk_curve(probs: np.ndarray, labels: np.ndarray):
    """Risk-coverage curve using max-prob as the confidence score.

    Returns (coverages, risks) where risks[k] is the error rate on the k most
    confident points. Ties are broken by the sort order.
    """
    conf = probs.max(axis=1)
    err = (probs.argmax(axis=1) != labels).astype(float)
    order = np.argsort(-conf)
    err_sorted = err[order]
    n = len(labels)
    cum_err = np.cumsum(err_sorted)
    ks = np.arange(1, n + 1)
    return ks / n, cum_err / ks


def aurc(probs: np.ndarray, labels: np.ndarray) -> float:
    """Area under the risk-coverage curve. Lower = better ordering of confidence."""
    cov, risk = selective_risk_curve(probs, labels)
    return float(np.trapezoid(risk, cov) / (cov[-1] - cov[0])) if len(cov) > 1 else float(risk[0])


def selective_risk_at(probs: np.ndarray, labels: np.ndarray, target_coverage: float) -> float:
    """Error rate when abstaining on all but the top `target_coverage` fraction."""
    cov, risk = selective_risk_curve(probs, labels)
    k = int(np.searchsorted(cov, target_coverage, side="left"))
    return float(risk[min(k, len(risk) - 1)])


# ------------------------------------------------- continual-learning aggregate views
def acc_matrix_metrics(R: np.ndarray) -> dict:
    """Standard CL scalars from the accuracy matrix R[i, j] = acc on task j after step i.

    ACC  : mean over tasks of the final row.
    BWT  : mean_{j<T} (R[T-1, j] - R[j, j])  (Lopez-Paz & Ranzato 2017; negative = forgetting)
    FGT  : mean_{j<T} max_i<=T-1 (R[i, j]) - R[T-1, j]  (Chaudhry et al. 2018)
    """
    T = R.shape[0]
    final = R[T - 1]
    acc = float(np.nanmean(final))
    if T < 2:
        return {"ACC": acc, "BWT": 0.0, "FGT": 0.0}
    bwt = float(np.nanmean([R[T - 1, j] - R[j, j] for j in range(T - 1)]))
    fgt = float(np.nanmean([np.nanmax(R[: T, j]) - R[T - 1, j] for j in range(T - 1)]))
    return {"ACC": acc, "BWT": bwt, "FGT": fgt}


def forgetting_of(M: np.ndarray, higher_is_better: bool = True) -> float:
    """Generic forgetting for ANY per-task quantity matrix M[i, j] (same layout as R).

    Defined as mean over earlier tasks of (best value seen) - (final value), signed so
    that a POSITIVE number always means degradation. This is what turns coverage,
    set size and selective risk into 'forgetting' quantities comparable to FGT.
    """
    T = M.shape[0]
    if T < 2:
        return 0.0
    vals = []
    for j in range(T - 1):
        col = M[: T, j]
        if higher_is_better:
            vals.append(np.nanmax(col) - M[T - 1, j])
        else:
            vals.append(M[T - 1, j] - np.nanmin(col))
    return float(np.nanmean(vals))


def coverage_gap(cov_matrix: np.ndarray, alpha: float) -> dict:
    """Validity accounting against the nominal level on the FINAL model.

    CovGap      : mean_j (1-alpha) - cov[T-1, j], clipped at 0 (undercoverage only).
    WorstTaskGap: the largest single-task undercoverage.
    ValidTasks  : fraction of tasks still meeting the nominal level.
    """
    final = cov_matrix[-1]
    nominal = 1.0 - alpha
    gaps = np.clip(nominal - final, 0.0, None)
    return {"CovGap": float(np.nanmean(gaps)),
            "WorstTaskGap": float(np.nanmax(gaps)),
            "ValidTaskFrac": float(np.nanmean(final >= nominal - EPS))}


# --------------------------------------------------------- class-conditional (Mondrian)

def mondrian_quantiles(cal_scores, cal_labels, alpha, n_classes, fallback=None):
    """Per-class conformal thresholds — the Mondrian construction with class taxonomy.

    Returns an array q[c]. A class with fewer than ceil(1/alpha)-1 reserved points
    cannot attain the level from its own sample; for those we fall back to the pooled
    quantile and REPORT how many, because silently using a pooled threshold for some
    classes would void the per-class guarantee without the reader seeing it.

    This is the construction the reserved budget makes available: k slots per class,
    held out from training, give k exchangeable scores per class.
    """
    import numpy as _np
    kmin = int(_np.ceil(1.0 / alpha)) - 1
    pooled = conformal_quantile(cal_scores, alpha) if fallback is None else fallback
    q = _np.full(n_classes, pooled, dtype=float)
    n_fallback = 0
    for c in range(n_classes):
        m = cal_labels == c
        if m.sum() >= kmin:
            q[c] = conformal_quantile(cal_scores[m], alpha)
        elif m.sum() > 0:
            n_fallback += 1
        else:
            n_fallback += 1
    return q, n_fallback, kmin


def predict_sets_mondrian(probs, q_per_class, method="lac", rng=None):
    """Prediction sets under per-class thresholds: y is in the set iff s(x,y) <= q_y."""
    import numpy as _np
    if method == "lac":
        S = 1.0 - probs                      # score of label y at row i
        return S <= q_per_class[None, :]
    S = _aps_set_scores(probs, rng=rng)
    return S <= q_per_class[None, :]


def class_conditional_coverage(sets, labels, n_classes):
    """Coverage within each class. Returns (per_class array, worst, fraction meeting)."""
    import numpy as _np
    out = _np.full(n_classes, _np.nan)
    for c in range(n_classes):
        m = labels == c
        if m.sum():
            out[c] = float(sets[m, c].mean())
    ok = ~_np.isnan(out)
    return out, float(_np.nanmin(out)) if ok.any() else _np.nan, out


def coverage_under_mixture(sets, labels, weights):
    """Marginal coverage when classes are re-weighted to a deployment mixture.

    weights[c] is the target probability of class c. Computed as a weighted average of
    the per-class coverages, which is what the marginal coverage would be if the test
    set were drawn from that mixture — no resampling, so no Monte-Carlo noise.
    """
    import numpy as _np
    n_classes = len(weights)
    per, _, _ = class_conditional_coverage(sets, labels, n_classes)
    w = _np.asarray(weights, dtype=float)
    ok = ~_np.isnan(per)
    w = w * ok
    if w.sum() == 0:
        return _np.nan
    w = w / w.sum()
    return float(_np.nansum(w * _np.where(ok, per, 0.0)))
