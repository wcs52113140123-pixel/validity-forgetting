"""Phase 3-4: run the continual-learning x reliability experiment matrix.

The measurement question
------------------------
Standard CL evaluation asks "how much accuracy survived?". We additionally ask
"is the model's uncertainty still VALID on earlier tasks?", where validity means
the conformal prediction set still covers the truth at the nominal rate.

Three calibration protocols are compared at every incremental step:

  A  "current practice"  -- calibrate on the CURRENT task's calibration split only.
                            This is what a practitioner who conformalises a deployed
                            continual learner implicitly does, and it is the protocol
                            whose failure this paper documents.
  B  "oracle / pooled"   -- calibrate on a pooled split from ALL tasks seen so far.
                            Requires retaining old data, so it violates the CL memory
                            constraint; included as an upper bound on recoverability.
  C  "RBCC (ours)"       -- calibrate on the exemplar memory the replay-based method
                            already stores. No additional memory beyond the buffer.

For each protocol we report, per (step i, task j): accuracy, coverage, mean set size,
ECE and selective risk. The aggregate views (Coverage Forgetting, Set-Size Inflation,
Selective-Risk Forgetting, CovGap) are computed from those matrices by `reliability`.

Every method receives the same post-hoc temperature-scaling opportunity, fitted on the
same split used for its conformal quantile, so prototype/ridge methods are not
penalised merely for emitting scores rather than probabilities.
"""
from __future__ import annotations
import os, json, itertools, time
os.environ.setdefault("KMP_AFFINITY", "disabled")
os.environ.setdefault("OMP_NUM_THREADS", "32")

import numpy as np
import pandas as pd

import cl_methods as CM
import reliability as RL

ALPHA_DEFAULT = 0.10
CAL_FRAC = 0.20


# ------------------------------------------------------------------------------- utils
def stratified_split(y, frac, seed):
    """Class-stratified index split; returns (major_idx, minor_idx) with |minor| ~ frac."""
    rng = np.random.default_rng(seed)
    a, b = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        k = int(round(frac * len(idx)))
        k = min(max(k, 1), len(idx) - 1) if len(idx) > 1 else 0
        b += idx[:k].tolist(); a += idx[k:].tolist()
    return np.array(sorted(a)), np.array(sorted(b))


def eval_block(P, y, qhat, alpha, method_name):
    """All per-(step, task) scalars for one probability block."""
    sets = RL.predict_sets(P, qhat, method=method_name)
    return {
        "acc": float((P.argmax(1) == y).mean()),
        "cov": RL.coverage(sets, y),
        "size": RL.avg_set_size(sets),
        "ece": RL.ece(P, y),
        "srisk80": RL.selective_risk_at(P, y, 0.80),
        "aurc": RL.aurc(P, y),
    }


# --------------------------------------------------------------------------- one run
def run_one(Xtr, ytr, Xte, yte, method_key, n_tasks, seed,
            alpha=ALPHA_DEFAULT, score="lac", mem_per_class=20, cal_per_class=0,
            hold_out_cal=False,
            temp_scale=True):
    """One (method, n_tasks, seed) run; returns a long-format list of records.

    `mem_per_class` is the TOTAL per-class exemplar budget. `cal_per_class` of those
    slots are reserved for calibration (never replayed), so raising it does not raise
    total memory — it trades replay capacity for calibration validity.
    """
    C = int(max(ytr.max(), yte.max())) + 1
    tasks = CM.class_incremental_split(ytr, n_tasks, seed=seed)
    cls = CM.METHODS[method_key]
    # The RBCC protocol calibrates on the method's reserved slots, which cl_methods
    # fills from the data passed to observe. With hold_out_cal the slots are drawn
    # before the training pass and withheld from it, as the guarantee requires; the
    # measured effect of the two choices is in holdout_effect.csv.
    kw = ({"mem_per_class": mem_per_class, "cal_per_class": cal_per_class,
           "hold_out_cal": hold_out_cal}
          if method_key in ("Replay", "iCaRL") else {})
    m = cls(Xtr.shape[1], C, seed=seed, **kw)

    # per-task fit / calibration splits, fixed before training
    fit_idx, cal_idx, test_idx = {}, {}, {}
    for j, tc in enumerate(tasks):
        tr_j = np.where(np.isin(ytr, tc))[0]
        a, b = stratified_split(ytr[tr_j], CAL_FRAC, seed=1000 + seed * 97 + j)
        fit_idx[j], cal_idx[j] = tr_j[a], tr_j[b]
        test_idx[j] = np.where(np.isin(yte, tc))[0]

    recs = []
    for i, tc in enumerate(tasks):
        m.observe(Xtr[fit_idx[i]], ytr[fit_idx[i]], tc)

        # ---- temperature: fitted once per step on the current task's calibration split
        T = 1.0
        if temp_scale:
            z = m.logits(Xtr[cal_idx[i]])
            z = np.where(np.isfinite(z), z, -1e9)
            T = CM.fit_temperature(z, ytr[cal_idx[i]])

        def proba(X):
            z = m.logits(X)
            return CM.apply_temperature(z, T)

        sfn = RL.score_fn(score)
        # ---- protocol A: current task only
        qA = RL.conformal_quantile(sfn(proba(Xtr[cal_idx[i]]), ytr[cal_idx[i]]), alpha)
        # ---- protocol B: pooled over all seen tasks (oracle; needs retained data)
        pooled = np.concatenate([cal_idx[j] for j in range(i + 1)])
        qB = RL.conformal_quantile(sfn(proba(Xtr[pooled]), ytr[pooled]), alpha)
        # ---- protocol C: RBCC -- calibration-RESERVED slice of the same memory budget
        qC = np.nan
        cX, cy = (m.cal_buffer() if hasattr(m, "cal_buffer") else (None, None))
        if cX is not None and len(cy) >= int(np.ceil(1 / alpha)) - 1:
            qC = RL.conformal_quantile(sfn(proba(cX), cy), alpha)
        # ---- protocol C': the naive variant that calibrates on REPLAYED exemplars,
        #      included to show why the reservation is necessary (they are not
        #      exchangeable with test data, having been fitted during training).
        qCn = np.nan
        bX, by = (m.buffer() if hasattr(m, "buffer") else (None, None))
        if bX is not None and len(by) >= int(np.ceil(1 / alpha)) - 1:
            qCn = RL.conformal_quantile(sfn(proba(bX), by), alpha)

        for j in range(i + 1):
            Pj = proba(Xte[test_idx[j]]); yj = yte[test_idx[j]]
            ncal = {"A_current": len(cal_idx[i]), "B_pooled": len(pooled),
                    "C_rbcc": (len(cy) if cX is not None else 0),
                    "Cn_replayed": (len(by) if bX is not None else 0)}
            for prot, q in (("A_current", qA), ("B_pooled", qB),
                            ("C_rbcc", qC), ("Cn_replayed", qCn)):
                if not np.isfinite(q):
                    continue
                r = eval_block(Pj, yj, q, alpha, score)
                recs.append(dict(method=method_key, n_tasks=n_tasks, seed=seed, alpha=alpha,
                                 score=score, protocol=prot, step=i, task=j, T=T,
                                 n_cal=ncal[prot], **r))
    return recs


def matrices_from(df, metric, protocol, n_tasks):
    """Rebuild the (step x task) matrix for one metric/protocol."""
    Mx = np.full((n_tasks, n_tasks), np.nan)
    sub = df[df.protocol == protocol]
    for _, r in sub.iterrows():
        Mx[int(r["step"]), int(r["task"])] = r[metric]
    return Mx


def summarise(df, n_tasks, alpha=ALPHA_DEFAULT):
    """Aggregate scalars per (method, protocol)."""
    out = []
    for (meth, prot), g in df.groupby(["method", "protocol"]):
        A = matrices_from(g, "acc", prot, n_tasks)
        Cv = matrices_from(g, "cov", prot, n_tasks)
        Sz = matrices_from(g, "size", prot, n_tasks)
        Sr = matrices_from(g, "srisk80", prot, n_tasks)
        Ec = matrices_from(g, "ece", prot, n_tasks)
        acc = RL.acc_matrix_metrics(A)
        gap = RL.coverage_gap(Cv, alpha)
        out.append(dict(method=meth, protocol=prot,
                        ACC=acc["ACC"], BWT=acc["BWT"], FGT=acc["FGT"],
                        FinalCov=float(np.nanmean(Cv[-1])),
                        CovForget=RL.forgetting_of(Cv, higher_is_better=True),
                        SizeInflation=RL.forgetting_of(Sz, higher_is_better=False),
                        SRiskForget=RL.forgetting_of(Sr, higher_is_better=False),
                        ECEForget=RL.forgetting_of(Ec, higher_is_better=False),
                        FinalECE=float(np.nanmean(Ec[-1])),
                        FinalSize=float(np.nanmean(Sz[-1])),
                        **gap))
    return pd.DataFrame(out)
