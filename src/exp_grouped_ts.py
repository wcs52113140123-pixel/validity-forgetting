"""Grouped (task-wise) temperature scaling as a fix for validity forgetting.

Section 3.7 argues that a single global temperature cannot close the coverage gap,
because the gap follows the ranking of confidences and a global monotone map leaves the
ranking unchanged. Task-aware / multi-group temperature scaling (DATS, MT-CC) is not a
global map: each task's classes get their own temperature, so the ordering ACROSS task
groups can change. This script tests whether that closes the gap.

Every run trains exactly as the main sweep (same splits, same memory and reservation).
At the final step, logits are rescaled per task group,  z'_c = z_c / T_{g(c)},  with the
group temperatures fitted jointly by NLL on one of these calibration sources:

  global          one temperature on the current task (= the main sweep; sanity check)
  group_oracle    per-task temperatures on held-out data of EVERY seen task
                  (the most favourable data, not available to a continual learner)
  group_reserved  per-task temperatures on the reserved calibration slots (Replay, iCaRL;
                  class-balanced, 5 per seen class, never trained on)
  group_replayed  per-task temperatures on the replayed exemplars (what a memory-based
                  learner could do without reserving; the optimiser has fitted them)

The conformal threshold is then set the practitioner's way (protocol A: current task).
For global and group_oracle the pooled threshold (protocol B) is also reported, and for
global and group_reserved the RBCC threshold (protocol C, reserved slots).

    python exp_grouped_ts.py <src_dir>        (features/ must hold the .npz caches)
"""
import os, sys, time, itertools
os.environ.setdefault("KMP_AFFINITY", "disabled")
sys.path.append(sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.special import log_softmax, softmax
from concurrent.futures import ProcessPoolExecutor
import experiment as EX
import cl_methods as CM
import reliability as RL

ALPHA = 0.10
OUT = os.environ.get("GTS_OUT", "grouped_ts_records.csv")


def fit_group_temperature(z, y, group_of_class, n_groups, bounds=(0.05, 20.0)):
    """Joint NLL fit of one temperature per class group (log-parameterised, L-BFGS-B)."""
    gi = np.where(group_of_class >= 0, group_of_class, 0)
    N = len(y)

    def f(logT):
        zz = z / np.exp(logT)[gi]
        ls = log_softmax(zz, axis=1)
        nll = -ls[np.arange(N), y].mean()
        p = np.exp(ls); p[np.arange(N), y] -= 1.0
        col = (p * (-zz)).sum(0) / N                 # d nll / d logT_{g(c)}, per class
        return nll, np.bincount(gi, weights=col, minlength=n_groups)
    lo, hi = np.log(bounds[0]), np.log(bounds[1])
    r = minimize(f, np.zeros(n_groups), jac=True, method="L-BFGS-B", bounds=[(lo, hi)] * n_groups)
    return np.exp(r.x)


def apply_group(z, Tg, group_of_class):
    gi = np.where(group_of_class >= 0, group_of_class, 0)
    return softmax(z / Tg[gi], axis=1)


def one(args):
    ds, bb, mk, sd = args
    z = np.load(f"features/{ds}__{bb}.npz"); Xtr, ytr, Xte, yte = z["Xtr"], z["ytr"], z["Xte"], z["yte"]
    ncl = int(len(np.unique(ytr))); nt = 10 if ncl % 10 == 0 else max(2, min(10, ncl // 2))
    t0 = time.time()
    C = int(max(ytr.max(), yte.max())) + 1
    tasks = CM.class_incremental_split(ytr, nt, seed=sd)
    kw = ({"mem_per_class": 20, "cal_per_class": 5, "hold_out_cal": True} if mk in ("Replay", "iCaRL") else {})
    m = CM.METHODS[mk](Xtr.shape[1], C, seed=sd, **kw)
    fit_idx, cal_idx, test_idx = {}, {}, {}
    for j, tc in enumerate(tasks):
        tr_j = np.where(np.isin(ytr, tc))[0]
        a, b = EX.stratified_split(ytr[tr_j], EX.CAL_FRAC, seed=1000 + sd * 97 + j)
        fit_idx[j], cal_idx[j] = tr_j[a], tr_j[b]
        test_idx[j] = np.where(np.isin(yte, tc))[0]
    for i, tc in enumerate(tasks):
        m.observe(Xtr[fit_idx[i]], ytr[fit_idx[i]], tc)
    last = len(tasks) - 1
    goc = np.full(C, -1)
    for j, tc in enumerate(tasks):
        goc[np.asarray(tc)] = j
    ng = len(tasks)

    def L(X):
        zz = m.logits(X)
        return np.where(np.isfinite(zz), zz, -1e9)
    sfn = RL.score_fn("lac")
    cur = cal_idx[last]
    pooled = np.concatenate([cal_idx[j] for j in range(ng)])

    variants = {}
    T0 = CM.fit_temperature(L(Xtr[cur]), ytr[cur])
    variants["global"] = lambda X: CM.apply_temperature(L(X), T0)
    Tor = fit_group_temperature(L(Xtr[pooled]), ytr[pooled], goc, ng)
    variants["group_oracle"] = lambda X: apply_group(L(X), Tor, goc)
    Ts = {"group_oracle": Tor}
    if hasattr(m, "cal_buffer"):
        cX, cy = m.cal_buffer()
        # class-balanced sources: the memory holds the same number of points for every seen
        # class, including the current task's (mixing in the current task's full held-out
        # split would weight the joint fit towards the newest classes)
        Tres = fit_group_temperature(L(cX), cy, goc, ng)
        variants["group_reserved"] = lambda X: apply_group(L(X), Tres, goc)
        bX, by = m.buffer()
        Trep = fit_group_temperature(L(bX), by, goc, ng)
        variants["group_replayed"] = lambda X: apply_group(L(X), Trep, goc)
        Ts.update(group_reserved=Tres, group_replayed=Trep)
        RBCC_CAL = (cX, cy)

    recs = []
    for vname, proba in variants.items():
        thr = {"A_current": RL.conformal_quantile(sfn(proba(Xtr[cur]), ytr[cur]), ALPHA)}
        if vname in ("global", "group_oracle"):
            thr["B_pooled"] = RL.conformal_quantile(sfn(proba(Xtr[pooled]), ytr[pooled]), ALPHA)
        if vname in ("global", "group_reserved") and hasattr(m, "cal_buffer"):
            thr["C_rbcc"] = RL.conformal_quantile(sfn(proba(RBCC_CAL[0]), RBCC_CAL[1]), ALPHA)
        for prot, q in thr.items():
            for j in range(ng):
                r = EX.eval_block(proba(Xte[test_idx[j]]), yte[test_idx[j]], q, ALPHA, "lac")
                Tj = (T0 if vname == "global" else float(Ts[vname][j]))
                recs.append(dict(dataset=ds, backbone=bb, method=mk, seed=sd, variant=vname, protocol=prot,
                                 step=last, task=j, n_tasks=ng, T=Tj, **r))
    return pd.DataFrame(recs), f"{ds} {bb} {mk} s{sd} {time.time() - t0:.0f}s"


if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "2"
    cells = list(itertools.product(["cifar100", "cub200", "imagenet_r", "medmix26"], ["vit_b16_in21k", "dinov2_base"],
                                   ["Seq-FT", "EWC", "LwF", "Replay", "iCaRL", "NCM", "RanPAC"], range(5)))
    if os.environ.get("GTS_CELLS"):
        sel = os.environ["GTS_CELLS"].split(",")
        cells = [c for c in cells if f"{c[0]}|{c[1]}|{c[2]}|{c[3]}" in sel]
    out = []
    with ProcessPoolExecutor(max_workers=int(os.environ.get("GTS_WORKERS", 60))) as ex:
        for df, msg in ex.map(one, cells):
            out.append(df); print(msg, flush=True)
    pd.concat(out).to_csv(OUT, index=False); print("done", len(out), "cells")
