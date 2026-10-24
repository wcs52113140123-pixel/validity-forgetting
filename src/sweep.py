"""Multi-seed / multi-dataset sweep driver.

Each (dataset, backbone, method, seed) run is appended to a long-format CSV, so the
sweep is resumable: an interrupted run is simply repeated, and finished cells are
skipped on restart. Aggregation with confidence intervals happens in `aggregate()`.

`Joint` is excluded by default because it is an upper bound trained to convergence
(300 epochs, cumulative) and costs ~1000 s per run — two orders of magnitude more
than any continual method here. It is run separately on fewer seeds.
"""
from __future__ import annotations
import os, sys, glob, json, time, itertools
os.environ.setdefault("KMP_AFFINITY", "disabled")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experiment as EX
import reliability as RL

FAST = ["Seq-FT", "EWC", "LwF", "Replay", "iCaRL", "NCM", "RanPAC"]
OUTFILE = "sweep_records.csv"


def done_cells(path=OUTFILE):
    if not os.path.exists(path):
        return set()
    d = pd.read_csv(path, usecols=["dataset", "backbone", "method", "seed", "n_tasks", "score"])
    return set(map(tuple, d.drop_duplicates().values))


def run_sweep(datasets, backbones, methods=FAST, seeds=(0, 1, 2, 3, 4), n_tasks=10,
              score="lac", mem_per_class=20, cal_per_class=5, path=OUTFILE, verbose=True):
    have = done_cells(path)
    for ds, bb in itertools.product(datasets, backbones):
        f = f"features/{ds}__{bb}.npz"
        if not os.path.exists(f):
            if verbose:
                print(f"  [skip] {f} 不存在", flush=True)
            continue
        z = np.load(f)
        Xtr, ytr, Xte, yte = z["Xtr"], z["ytr"], z["Xte"], z["yte"]
        ncl = int(len(np.unique(ytr)))
        nt = n_tasks if ncl % n_tasks == 0 else max(2, min(n_tasks, ncl // 2))
        for mk, sd in itertools.product(methods, seeds):
            key = (ds, bb, mk, sd, nt, score)
            if key in have:
                continue
            t0 = time.time()
            recs = EX.run_one(Xtr, ytr, Xte, yte, mk, n_tasks=nt, seed=sd, score=score,
                              mem_per_class=mem_per_class, cal_per_class=cal_per_class)
            df = pd.DataFrame(recs)
            df["dataset"] = ds; df["backbone"] = bb; df["n_classes"] = ncl
            df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
            have.add(key)
            if verbose:
                print(f"  {ds:12s} {bb:14s} {mk:8s} s{sd} nt={nt} "
                      f"{len(df):>4}行 {time.time()-t0:>5.0f}s", flush=True)


def aggregate(path=OUTFILE, alpha=0.10, protocol="A_current"):
    """Per-(dataset, backbone, method) scalars with mean +/- 95% CI over seeds."""
    D = pd.read_csv(path)
    rows = []
    for (ds, bb, mk, sd, nt), g in D.groupby(["dataset", "backbone", "method", "seed", "n_tasks"]):
        s = EX.summarise(g, int(nt), alpha=alpha)
        s = s[s.protocol == protocol]
        if len(s) == 0:
            continue
        r = s.iloc[0].to_dict()
        r.update(dataset=ds, backbone=bb, method=mk, seed=sd, n_tasks=nt)
        rows.append(r)
    R = pd.DataFrame(rows)
    met = ["ACC", "FGT", "FinalCov", "CovGap", "WorstTaskGap", "ValidTaskFrac",
           "CovForget", "SizeInflation", "FinalSize", "FinalECE"]
    agg = R.groupby(["dataset", "backbone", "method"])[met].agg(["mean", "std", "count"])
    out = pd.DataFrame(index=agg.index)
    for m in met:
        mu, sd_, n = agg[(m, "mean")], agg[(m, "std")], agg[(m, "count")]
        out[m] = mu.round(4)
        out[m + "_ci95"] = (1.96 * sd_ / np.sqrt(n)).round(4)
    out["n_seeds"] = agg[(met[0], "count")]
    return R, out.reset_index()


if __name__ == "__main__":
    ds = sys.argv[1].split(",") if len(sys.argv) > 1 else ["cifar100"]
    bb = sys.argv[2].split(",") if len(sys.argv) > 2 else ["vit_b16_in21k", "dinov2_base"]
    sc = sys.argv[3] if len(sys.argv) > 3 else "lac"
    t0 = time.time()
    run_sweep(ds, bb, score=sc)
    print(f"\nsweep 完成 {time.time()-t0:.0f}s")
