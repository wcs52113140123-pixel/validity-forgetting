"""Does withholding the reserved slots change the paper's headline fix result?

Section 3.10 reports that calibrating on reserved replay slots (protocol C) cuts the
coverage gap relative to calibrating on the current task alone (protocol A). The
reserved slots were filled from the data the model had just trained on, which biases
their conformal scores low and makes protocol C undercover. The bias therefore works
AGAINST the result the section reports, but its size has to be measured rather than
assumed.

This runs the identical configuration with the slots inside and outside the training
pass, for the two replay-based methods, five class orderings each.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
import experiment as EX  # noqa: E402

SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2,3,4").split(",")]
METHODS = os.environ.get("METHODS", "Replay,iCaRL").split(",")
NTASK = int(os.environ.get("NTASK", "10"))
ALPHA = 0.10


def main(feat):
    z = np.load(feat, allow_pickle=True)
    X, y, Xt, yt = z["Xtr"], z["ytr"], z["Xte"], z["yte"]
    rows = []
    for mk in METHODS:
        for ho in (False, True):
            for s in SEEDS:
                r = EX.run_one(X, y, Xt, yt, mk, NTASK, seed=s, alpha=ALPHA,
                               mem_per_class=20, cal_per_class=5, hold_out_cal=ho)
                d = pd.DataFrame(r)
                d["hold_out"] = ho
                rows.append(d)
            print(f"  {mk} hold_out={ho} done", flush=True)
    D = pd.concat(rows, ignore_index=True)
    D.to_csv("rbcc_holdout.csv", index=False)

    fin = D[D.step == NTASK - 1]
    g = (fin.groupby(["method", "protocol", "hold_out"])
            [["final_cov", "cov_gap", "worst_task_cov", "set_size", "acc"]].mean())
    print()
    print(g.round(4).to_string())
    print("\n=== effect on the reported fix (protocol C_rbcc) ===")
    for mk in METHODS:
        for pr in ("A_current", "C_rbcc", "Cn_replayed"):
            a = fin[(fin.method == mk) & (fin.protocol == pr) & (~fin.hold_out)]
            b = fin[(fin.method == mk) & (fin.protocol == pr) & (fin.hold_out)]
            if len(a) == 0 or len(b) == 0:
                continue
            print(f"  {mk:8s} {pr:12s} gap {a.cov_gap.mean():.4f} -> {b.cov_gap.mean():.4f}"
                  f"  cov {a.final_cov.mean():.4f} -> {b.final_cov.mean():.4f}")
    print(f"\n{len(D)} rows -> rbcc_holdout.csv")


if __name__ == "__main__":
    import glob
    fs = sys.argv[1:] or sorted(glob.glob("features/*.npz"))
    main(fs[0])
