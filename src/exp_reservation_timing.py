"""How much does fitting the calibration slots bias the reserved-slot guarantee?

The proposition (Section 2.6 of the paper) requires the reserved slots never to have
been trained on. This script compares reserving them before the training pass with
reserving them after it, from the data just fitted. A point the
optimiser has fitted carries a systematically low conformal score, which shrinks the
quantile and makes the predictor under-cover.

This script measures the size of that bias with frozen pre-trained features by running the identical configuration with the
slots inside and outside the training pass. If the difference is small the published
numbers stand with a stated bound; if it is not, they have to be re-run. Either way
the answer is measured rather than argued.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
import cl_methods as M  # noqa: E402
import reliability as RL  # noqa: E402

ALPHA = 0.10
NTASK = 10
KS = [int(x) for x in os.environ.get("KS", "20,40").split(",")]
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2,3,4").split(",")]
REPLAY = int(os.environ.get("REPLAY_SLOTS", "20"))


def one(X, y, Xt, yt, n_classes, k, seed, hold_out):
    tasks = M.class_incremental_split(y, NTASK, seed=seed)
    mem = REPLAY + k
    mdl = M.Replay(X.shape[1], n_classes, seed=seed, mem_per_class=mem,
                   cal_per_class=k, hold_out_cal=hold_out)
    rng = np.random.default_rng(10_000 + seed)
    pool = []
    for tc in tasks:
        idx = np.where(np.isin(y, tc))[0]
        rng.shuffle(idx)
        n_hold = max(len(idx) // 5, 1)
        pool.append(idx[:n_hold])
        mdl.observe(X[idx[n_hold:]], y[idx[n_hold:]], tc)

    seen = np.concatenate(tasks)
    te = np.isin(yt, seen)
    Pte, yte = mdl.predict_proba(Xt[te]), yt[te]
    cX, cy = mdl.cal_buffer()
    if cX is None or len(cy) == 0:
        return None
    sfn = RL.score_fn("lac")
    pooled_idx = np.concatenate(pool)

    out = []
    for name, sc, lab, rule in (
            ("pooled_oracle", sfn(mdl.predict_proba(X[pooled_idx]), y[pooled_idx]),
             y[pooled_idx], "pooled"),
            ("rbcc_pooled", sfn(mdl.predict_proba(cX), cy), cy, "pooled"),
            ("rbcc_mondrian", sfn(mdl.predict_proba(cX), cy), cy, "mondrian")):
        if rule == "pooled":
            S = RL.predict_sets(Pte, RL.conformal_quantile(sc, ALPHA))
            nfb = 0
        else:
            qv, nfb, _ = RL.mondrian_quantiles(sc, lab, ALPHA, n_classes)
            S = RL.predict_sets_mondrian(Pte, qv, "lac")
        per, worst, _ = RL.class_conditional_coverage(S, yte, n_classes)
        ok = ~np.isnan(per)
        out.append(dict(scheme=name, k=k, seed=seed, hold_out=hold_out,
                        mean_class=float(np.nanmean(per)), worst_class=float(worst),
                        frac_classes_ok=float((per[ok] >= 1 - ALPHA - 1e-9).mean()),
                        marginal=float(S[np.arange(len(yte)), yte].mean()),
                        size=float(S.sum(1).mean()),
                        acc=float((Pte.argmax(1) == yte).mean()),
                        n_fallback_classes=int(nfb)))
    return out


def main(feat):
    z = np.load(feat, allow_pickle=True)
    X, y, Xt, yt = z["Xtr"], z["ytr"], z["Xte"], z["yte"]
    n_classes = int(max(y.max(), yt.max())) + 1
    rows = []
    for k in KS:
        for ho in (False, True):
            for s in SEEDS:
                r = one(X, y, Xt, yt, n_classes, k, s, ho)
                if r:
                    rows.append(pd.DataFrame(r))
            print(f"  k={k} hold_out={ho} done", flush=True)
    D = pd.concat(rows, ignore_index=True)
    D.to_csv("holdout_effect.csv", index=False)
    g = D.groupby(["k", "scheme", "hold_out"])[["mean_class", "size", "acc"]].mean()
    print()
    print(g.round(4).to_string())
    print("\n=== effect of holding the slots out ===")
    for k in KS:
        for s in ("rbcc_pooled", "rbcc_mondrian", "pooled_oracle"):
            a = D[(D.k == k) & (D.scheme == s) & (~D.hold_out)].mean_class.mean()
            b = D[(D.k == k) & (D.scheme == s) & (D.hold_out)].mean_class.mean()
            print(f"  k={k:>2} {s:14s} in-training {a:.4f} -> held-out {b:.4f}  "
                  f"delta {b - a:+.4f}")
    print(f"\n{len(D)} rows -> holdout_effect.csv")


if __name__ == "__main__":
    import glob
    fs = sys.argv[1:] or sorted(glob.glob("features/*.npz"))
    main(fs[0])
