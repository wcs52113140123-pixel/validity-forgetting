"""Class-conditional validity of the reserved budget: the experiment behind the
proposition in Section 2.6 of the paper.

Three calibration schemes are compared at the end of a ten-task class-incremental
stream, all on the SAME trained model so that any difference is attributable to the
calibration set and the thresholding rule alone:

    pooled-oracle      all past data, one threshold          (needs full storage)
    rbcc-pooled        k reserved slots per class, one threshold
    rbcc-mondrian      k reserved slots per class, per-class thresholds   (ours)

and measured on three quantities:

    marginal coverage          what the field reports
    class-conditional coverage what a continual setting actually asks for
    coverage under a deployment mixture re-weighted toward the oldest classes

The proposition requires k >= ceil(1/alpha) - 1 (= 9 at alpha = 0.10); the sweep
therefore runs k from below that bound to well above it, so the predicted threshold
effect is visible rather than assumed.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cl_methods as M
import reliability as RL

ALPHA = 0.10

# Budget knobs come from the environment so a sweep cannot silently run one setting
# repeatedly. REPLAY_SLOTS fixes the slots the optimiser sees; KS lists the reserved
# budgets. Total memory is their sum, which is what Replay partitions.
import os as _os
REPLAY_SLOTS = int(_os.environ.get("REPLAY_SLOTS", "20"))
KS = tuple(int(x) for x in _os.environ.get("KS", "3,5,7,9,12,16,20").split(","))
TAG = _os.environ.get("TAG", "records")
# Withhold the calibration slots from the training pass, which is what the
# proposition requires. Reserving them after training instead (HOLD_OUT=0) biases
# class-conditional coverage by about 0.018 at k=20 (results/validation/holdout_effect.csv).
HOLD_OUT = _os.environ.get("HOLD_OUT", "0") == "1"
SEEDS = (0, 1, 2, 3, 4)
NTASK = 10


def run_cell(X, y, Xt, yt, k, seed, n_classes, score="lac"):
    """One (k, seed) cell: train RBCC, then calibrate three ways on the same model."""
    tasks = M.class_incremental_split(y, NTASK, seed=seed)
    # The budget has to exceed the largest k on the sweep, or the high-k end would
    # confound a stronger guarantee with a crippled replay set: Replay clips
    # cal_per_class to mem_per_class-1, so k=20 at mem=20 leaves ONE replay exemplar
    # per class. At mem=40 every k on the sweep leaves at least 20 replay slots, so
    # the only thing varying across k is how the budget is split.
    mem = REPLAY_SLOTS + k
    # RBCC is a configuration of Replay, not a separate class: cal_per_class>0
    # partitions the per-class budget into replay and calibration-reserved slots.
    mdl = M.Replay(X.shape[1], n_classes, seed=seed, mem_per_class=mem, cal_per_class=k, hold_out_cal=HOLD_OUT)
    cal_pool_idx = []
    rng = np.random.default_rng(10_000 + seed)
    for t, tc in enumerate(tasks):
        idx = np.where(np.isin(y, tc))[0]
        # The oracle's calibration pool is HELD OUT of training. An earlier version of
        # this script sampled the pool from the data the model had just fitted, which
        # biases its scores low, shrinks the quantile and makes the oracle undercover
        # by construction. That is the same exchangeability violation this paper
        # argues about for replayed exemplars (Section 2.5), and the oracle must not
        # be handicapped by it: the comparison is only meaningful if every scheme gets
        # a calibration set the optimiser has never seen.
        rng.shuffle(idx)
        n_hold = max(int(len(idx) // 5), 1)
        hold, fit = idx[:n_hold], idx[n_hold:]
        mdl.observe(X[fit], y[fit], tc)
        cal_pool_idx.append(hold)

    seen = np.concatenate(tasks)
    te = np.isin(yt, seen)
    Pte, yte = mdl.predict_proba(Xt[te]), yt[te]
    sfn = RL.score_fn(score)

    pooled_idx = np.concatenate(cal_pool_idx)
    cX, cy = mdl.cal_buffer()
    out = {}
    if cX is None or len(cy) == 0:
        return None

    schemes = {
        "pooled_oracle": (sfn(mdl.predict_proba(X[pooled_idx]), y[pooled_idx]), y[pooled_idx], "pooled"),
        "rbcc_pooled":   (sfn(mdl.predict_proba(cX), cy), cy, "pooled"),
        "rbcc_mondrian": (sfn(mdl.predict_proba(cX), cy), cy, "mondrian"),
    }
    # the oldest half of the classes define the shifted deployment mixture
    old = set(np.concatenate(tasks[: NTASK // 2]).tolist())
    for name, (sc, lab, rule) in schemes.items():
        if rule == "pooled":
            q = RL.conformal_quantile(sc, ALPHA)
            S = RL.predict_sets(Pte, q, score)
            nfb = 0
        else:
            qv, nfb, _ = RL.mondrian_quantiles(sc, lab, ALPHA, n_classes)
            S = RL.predict_sets_mondrian(Pte, qv, score)
        per, worst, _ = RL.class_conditional_coverage(S, yte, n_classes)
        ok = ~np.isnan(per)
        marg = float(S[np.arange(len(yte)), yte].mean())
        row = dict(scheme=name, k=k, seed=seed, n_cal=len(lab), n_fallback_classes=int(nfb),
                   marginal=marg, worst_class=float(worst),
                   frac_classes_ok=float((per[ok] >= 1 - ALPHA - 1e-9).mean()),
                   mean_class=float(np.nanmean(per)), size=float(S.sum(1).mean()),
                   acc=float((Pte.argmax(1) == yte).mean()))
        for w, tag in ((1.0, "old100"), (0.8, "old80"), (0.5, "uniform")):
            wt = np.array([w if c in old else (1 - w) for c in range(n_classes)], dtype=float)
            row[f"cov_{tag}"] = RL.coverage_under_mixture(S, yte, wt)
        out[name] = row
    return list(out.values())


def main(feat_files):
    rows = []
    for f in feat_files:
        z = np.load(f, allow_pickle=True)
        X, y, Xt, yt = z["Xtr"], z["ytr"], z["Xte"], z["yte"]
        meta = json.loads(str(z["meta"])) if "meta" in z else {}
        nC = int(meta.get("n_classes", int(max(y.max(), yt.max())) + 1))
        tag = os.path.basename(f).replace(".npz", "")
        for k in KS:
            for s in SEEDS:
                t0 = time.time()
                r = run_cell(X, y, Xt, yt, k, s, nC)
                if r is None:
                    continue
                for d in r:
                    d["cell"] = tag; d["replay_slots"] = REPLAY_SLOTS
                rows.append(pd.DataFrame(r))
                print(f"{tag} k={k} seed={s}  {time.time()-t0:.0f}s", flush=True)
                pd.concat(rows).to_csv(f"mondrian_{TAG}.csv", index=False)
    D = pd.concat(rows)
    D.to_csv(f"mondrian_{TAG}.csv", index=False)
    print(f"\n{len(D)} rows -> mondrian_{TAG}.csv  (replay={REPLAY_SLOTS}, KS={KS})")
    g = D.groupby(["scheme", "k"])[["marginal", "worst_class", "frac_classes_ok",
                                    "size", "cov_old100"]].mean()
    print(g.round(4).to_string())


if __name__ == "__main__":
    fs = sys.argv[1:] or sorted(__import__("glob").glob("features/*.npz"))
    main(fs)
