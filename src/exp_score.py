"""Score-function robustness: does the coverage collapse survive the change from LAC to APS?

Section 3.6 of the manuscript makes a claim about this comparison, and a referee
correctly observed that the manuscript reports no APS numbers anywhere. The
comparison was run during development but the result file was never retained, so it
is re-derived here and written to a table that the paper can cite.

Why it matters: LAC can return the empty set, so a reader can reasonably suspect that
the coverage collapse is an artefact of that one score's capacity to abstain. APS is
non-empty by construction. If coverage still falls far below the nominal level under
APS, the collapse is a property of the learner rather than of the score.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
import cl_methods as M  # noqa: E402
import experiment as EX  # noqa: E402
import reliability as RL  # noqa: E402

ALPHA = 0.10
NTASK = 10
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2,3,4").split(",")]
METHODS = os.environ.get("METHODS", "Seq-FT,EWC,LwF,Replay,iCaRL,NCM,RanPAC").split(",")


def main(feat_files):
    rows = []
    for f in feat_files:
        z = np.load(f, allow_pickle=True)
        Xtr, ytr, Xte, yte = z["Xtr"], z["ytr"], z["Xte"], z["yte"]
        tag = os.path.basename(f).replace(".npz", "")
        ds, bb = tag.split("__")
        for score in ("lac", "aps"):
            for mk in METHODS:
                for seed in SEEDS:
                    r = EX.run_one(Xtr, ytr, Xte, yte, mk, NTASK, seed=seed,
                                   alpha=ALPHA, score=score)
                    d = pd.DataFrame(r)
                    # Protocol A is the practitioner protocol; the final model's view
                    # of every task it has seen is what the collapse claim is about.
                    fin = d[(d.protocol == "A_current") & (d.step == d.step.max())]
                    rows.append(dict(
                        dataset=ds, backbone=bb, method=mk, seed=seed, score=score,
                        final_cov=float(fin["cov"].mean()),
                        worst_task_cov=float(fin["cov"].min()),
                        worst_gap=float(max(0.0, (1 - ALPHA) - fin["cov"].min())),
                        set_size=float(fin["size"].mean()),
                        empty_frac=float((fin["size"] < 1e-9).mean()),
                        acc=float(fin["acc"].mean())))
                print(f"  {tag} {score} done", flush=True)
        pd.DataFrame(rows).to_csv("score_comparison.csv", index=False)
    D = pd.DataFrame(rows)
    D.to_csv("score_comparison.csv", index=False)
    g = D.groupby(["method", "score"])[["final_cov", "worst_task_cov", "set_size",
                                        "empty_frac"]].mean().unstack()
    print()
    print(g.round(4).to_string())
    lac = D[D.score == "lac"].final_cov.mean()
    aps = D[D.score == "aps"].final_cov.mean()
    print(f"\nmean final coverage: LAC {lac:.4f}  APS {aps:.4f}  (nominal {1-ALPHA})")
    print(f"{len(D)} rows -> score_comparison.csv")


if __name__ == "__main__":
    import glob
    fs = sys.argv[1:] or sorted(glob.glob("features/*.npz"))
    main(fs)
