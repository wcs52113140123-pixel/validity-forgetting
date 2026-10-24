"""Main study, replay-memory methods: calibration slots are drawn before the training pass
and withheld from it (Section 2.5)."""
import os, sys, time, itertools
os.environ.setdefault("KMP_AFFINITY", "disabled")
sys.path.append(sys.argv[1])
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
import experiment as EX
OUT = "sweep_records_ho.csv"
def one(args):
    ds, bb, mk, sd = args
    z = np.load(f"features/{ds}__{bb}.npz"); Xtr, ytr, Xte, yte = z["Xtr"], z["ytr"], z["Xte"], z["yte"]
    ncl = int(len(np.unique(ytr))); nt = 10 if ncl % 10 == 0 else max(2, min(10, ncl // 2))
    t0 = time.time()
    df = pd.DataFrame(EX.run_one(Xtr, ytr, Xte, yte, mk, n_tasks=nt, seed=sd, score="lac",
                                 mem_per_class=20, cal_per_class=5, hold_out_cal=True))
    df["dataset"] = ds; df["backbone"] = bb; df["n_classes"] = ncl
    return df, f"{ds} {bb} {mk} s{sd} {time.time()-t0:.0f}s"
if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "4"
    cells = list(itertools.product(["cifar100","cub200","imagenet_r","medmix26"], ["vit_b16_in21k","dinov2_base"],
                                   ["Replay","iCaRL"], range(5)))
    out = []
    with ProcessPoolExecutor(max_workers=24) as ex:
        for df, msg in ex.map(one, cells):
            out.append(df); print(msg, flush=True)
    pd.concat(out).to_csv(OUT, index=False); print("done", len(out), "cells")
