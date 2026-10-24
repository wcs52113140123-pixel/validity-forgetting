"""R1b: reservation-size sweep on the paper's four benchmarks (corrected reservation).
Experience replay, total memory m = 20 per class, k reserved of those, five class orderings."""
import os, sys, time, itertools
os.environ.setdefault("KMP_AFFINITY", "disabled")
sys.path.append(sys.argv[1])
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
import experiment as EX
M_TOTAL, K_GRID, SEEDS = 20, [0, 1, 2, 3, 5, 7, 10], range(5)
def one(args):
    ds, bb, k, sd = args
    z = np.load(f"features/{ds}__{bb}.npz"); Xtr, ytr, Xte, yte = z["Xtr"], z["ytr"], z["Xte"], z["yte"]
    ncl = int(len(np.unique(ytr))); nt = 10 if ncl % 10 == 0 else max(2, min(10, ncl // 2))
    df = pd.DataFrame(EX.run_one(Xtr, ytr, Xte, yte, "Replay", nt, sd, score="lac",
                                 mem_per_class=M_TOTAL, cal_per_class=k, hold_out_cal=True))
    S = EX.summarise(df, nt, alpha=0.10)
    rows = [{"dataset": ds, "backbone": bb, "k": k, "seed": sd, "m_total": M_TOTAL, "protocol": r["protocol"],
             "ACC": r["ACC"], "FinalCov": r["FinalCov"], "CovGap": r["CovGap"], "WorstTaskGap": r["WorstTaskGap"],
             "SetSize": r.get("SetSize", np.nan), "ValidTaskFrac": r.get("ValidTaskFrac", np.nan)} for _, r in S.iterrows()]
    return rows
if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "4"
    cells = list(itertools.product(["cifar100","cub200","imagenet_r","medmix26"], ["vit_b16_in21k","dinov2_base"], K_GRID, SEEDS))
    out = []; t0 = time.time()
    with ProcessPoolExecutor(max_workers=28) as ex:
        for i, rows in enumerate(ex.map(one, cells)):
            out += rows
            if i % 20 == 0: print(i, f"{time.time()-t0:.0f}s", flush=True)
    pd.DataFrame(out).to_csv("ksweep_bench_ho.csv", index=False); print("done", len(cells))
