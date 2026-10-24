"""Derive every main-results table from results/main/sweep_records.csv.

    python derive_main_tables.py results/main

Writes sweep_aggregate.csv, protocol_comparison.csv, disproportionality_analysis.csv,
reranking_analysis.csv, mechanism_decomposition.csv (results/main) and
audit_null_validtaskfrac.csv (same folder). Same procedures as the paper's Sections 3.1-3.8.
"""
import os, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr, kendalltau, binom
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sweep as SW

FAM = {"Seq-FT": "Gradient", "EWC": "Gradient", "LwF": "Gradient", "Replay": "Replay",
       "iCaRL": "Prototype", "NCM": "Prototype", "RanPAC": "Prototype"}
# P(a valid split-conformal predictor meets 0.90 on a task), simulated for each benchmark's per-task
# calibration and test sizes (Beta-binomial; Section 3.3; see null_rates.py)
NULL_P = {"cifar100": 0.530, "cub200": 0.544, "imagenet_r": 0.541, "medmix26": 0.536}


def main(d):
    cwd = os.getcwd(); os.chdir(d)
    try:
        _, AGG = SW.aggregate(); AGG.to_csv("sweep_aggregate.csv", index=False)
        fr = []
        for p in ("A_current", "B_pooled", "C_rbcc", "Cn_replayed"):
            _, a = SW.aggregate(protocol=p); a["protocol"] = p; fr.append(a)
        P4 = pd.concat(fr, ignore_index=True); P4[P4.method.isin(["Replay", "iCaRL"])].to_csv("protocol_comparison.csv", index=False)
        J, R = [], []
        for (ds, bb), s in AGG.groupby(["dataset", "backbone"]):
            s = s.set_index("method"); b = s.ACC.idxmax()
            for m in s.index:
                al = s.loc[b, "ACC"] - s.loc[m, "ACC"]; cl = s.loc[b, "FinalCov"] - s.loc[m, "FinalCov"]
                J.append(dict(dataset=ds, backbone=bb, method=m, reference=b, acc_loss=al, cov_loss=cl,
                              ratio=cl / al if al != 0 else np.nan))
            r1 = s.ACC.rank(ascending=False); r2 = s.CovGap.rank(ascending=True)
            rho, prho = spearmanr(r1, r2); tau, ptau = kendalltau(r1, r2)
            R.append(dict(dataset=ds, backbone=bb, n=len(s), spearman_rho=rho, p_rho=prho, kendall_tau=tau, p_tau=ptau,
                          mean_rank_shift=float((r1 - r2).abs().mean()), max_rank_shift=int((r1 - r2).abs().max()),
                          best_accuracy=s.ACC.idxmax(), best_validity=s.CovGap.idxmin()))
        pd.DataFrame(J).to_csv("disproportionality_analysis.csv", index=False)
        pd.DataFrame(R).to_csv("reranking_analysis.csv", index=False)
        N = []
        for _, r in AGG.iterrows():
            k = int(round(r.ValidTaskFrac * 10)); p0 = NULL_P[r.dataset]
            N.append(dict(dataset=r.dataset, backbone=r.backbone, method=r.method, family=FAM[r.method],
                          vtf=r.ValidTaskFrac, k=k, p0=p0, pval=binom.cdf(k, 10, p0)))
        N = pd.DataFrame(N); N["sig"] = N.pval < 0.05; N.to_csv("audit_null_validtaskfrac.csv", index=False)
        REC = pd.read_csv("sweep_records.csv"); A = REC[REC.protocol == "A_current"].copy()
        A = A[A.step == A.groupby(["dataset", "backbone", "method", "seed"]).step.transform("max")]
        rows = []
        for k, g in A.groupby(["dataset", "backbone", "method", "seed"]):
            cur = g[g.task == g.step]; old = g[g.task != g.step]
            rows.append(dict(dataset=k[0], backbone=k[1], method=k[2], seed=k[3], cov_old=old["cov"].mean(),
                             cov_cur=float(cur["cov"].iloc[0]), dAURC=old["aurc"].mean() - float(cur["aurc"].iloc[0]),
                             dECE=old["ece"].mean() - float(cur["ece"].iloc[0]), covgap=max(0.0, 0.90 - old["cov"].mean())))
        G = pd.DataFrame(rows).groupby(["dataset", "backbone", "method"])[["cov_old", "cov_cur", "dAURC", "dECE", "covgap"]].mean().reset_index()
        G["family"] = G.method.map(FAM); G.to_csv("mechanism_decomposition.csv", index=False)
    finally:
        os.chdir(cwd)


if __name__ == "__main__":
    main(sys.argv[1])
