"""Fold the 150-epoch end-to-end matrix into the manuscript, tables and figure 6.

Written before the run lands so the update is one execution rather than a session
of improvisation against a deadline. Everything it reports is computed from the
records file; nothing is carried over from the 70-epoch draft.

The 70-epoch numbers are NOT discarded — they become the under-trained comparison
that Section 3.9 argues about, and the script writes both budgets into the same
table so the manuscript can state the direction of the difference rather than
quietly replacing one set of numbers with another.
"""
import os
import re

import numpy as np
import pandas as pd

NOM = 0.90
REPO = "validity-forgetting"


def load(path, epochs):
    d = pd.read_csv(path)
    d["epochs"] = epochs
    return d


def summarise(R):
    """Per-method final-step quantities, averaged over seeds, with 95% CIs."""
    from scipy import stats

    f = R[R.step == R.step.max()].copy()
    f["which"] = np.where(f.task == f.step, "current", "earlier")
    cur = (f[(f.which == "current") & (f.protocol == "A_current")]
           .groupby(["method", "seed"])["cov"].mean())
    ear = (f[(f.which == "earlier") & (f.protocol == "A_current")]
           .groupby(["method", "seed"])["cov"].mean())
    acc = (f[f.protocol == "A_current"].groupby(["method", "seed"])["acc"].mean())
    siz = (f[(f.which == "earlier") & (f.protocol == "A_current")]
           .groupby(["method", "seed"])["size"].mean())
    gap = (NOM - ear).clip(lower=0)

    def agg(s, name):
        g = s.groupby("method").agg(["mean", "std", "count"])
        hw = stats.t.ppf(0.975, g["count"] - 1) * g["std"] / np.sqrt(g["count"])
        return pd.DataFrame({name: g["mean"], f"{name}_ci": hw})

    out = pd.concat([agg(acc, "ACC"), agg(cur, "CovCurrent"),
                     agg(ear, "CovEarlier"), agg(gap, "CovGap"),
                     agg(siz, "SetSize")], axis=1)
    out["n_seeds"] = acc.groupby("method").size()
    return out.round(4)


def protocol_table(R):
    """Coverage on earlier tasks by calibration protocol, RBCC arm only."""
    from scipy import stats

    f = R[(R.step == R.step.max()) & (R.method == "E2E-RBCC")].copy()
    f = f[f.task != f.step]
    g = f.groupby(["protocol", "seed"])["cov"].mean().groupby("protocol").agg(
        ["mean", "std", "count"])
    g["ci95"] = stats.t.ppf(0.975, g["count"] - 1) * g["std"] / np.sqrt(g["count"])
    order = ["Cn_replayed", "A_current", "C_rbcc", "B_pooled"]
    return g.reindex([o for o in order if o in g.index]).round(4)


def disproportionality(S):
    """Coverage loss against accuracy loss, relative to the best-accuracy method.

    Reported as the RATIO OF MEANS, matching Section 3.2 — the mean of
    per-configuration ratios is unstable when a denominator approaches zero.
    """
    best = S.ACC.idxmax()
    d = pd.DataFrame({
        "acc_loss": S.ACC[best] - S.ACC,
        "cov_loss": S.CovEarlier[best] - S.CovEarlier,
    })
    d = d.drop(index=best)
    r = float(d.cov_loss.mean() / d.acc_loss.mean()) if d.acc_loss.mean() else np.nan
    return d.round(4), r


def matched_ratio(S, shared=("Seq-FT", "LwF", "Replay"), prefix=""):
    """Disproportionality on the method set both arms share, with a shared reference.

    The unrestricted ratio uses each arm's own best-accuracy method as reference,
    and the two arms do not have the same best: the frozen arm's is a prototype
    method (RanPAC, accuracy 0.855) that barely loses coverage, while the
    end-to-end arm has no prototype methods at all and tops out near 0.37. Quoting
    one arm's ratio beside the other's would compare values computed against
    structurally different references.

    Restricting both arms to {Seq-FT, LwF, Replay} with experience replay as the
    reference makes the comparison like-for-like. On the frozen arm this gives
    1.32, not the 4.27 that Section 3.2 reports for the full method set — the
    difference is the reference, not the training regime, and the manuscript has
    to say which number it is quoting.
    """
    idx = [f"{prefix}{m}" for m in shared]
    sub = S.loc[[i for i in idx if i in S.index]]
    if len(sub) < 2:
        return None, None
    acc_col = "ACC"
    cov_col = "CovEarlier" if "CovEarlier" in sub.columns else "FinalCov"
    ref = sub[acc_col].idxmax()
    al = (sub[acc_col][ref] - sub[acc_col]).drop(index=ref)
    cl = (sub[cov_col][ref] - sub[cov_col]).drop(index=ref)
    r = float(cl.mean() / al.mean()) if al.mean() else float("nan")
    return pd.DataFrame({"acc_loss": al, "cov_loss": cl}).round(4), r


def compare_budgets(S70, S150):
    """Side-by-side, with the direction Section 3.9 predicts made explicit."""
    j = S70[["ACC", "CovEarlier", "CovGap"]].join(
        S150[["ACC", "CovEarlier", "CovGap"]], lsuffix="_70", rsuffix="_150",
        how="outer")
    j["dCov"] = j.CovEarlier_150 - j.CovEarlier_70
    j["dGap"] = j.CovGap_150 - j.CovGap_70
    return j.round(4)


def patch_manuscript(path, repl, dry=False):
    """Apply (old, new) substitutions, asserting each matches exactly once."""
    s = open(path).read()
    applied = []
    for old, new in repl:
        n = s.count(old)
        if n != 1:
            applied.append((old[:60], f"SKIPPED — {n} matches"))
            continue
        s = s.replace(old, new, 1)
        applied.append((old[:60], "ok"))
    if not dry:
        open(path, "w").write(s)
    return applied, s


if __name__ == "__main__":
    import sys

    new_csv = sys.argv[1] if len(sys.argv) > 1 else "e2e_records_150ep.csv"
    R150 = load(new_csv, 150)
    R70 = load(f"{REPO}/results/ablation/e2e_records.csv", 70)
    S150, S70 = summarise(R150), summarise(R70)
    P150 = protocol_table(R150)
    D150, ratio150 = disproportionality(S150)
    M150, rm150 = matched_ratio(S150, prefix="E2E-")
    CMP = compare_budgets(S70, S150)

    S150.to_csv("e2e_summary_150ep.csv")
    P150.to_csv("e2e_protocols_150ep.csv")
    CMP.to_csv("e2e_budget_comparison.csv")
    D150.assign(ratio_of_means=ratio150).to_csv("e2e_disproportionality_150ep.csv")

    print("=== 150 轮，逐方法（均值 ± 95% CI） ===")
    print(S150.to_string())
    print("\n=== 协议对照（RBCC 臂，旧任务覆盖率） ===")
    print(P150.to_string())
    print(f"\n=== 不成比例倍数，端到端臂自身基准（{S150.ACC.idxmax()}）: {ratio150:.2f} ===")
    print(D150.to_string())
    if rm150 is not None:
        print(f"\n=== 同口径对照：限两臂共有方法、基准 Replay ===")
        print(M150.to_string())
        print(f"  端到端 {rm150:.2f}  vs  冻结臂 1.32（同口径）")
        print(f"  注意：§3.2 的 4.27 是冻结臂全方法集、基准 RanPAC，不可与此直接比较")
    print("\n=== 70 轮 vs 150 轮 ===")
    print(CMP.to_string())
    worse = int((CMP.dGap > 0).sum())
    tot = int(CMP.dGap.notna().sum())
    print(f"\n§3.9 预测失效幅度更大：{worse}/{tot} 个方法的覆盖缺口在 150 轮下增大")
