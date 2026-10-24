import os
ROOT = os.environ['VF_ROOT']
import os
import sys
import importlib
import numpy as np
import pandas as pd
from scipy import stats as st
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

os.environ.update(KMP_AFFINITY="disabled", OMP_NUM_THREADS="8")
sys.path.insert(0, os.getcwd())

import tcl_kernel as T
import figs_tcl as FT
import figs_cns2 as F2

RD = os.path.join(ROOT, "results")
SR = pd.read_csv(f"{RD}/main/sweep_records.csv")
SA = pd.read_csv(f"{RD}/main/sweep_aggregate.csv")
RR = pd.read_csv(f"{RD}/main/reranking_analysis.csv")
NU = pd.read_csv(f"{RD}/validation/audit_null_validtaskfrac.csv")

FAM = FT.FAM
SA["family"] = SA.method.map(FAM)
SR["family"] = SR.method.map(FAM)
NU["family"] = NU.method.map(FAM)

rows = []
for (ds, bb), g in SA.groupby(["dataset", "backbone"]):
    best = g.loc[g.ACC.idxmax()]
    for _, r in g.iterrows():
        rows.append(dict(dataset=ds, backbone=bb, method=r.method, family=r.family,
                         acc_loss=best.ACC - r.ACC, cov_loss=best.FinalCov - r.FinalCov, best=best.method))
DPx = pd.DataFrame(rows)

T.tcl_style()

L2 = ["aaaa..bbbbbbb...cccccccc",
      "aaaa..ddddddd...eeeeeeee",
      "aaaa..fffffff...gggggggg"]

def build_fig2():
    plt.close("all")
    fig, A = T.tcl_mosaic(L2, figsize=(7.09, 5.4), hspace=0.62, wspace=0.25)
    fo = F2.loss_forest(A["a"], DPx, "Loss, 56 configurations")
    rr = F2.ratio_raincloud(A["b"], DPx, "Loss ratio, bootstrap")
    F2.ridgeline(A["c"], SR, "Per-task coverage")
    nl = F2.null_pmf(A["d"], NU, "Against a valid predictor")
    ec = F2.ecdf(A["e"], SR, "The mean hides the tail")
    al = F2.rank_alluvial(A["f"], SA, "Rank flow")
    rl = F2.rho_lollipop(A["g"], SA, RR, "Rank agreement")
    T.tcl_finalize(fig, A)
    return fig, A, dict(forest=fo, ratio=rr, null=nl, ecdf=ec, same_rank=al)

