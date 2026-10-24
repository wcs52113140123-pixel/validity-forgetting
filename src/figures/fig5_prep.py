import os
ROOT = os.environ['VF_ROOT']
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import importlib
from PIL import Image
from scipy import stats as st

os.environ.update(KMP_AFFINITY="disabled")
sys.path.insert(0, os.getcwd())

import shutil


import tcl_kernel as T
import figs_cns2 as F2
import figs_tcl as FT

T.tcl_style()

SA = pd.read_csv(os.path.join(ROOT, "results/main/sweep_aggregate.csv"))
EC = pd.read_csv(os.path.join(ROOT, "results/ablation/e2e_ceiling.csv"))
EV = pd.read_csv(os.path.join(ROOT, "results/ablation/e2e_vs_frozen_backbone_150ep.csv"))
EP = pd.read_csv(os.path.join(ROOT, "results/ablation/e2e_protocols_150ep.csv"))
SC = pd.read_csv(os.path.join(ROOT, "results/ablation/score_comparison.csv"))

FAM = FT.FAM
SA["family"] = SA.method.map(FAM)

L5 = ["aaaaaaaa..bbbbbbb..ccccccccccc", "dddddd..eeeeee..ffffff..gggggg"]

def build_fig5(hr=(1.5, 1), figh=4.25, s4=None):
    plt.close("all")
    fig, A = T.tcl_mosaic(L5, figsize=(7.09, figh), hspace=0.72, wspace=0.25,
                          height_ratios=list(hr),
                          per_subplot_kw={"g": {"projection": "polar"}})
    if s4 is None:
        T.tcl_placeholder(A["a"], "S4", "Two regimes")
    else:
        A["a"].imshow(np.asarray(s4), interpolation="lanczos")
        A["a"].set_aspect("equal")
        A["a"].axis("off")
        A["a"].set_anchor("NW")
    A["a"].set_title("Two regimes")
    F2.fit_curve(A["b"], EC, "Fit adequacy")
    F2.parallel_coords(A["c"], EV, "Deployment check across regimes")
    F2.protocol_lollipop_inset(A["d"], EP, None, "Calibration, end-to-end")
    F2.bland_altman(A["e"], SA, "ViT vs DINOv2")
    F2.connected_scores(A["f"], SC, "LAC → APS")
    F2.radar(A["g"], SA, "Four datasets")
    T.tcl_finalize(fig, A)
    return fig, A

