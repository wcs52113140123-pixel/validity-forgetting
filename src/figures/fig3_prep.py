import os
ROOT = os.environ['VF_ROOT']
import os
import sys
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy import stats as st

os.environ.update(KMP_AFFINITY="disabled", OMP_NUM_THREADS="8")
sys.path.insert(0, os.getcwd())


import tcl_kernel as T
import figs_tcl as FT
import figs_cns2 as F2

T.tcl_style()

SR = pd.read_csv(os.path.join(ROOT, "results/main/sweep_records.csv"))
SA = pd.read_csv(os.path.join(ROOT, "results/main/sweep_aggregate.csv"))
MC = pd.read_csv(os.path.join(ROOT, "results/main/mechanism_decomposition.csv"))

FAM = FT.FAM
SA["family"] = SA.method.map(FAM)
SR["family"] = SR.method.map(FAM)

A_ = SR[SR.protocol == "A_current"]
fin = A_[A_.step == A_.step.max()]

MATc = fin[(fin.dataset == "cifar100") & (fin.backbone == "vit_b16_in21k")].groupby(["method", "task"])["cov"].mean().unstack().loc[FT.METHOD_ORDER]


L3n = ["aaaaaaa..bbbbbbbbbb..ccccccc", "aaaaaaa..bbbbbbbbbb..ddddddd", "eeeeeeee.....ffffff.ggggggg."]

def build_fig3n(s2img=None, figh=5.2):
    plt.close("all")
    fig, A = T.tcl_mosaic(L3n, figsize=(7.09, figh), hspace=0.75, wspace=0.25, height_ratios=[1, 1, 1.3],
                          per_subplot_kw={"g": {"projection": "polar"}})
    if s2img is None:
        T.tcl_placeholder(A["a"], "S2", "Two mechanisms")
    else:
        A["a"].imshow(np.asarray(s2img), interpolation="lanczos")
        A["a"].set_aspect("equal")
        A["a"].axis("off")
        A["a"].set_anchor("N")
    A["a"].set_title("Two mechanisms")
    F2.joint_mechanism(A["b"], MC, "Coverage gap follows ranking")
    F2.commonality(A["c"], MC, "Variance partition")
    F2.beta_densities(A["d"], MC, "Joint regression")
    F2.strip_heat(A["e"], SR, "Seen classes along the stream")
    F2.split_violin(A["f"], MC, "Both drift, by family")
    F2.polar_heat(A["g"], MATc, "Per task, CIFAR-100 ViT")
    T.tcl_finalize(fig, A)
    return fig, A

