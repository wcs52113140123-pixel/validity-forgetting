import os
ROOT = os.environ['VF_ROOT']
import os
import sys
import shutil
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

os.environ.update(KMP_AFFINITY="disabled", OMP_NUM_THREADS="8")


sys.path.insert(0, os.getcwd())

import tcl_kernel as T
import figs_tcl as FT
import figs_cns2 as F2

T.tcl_style()

PCm = pd.read_csv(os.path.join(ROOT, "results/main/protocol_comparison.csv"))
KS = pd.read_csv(os.path.join(ROOT, "results/ablation/reservation_size_records.csv"))
MK = pd.read_csv(os.path.join(ROOT, "results/ablation/class_conditional_reservation_size.csv"))
MF = pd.read_csv(os.path.join(ROOT, "results/ablation/class_conditional_forgetting.csv"))
EM = pd.read_csv(os.path.join(ROOT, "results/ablation/e2e_class_conditional.csv"))


L4 = ["aaaaaaa..bbbbbbbb..cccccccc.", "ddddddddddd..eeeeeeee..ffff.", "gggggggg..hhhhhhhhhhh..iiii."]


def build_fig4h(hr, figh, s3=None):
    plt.close("all")
    fig, A = T.tcl_mosaic(L4, figsize=(7.09, figh), hspace=0.72, wspace=0.25, height_ratios=hr)
    if s3 is None:
        T.tcl_placeholder(A["a"], "S3", "RBCC")
    else:
        A["a"].imshow(np.asarray(s3), interpolation="lanczos")
        A["a"].set_aspect("equal")
        A["a"].axis("off")
        A["a"].set_anchor("NW")
    A["a"].set_title("RBCC")
    F2.protocol_raincloud(A["b"], PCm, "Four calibration sets")
    F2.estimation(A["c"], PCm, "A_current", "C_rbcc", "Current → reserved")
    F2.tradeoff(A["d"], KS, "Gap–accuracy trade-off, 8 configurations")
    F2.k_bounds(A["e"], MK, "Set size vs k")
    F2.scheme_k_heat(A["f"], MK, "Balanced test set")
    F2.separation_area(A["g"], MF, "Old-class shift, k = 20")
    F2.halfeye(A["h"], MF, EM, "Coverage lost to the shift")
    F2.shift_dose(A["i"], MF, "Dose–response")
    T.tcl_finalize(fig, A)
    return fig, A



L4 = ["aaaaaaa..bbbbbbbb..cccccccc.", "ddddddddddd..eeeeeeee..ffff.", "ggggggg...hhhhhhhhhhh..iiii."]
