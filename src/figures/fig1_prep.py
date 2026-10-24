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
seen = A_[A_.task <= A_.step].groupby(
    ["dataset", "backbone", "method", "family", "seed", "step"]
)[["acc", "cov"]].mean().reset_index()
traj = seen.groupby(
    ["family", "dataset", "backbone", "step"]
)[["acc", "cov"]].mean().reset_index()
TR = traj.groupby(["family", "step"]).agg(
    acc=("acc", "mean"), acc_sd=("acc", "std"),
    cov=("cov", "mean"), cov_sd=("cov", "std"),
    n=("acc", "size")
).reset_index()
TR["acc_ci"] = 1.96 * TR.acc_sd / np.sqrt(TR.n)
TR["cov_ci"] = 1.96 * TR.cov_sd / np.sqrt(TR.n)

fin = A_[A_.step == A_.step.max()]
MAT = fin[
    (fin.dataset == "medmix26") & (fin.backbone == "vit_b16_in21k")
].groupby(["method", "task"])["cov"].mean().unstack().loc[FT.METHOD_ORDER]

S1img = Image.open(os.path.join(ROOT, "figures/schematics/S1_Fig1a.png")).convert("RGB")

L1 = ["aaaaaaaaaa..bbbbbbbbbb", "cccccccccccccccccccccc", "dddddddd..eeeee..ffff."]

def build_fig1():
    plt.close("all")
    fig, A = T.tcl_mosaic(
        L1, figsize=(7.09, 4.85), hspace=0.55, wspace=0.25,
        height_ratios=[0.95, 0.85, 1.2]
    )
    A["a"].imshow(np.asarray(S1img), interpolation="lanczos")
    A["a"].set_aspect("equal")
    A["a"].axis("off")
    A["a"].set_anchor("NW")
    A["a"].set_title("Setting")
    F2.gauge_matrix(A["b"], SA, "Final coverage, 56 configurations")
    F2.divergence_multiples(A["c"], TR, "Accuracy (dashed) and coverage (solid) of seen classes, by family")
    F2.bubble_matrix(A["d"], MAT, "Per-task coverage, MedMix-26 ViT")
    F2.paired_swarm(A["e"], MC, "Deployment check")
    F2.joint_size(A["f"], SA, "Set size")
    T.tcl_finalize(fig, A)
    return fig, A

