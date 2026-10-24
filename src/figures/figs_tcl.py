"""Figure builders for the validity-forgetting paper, CNS layout (180 mm, aligned).

Every builder takes an Axes and tidy data, draws one panel, sets a DESCRIPTIVE title
(dataset / setting, not a claim; the claim goes in the caption) and returns any
numbers the caption quotes. Colours come from tcl_kernel semantics only.
"""
import numpy as np
import pandas as pd
from scipy import stats as st
import tcl_kernel as T

FAM = {"Seq-FT": "Gradient", "EWC": "Gradient", "LwF": "Gradient", "Replay": "Replay",
       "iCaRL": "Prototype", "NCM": "Prototype", "RanPAC": "Prototype"}
ROLE = {"Gradient": "gradient_family", "Replay": "replay_family", "Prototype": "prototype_family"}
FAM_ORDER = ["Prototype", "Replay", "Gradient"]
METHOD_ORDER = ["NCM", "iCaRL", "RanPAC", "Replay", "EWC", "LwF", "Seq-FT"]
NOM = 0.90


def family_trajectory(ax, TR, col, ylabel, title, nominal=False, shade_below=False, ylim=(0.25, 1.0)):
    """Seen-class metric across the stream, one line per family, band = 95% CI over
    the eight dataset-backbone configurations."""
    if shade_below:
        T.tcl_regions(ax, [(0.0, NOM, "failing_default", None)])
    ends = []
    for f in FAM_ORDER:
        d = TR[TR.family == f].sort_values("step")
        x = d.step.values + 1
        T.tcl_curve(ax, x, d[col].values, (d[col] - d[col + "_ci"]).values,
                    (d[col] + d[col + "_ci"]).values, role=ROLE[f], label=f.lower(), ms=2.0,
                    end_label=False)
        ends.append((x[-1], d[col].values[-1], f.lower(), ROLE[f]))
    ax.set_ylim(*ylim)
    T.tcl_end_labels(ax, ends)
    if nominal:
        T.tcl_nominal(ax, NOM, label="promised 0.90", side=0.42)
    ax.set_xlim(0.6, 14.2)
    ax.set_xticks([1, 4, 7, 10])
    ax.set_xlabel("tasks learned")
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def deployment_check(ax, MCm, title):
    """Coverage on the task just learned vs on earlier tasks, per method (dumbbell)."""
    m = list(MCm.index)
    d = MCm.loc[m]
    T.tcl_dumbbell(ax, d.cov_old.values, d.cov_cur.values, m, role_a="failing_default",
                   role_b="oracle", label_a="earlier tasks", label_b="just learned", size=16)
    T.tcl_nominal(ax, NOM, axis="x")
    ax.set_ylim(len(m) - 0.5, -0.5)          # top row = first method, same as the matrix
    ax.set_yticks(range(len(m))); ax.set_yticklabels(m)
    for t in ax.get_yticklabels():
        t.set_color(T.tcl_color(ROLE[FAM[t.get_text()]], "text"))
    ax.set_xlim(0.0, 1.02)
    ax.set_xticks([0, 0.5, 0.9])
    ax.set_xlabel("coverage")
    ax.legend(loc="upper left", fontsize=5.2, bbox_to_anchor=(-0.03, 1.03), handletextpad=0.2,
              borderaxespad=0.1)
    ax.set_title(title)
    return dict(cur_min=float(d.cov_cur.min()), cur_max=float(d.cov_cur.max()),
                old_min=float(d.cov_old.min()), old_max=float(d.cov_old.max()))


def task_matrix(ax, MAT, title, cbar=True, row_labels=False, annotate=False):
    """Per-task coverage of the final model, methods x tasks, centred on the promise."""
    T.tcl_coverage_matrix(ax, MAT.values, row_labels=list(MAT.index) if row_labels else None,
                          col_labels=[f"T{int(c) + 1}" for c in MAT.columns], nominal=NOM,
                          cbar=cbar, cbar_label=None, annotate=annotate)
    ax.set_xticks(range(0, MAT.shape[1], 3)); ax.set_xticklabels([f"T{c + 1}" for c in range(0, MAT.shape[1], 3)])
    for t in ax.get_yticklabels():
        if t.get_text() in FAM:
            t.set_color(T.tcl_color(ROLE[FAM[t.get_text()]], "text"))
    ax.set_title(title)


def cells_scatter(ax, SA, x, y, xlabel, ylabel, title, logy=False, hline=None, vline=None, size=11):
    """One point per method-dataset-backbone cell, coloured by family."""
    for f in FAM_ORDER:
        d = SA[SA.family == f]
        ax.scatter(d[x], d[y], s=size, fc=T.tcl_color(ROLE[f], "fill"),
                   ec=T.tcl_color(ROLE[f], "stroke"), lw=0.5, zorder=3, label=f.lower())
    if hline is not None:
        T.tcl_nominal(ax, hline, label=None)
    if vline is not None:
        T.tcl_nominal(ax, vline, axis="x")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


# ============================================================ shared encodings
PROTO = ["A_current", "Cn_replayed", "C_rbcc", "B_pooled"]
PROTO_LABEL = {"A_current": "current task", "Cn_replayed": "replayed", "C_rbcc": "reserved (RBCC)",
               "B_pooled": "all past data"}
PROTO_ROLE = {"A_current": "failing_default", "Cn_replayed": "replayed_exemplars", "C_rbcc": "ours",
              "B_pooled": "oracle"}
SCHEME = ["pooled_oracle", "rbcc_pooled", "rbcc_mondrian"]
PROTO_SHORT = {"A_current": "current", "Cn_replayed": "replayed", "C_rbcc": "reserved", "B_pooled": "all past"}
SCHEME_SHORT = {"pooled_oracle": "all past", "rbcc_pooled": "one threshold", "rbcc_mondrian": "per class"}
SCHEME_LABEL = {"pooled_oracle": "all past data", "rbcc_pooled": "reserved, one threshold",
                "rbcc_mondrian": "reserved, per class"}
DS_LABEL = {"cifar100": "CIFAR-100", "cub200": "CUB-200", "imagenet_r": "ImageNet-R", "medmix26": "MedMix-26"}
BB_LABEL = {"vit_b16_in21k": "ViT", "dinov2_base": "DINOv2"}


def scheme_style(s):
    """Oracle neutral; the two reserved-slot schemes share blue and differ in tone /
    fill (colour encodes one dimension: whose calibration data)."""
    if s == "pooled_oracle":
        return dict(c=T.tcl_color("oracle", "stroke"), f=T.tcl_color("oracle", "fill"), hollow=False, marker="s")
    if s == "rbcc_pooled":
        return dict(c=T.tcl_color("ours", "mid"), f="white", hollow=True, marker="o")
    return dict(c=T.tcl_color("ours", "stroke"), f=T.tcl_color("ours", "fill"), hollow=False, marker="o")


def ci95(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    return 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0


def _fam_points(ax, d, x, y, size=10):
    for f in FAM_ORDER:
        s = d[d.family == f]
        ax.scatter(s[x], s[y], s=size, fc=T.tcl_color(ROLE[f], "fill"), ec=T.tcl_color(ROLE[f], "stroke"),
                   lw=0.5, zorder=3, label=f.lower())


def _fit_line(ax, x, y, role="oracle", txt=True, loc=(0.04, 0.92)):
    x = np.asarray(x, float); y = np.asarray(y, float)
    r = st.linregress(x, y)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, r.intercept + r.slope * xs, color=T.TCL_NEUTRAL["ink"], lw=0.8, ls=(0, (3, 2)), zorder=2)
    if txt:
        ax.text(*loc, f"R² = {r.rvalue**2:.2f}", transform=ax.transAxes, fontsize=5.4, va="top",
                color=T.TCL_NEUTRAL["ink"])
    return r


# ============================================================ figure 2
def excess_loss(ax, DPx, title):
    _fam_points(ax, DPx, "acc_loss", "cov_loss")
    lim = max(DPx.acc_loss.max(), DPx.cov_loss.max()) * 1.05
    ax.plot([0, lim], [0, lim], color=T.TCL_NEUTRAL["grey"], lw=0.8, ls=(0, (4, 2.5)), zorder=1)
    ax.text(lim * 0.72, lim * 0.62, "y = x", fontsize=5.2, ha="left", va="top", color=T.TCL_NEUTRAL["grey"])
    ax.set_xlim(-0.02, lim); ax.set_ylim(-0.02, lim)
    ax.set_xlabel("accuracy loss"); ax.set_ylabel("coverage loss")
    ax.legend(loc="upper left", fontsize=5.2, handletextpad=0.1, borderaxespad=0.1)
    ax.set_title(title)


def ratio_of_means(ax, DPx, title, methods=("Seq-FT", "EWC", "LwF", "Replay"), n_boot=4000, seed=0):
    """Coverage loss / accuracy loss per method as a RATIO OF MEANS (the per-cell ratio
    is unstable where the accuracy loss is small), with a bootstrap 95% CI over cells."""
    rng = np.random.default_rng(seed); out = {}
    for i, m in enumerate(methods):
        d = DPx[DPx.method == m]
        a, c = d.acc_loss.values, d.cov_loss.values
        bs = []
        for _ in range(n_boot):
            k = rng.integers(0, len(a), len(a))
            if a[k].mean() > 1e-6:
                bs.append(c[k].mean() / a[k].mean())
        rom = c.mean() / a.mean(); lo, hi = np.percentile(bs, [2.5, 97.5])
        role = ROLE[FAM[m]]
        ax.plot([lo, hi], [i, i], color=T.tcl_color(role, "stroke"), lw=1.0, zorder=2)
        ax.scatter([rom], [i], s=22, fc=T.tcl_color(role, "fill"), ec=T.tcl_color(role, "stroke"), lw=0.8, zorder=3)
        ax.text(hi * 1.06, i, f"{rom:.2f}", va="center", fontsize=5.4, color=T.tcl_color(role, "text"))
        out[m] = (rom, lo, hi)
    ax.axvline(1, color=T.TCL_NEUTRAL["grey"], lw=0.8, ls=(0, (4, 2.5)))
    ax.set_xscale("log"); ax.set_xticks([1, 2, 4, 8]); ax.set_xticklabels(["1", "2", "4", "8"])
    ax.set_xlim(0.8, 14)
    ax.set_yticks(range(len(methods))); ax.set_yticklabels(methods); ax.set_ylim(len(methods) - 0.5, -0.5)
    for t in ax.get_yticklabels():
        t.set_color(T.tcl_color(ROLE[FAM[t.get_text()]], "text"))
    ax.set_xlabel("coverage loss / accuracy loss")
    ax.set_title(title)
    return out


def task_cov_violins(ax, SR, title):
    A = SR[(SR.protocol == "A_current")]
    fin = A[A.step == A.step.max()].groupby(["dataset", "backbone", "method", "family", "task"])["cov"].mean().reset_index()
    T.tcl_violin_points(ax, [fin[fin.family == f]["cov"].values for f in FAM_ORDER],
                        [ROLE[f] for f in FAM_ORDER], [f.lower() for f in FAM_ORDER], point_size=1.6)
    T.tcl_nominal(ax, NOM, label=None)
    ax.set_ylim(-0.03, 1.03); ax.set_ylabel("per-task coverage")
    ax.set_title(title)
    return fin


def gap_vs_valid(ax, SA, title, null_lo=0.529, null_hi=0.581):
    ax.axhspan(null_lo, null_hi, color=T.TCL_NEUTRAL["light"], lw=0, zorder=0)
    ax.text(0.98, (null_lo + null_hi) / 2, "valid-predictor\nexpectation", transform=ax.get_yaxis_transform(),
            fontsize=5.0, ha="right", va="center", color=T.TCL_NEUTRAL["ink"])
    _fam_points(ax, SA, "CovGap", "ValidTaskFrac")
    ax.set_xlabel("mean coverage gap"); ax.set_ylabel("tasks at or above 0.90")
    ax.set_ylim(-0.04, 1.0); ax.set_title(title)


def null_test(ax, NU, title):
    NU = NU.copy(); NU["family"] = NU.method.map(FAM)
    groups = [NU[NU.family == f].vtf.values for f in FAM_ORDER]
    T.tcl_box_points(ax, groups, [ROLE[f] for f in FAM_ORDER], [f.lower() for f in FAM_ORDER], point_size=5)
    ax.axhspan(NU.p0.min(), NU.p0.max(), color=T.TCL_NEUTRAL["light"], lw=0, zorder=0)
    for i, f in enumerate(FAM_ORDER):
        d = NU[NU.family == f]
        ax.text(i, 0.97, f"{int(d.sig.sum())}/{len(d)}", ha="center", fontsize=5.4, color=T.tcl_color(ROLE[f], "text"))
    ax.set_ylim(-0.04, 1.04); ax.set_ylabel(None)
    for t in ax.get_xticklabels(): t.set_rotation(25); t.set_ha("right")
    ax.set_title(title)


def worst_vs_mean(ax, SA, title):
    _fam_points(ax, SA, "CovGap", "WorstTaskGap")
    lim = max(SA.CovGap.max(), SA.WorstTaskGap.max()) * 1.04
    ax.plot([0, lim], [0, lim], color=T.TCL_NEUTRAL["grey"], lw=0.8, ls=(0, (4, 2.5)), zorder=1)
    ax.set_xlabel("mean coverage gap"); ax.set_ylabel("worst-task gap")
    ax.set_title(title)


def rank_slopes(ax, SA, title):
    rows = []
    for (ds, bb), g in SA.groupby(["dataset", "backbone"]):
        g = g.assign(r_acc=g.ACC.rank(ascending=False), r_val=g.CovGap.rank(ascending=True))
        rows.append(g[["method", "family", "r_acc", "r_val"]])
    R = pd.concat(rows)
    rng = np.random.default_rng(0)
    for _, r in R.iterrows():
        j = rng.uniform(-0.12, 0.12)
        ax.plot([0, 1], [r.r_acc + j, r.r_val + j], color=T.tcl_color(ROLE[r.family], "fill"), lw=0.5, zorder=1)
    M = R.groupby(["method", "family"])[["r_acc", "r_val"]].mean().reset_index()
    for _, r in M.iterrows():
        ax.plot([0, 1], [r.r_acc, r.r_val], color=T.tcl_color(ROLE[r.family], "stroke"), lw=1.2, marker="o", ms=2.6, zorder=3)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["accuracy\nrank", "validity\nrank"])
    ax.set_xlim(-0.25, 1.25); ax.set_ylim(7.6, 0.4); ax.set_yticks([1, 4, 7])
    ax.set_title(title)


def rho_per_config(ax, RR, title):
    R = RR.rename(columns={"数据集": "dataset", "骨干": "backbone", "Spearman_ρ": "rho", "p_ρ": "p"}).copy()
    R["lab"] = [f"{DS_LABEL[d]} {BB_LABEL[b]}" for d, b in zip(R.dataset, R.backbone)]
    R = R.sort_values("rho")
    y = np.arange(len(R))
    sig = R.p < 0.05
    ax.scatter(R.rho[sig], y[sig.values], s=16, fc=T.tcl_color("oracle", "stroke"), ec=T.tcl_color("oracle", "stroke"), zorder=3, label="p < 0.05")
    ax.scatter(R.rho[~sig], y[~sig.values], s=16, fc="white", ec=T.tcl_color("oracle", "stroke"), lw=0.8, zorder=3, label="n.s.")
    ax.set_yticks(y); ax.set_yticklabels(R.lab, fontsize=5.0)
    ax.yaxis.tick_right(); ax.spines["right"].set_visible(True); ax.spines["left"].set_visible(False)
    ax.axvline(0, color=T.TCL_NEUTRAL["light"], lw=0.8)
    ax.set_xlim(0.4, 1.05); ax.set_xticks([0.5, 0.75, 1]); ax.set_xlabel("Spearman ρ, accuracy vs validity rank")
    ax.legend(loc="lower left", fontsize=5.0, handletextpad=0.1, borderaxespad=0.1)
    ax.set_title(title)


# ============================================================ figure 3
def gap_vs_driver(ax, MC, col, xlabel, title):
    d = MC.copy(); d["family"] = d.method.map(FAM)
    _fam_points(ax, d, col, "covgap")
    r = _fit_line(ax, d[col], d.covgap)
    ax.set_xlabel(xlabel); ax.set_ylabel("coverage gap")
    ax.set_title(title)
    return r


def partial_scale(ax, MC, title):
    import statsmodels.api as sm
    d = MC.dropna(subset=["covgap", "dAURC", "dECE"]).copy(); d["family"] = d.method.map(FAM)
    d["rx"] = sm.OLS(d.dECE, sm.add_constant(d.dAURC)).fit().resid
    d["ry"] = sm.OLS(d.covgap, sm.add_constant(d.dAURC)).fit().resid
    _fam_points(ax, d, "rx", "ry")
    m = sm.OLS(d.ry, sm.add_constant(d.rx)).fit()
    xs = np.linspace(d.rx.min(), d.rx.max(), 50)
    ax.plot(xs, m.params.iloc[0] + m.params.iloc[1] * xs, color=T.TCL_NEUTRAL["ink"], lw=0.8, ls=(0, (3, 2)))
    ax.axhline(0, color=T.TCL_NEUTRAL["light"], lw=0.7, zorder=0)
    lo = d.ry.min(); span = d.ry.max() - lo
    ax.set_ylim(lo - 0.32 * span, d.ry.max() + 0.06 * span)
    ax.text(0.97, 0.04, f"β = {m.params.iloc[1]:+.2f}, p = {m.pvalues.iloc[1]:.2f}", transform=ax.transAxes,
            fontsize=5.4, va="bottom", ha="right", color=T.TCL_NEUTRAL["ink"])
    ax.set_xlabel("ΔECE | ΔAURC"); ax.set_ylabel("gap | ΔAURC")
    ax.set_title(title)
    return m


def components(ax, MC, title):
    d = MC.copy(); d["family"] = d.method.map(FAM)
    w = 0.36
    for j, (col, lab, hatch) in enumerate([("dAURC", "ranking", None), ("dECE", "scale", "////")]):
        for i, f in enumerate(FAM_ORDER):
            v = d[d.family == f][col].values
            x = i + (j - 0.5) * w
            ax.bar(x, v.mean(), w * 0.92, color=T.tcl_color(ROLE[f], "fill") if hatch is None else "white",
                   ec=T.tcl_color(ROLE[f], "stroke"), lw=0.7, hatch=hatch, zorder=2,
                   label=lab if i == 0 else None)
            ax.scatter(x + np.random.default_rng(i * 7 + j).uniform(-0.06, 0.06, len(v)), v, s=3,
                       fc=T.tcl_color(ROLE[f], "stroke"), ec="none", zorder=3)
    ax.axhline(0, color=T.TCL_NEUTRAL["ink"], lw=0.6)
    ax.set_xticks(range(3)); ax.set_xticklabels(["proto", "replay", "grad"])
    for t, fam in zip(ax.get_xticklabels(), FAM_ORDER): t.set_color(T.tcl_color(ROLE[fam], "text"))
    ax.set_ylabel("earlier − current task")
    leg = ax.legend(loc="upper left", fontsize=5.0, handletextpad=0.3, borderaxespad=0.1)
    for h in leg.legend_handles:
        h.set_edgecolor(T.TCL_NEUTRAL["ink"]); h.set_facecolor(T.TCL_NEUTRAL["light"] if h.get_hatch() is None else "white")
    ax.set_title(title)


def metric_trajectory(ax, SR, col, ylabel, title):
    A = SR[(SR.protocol == "A_current") & (SR.task <= SR.step)]
    g = A.groupby(["family", "dataset", "backbone", "step"])[col].mean().reset_index()
    G = g.groupby(["family", "step"])[col].agg(["mean", "std", "size"]).reset_index()
    G["ci"] = 1.96 * G["std"] / np.sqrt(G["size"])
    ends = []
    for f in FAM_ORDER:
        d = G[G.family == f].sort_values("step"); x = d.step.values + 1
        T.tcl_curve(ax, x, d["mean"].values, (d["mean"] - d.ci).values, (d["mean"] + d.ci).values,
                    role=ROLE[f], label=f.lower(), ms=2.0, end_label=False)
        ends.append((x[-1], d["mean"].values[-1], f.lower(), ROLE[f]))
    ax.set_xlim(0.6, 16.2); ax.set_xticks([1, 4, 7, 10]); ax.set_xlabel("tasks learned")
    T.tcl_end_labels(ax, ends); ax.set_ylabel(ylabel); ax.set_title(title)


# ============================================================ figure 4
def protocol_boxes(ax, PCm, title):
    groups, roles = [], []
    for p in PROTO:
        groups.append(PCm[PCm.protocol == p].CovGap.values); roles.append(PROTO_ROLE[p])
    T.tcl_box_points(ax, groups, roles, [PROTO_SHORT[p] for p in PROTO], point_size=4)
    ax.set_ylabel("mean coverage gap"); ax.set_ylim(bottom=-0.005)
    ax.set_title(title)


def paired_protocols(ax, PCm, p_from, p_to, title, ylabel="mean coverage gap"):
    W = PCm.pivot_table(index=["dataset", "backbone", "method"], columns="protocol", values="CovGap").reset_index()
    for _, r in W.iterrows():
        ax.plot([0, 1], [r[p_from], r[p_to]], color=T.TCL_NEUTRAL["light"], lw=0.6, zorder=1)
    for x, p in [(0, p_from), (1, p_to)]:
        for mk, mm in [("Replay", "o"), ("iCaRL", "^")]:
            v = W[W.method == mk][p]
            ax.scatter(np.full(len(v), x), v, s=9, marker=mm, fc=T.tcl_color(PROTO_ROLE[p], "fill"),
                       ec=T.tcl_color(PROTO_ROLE[p], "stroke"), lw=0.6, zorder=3,
                       label=mk if x == 1 and p == p_to else None)
    better = int((W[p_to] < W[p_from]).sum())
    ax.text(0.5, 0.97, f"{better}/{len(W)} lower", transform=ax.transAxes, ha="center", va="top", fontsize=5.4,
            color=T.tcl_color(PROTO_ROLE[p_to], "text"))
    ax.set_xticks([0, 1]); ax.set_xticklabels([PROTO_SHORT[p_from], PROTO_SHORT[p_to]])
    ax.set_xlim(-0.35, 1.35); ax.set_ylabel(ylabel)
    ax.legend(loc="upper right", fontsize=5.0, handletextpad=0.1, borderaxespad=0.1, bbox_to_anchor=(1.02, 0.88))
    ax.set_title(title)
    return better, len(W)


def ksweep(ax, KS, col, ylabel, title, protocols=("A_current", "Cn_replayed", "C_rbcc", "B_pooled")):
    """Memory fixed at m per class (experience replay); k of the m slots
    reserved. One line per calibration set (gap) or a single line (accuracy, which does
    not depend on the calibration set)."""
    ends = []
    for p in protocols:
        g = KS[KS.protocol == p].groupby("k")[col].agg(["mean", ci95]).reset_index()
        if len(g) == 0:
            continue
        role = PROTO_ROLE[p]
        T.tcl_curve(ax, g.k.values, g["mean"].values, (g["mean"] - g.ci95).values, (g["mean"] + g.ci95).values,
                    role=role, label=PROTO_LABEL[p], ms=2.2, end_label=False)
        ends.append((g.k.values[-1], g["mean"].values[-1], PROTO_SHORT[p], role))
    ax.set_xlim(-0.4, 10.6); ax.set_xticks([0, 1, 3, 5, 7]); ax.set_xlabel("reserved slots of 10, k")
    if len(ends) > 1:
        T.tcl_end_labels(ax, ends, fontsize=5.0)
    else:
        ax.set_xlim(0.5, 7.5); ax.set_xticks([1, 3, 5, 7])
    ax.set_ylabel(ylabel); ax.set_title(title)


def k_bounds(ax, MK, title, kmin=9, kcons=19):
    ax.axvspan(0, kmin, color=T.TCL_NEUTRAL["rule"], lw=0, zorder=0)
    ax.axvspan(kmin, kcons, color=T.tcl_color("replay", "tint"), lw=0, zorder=0)
    # bands explained in the caption: grey = no per-class quantile (fallback), yellow = quantile is the sample max
    for s in SCHEME:
        g = MK[MK.scheme == s].groupby("k")["size"].agg(["mean", ci95]).reset_index(); sty = scheme_style(s)
        ax.fill_between(g.k, g["mean"] - g.ci95, g["mean"] + g.ci95, color=sty["c"], alpha=0.15, lw=0)
        ax.plot(g.k, g["mean"], color=sty["c"], lw=1.0, marker=sty["marker"], ms=2.4,
                mfc="white" if sty["hollow"] else sty["c"], mec=sty["c"], label=SCHEME_LABEL[s])
    ax.set_xlim(0, 41); ax.set_xticks([3, 9, 19, 30, 40])
    ax.set_xlabel("reserved slots per class, k"); ax.set_ylabel("mean set size")
    hs = [l for l in ax.get_lines() if l.get_label() in SCHEME_LABEL.values()]
    ax.legend(hs, [SCHEME_SHORT[s] for s in SCHEME], loc="upper right", fontsize=4.9,
              handletextpad=0.3,
              borderaxespad=0.1, frameon=True, framealpha=1, edgecolor="none")
    ax.set_ylim(1.2, 6.6); ax.set_yticks([2, 3, 4, 5])
    ax.set_title(title)


def balanced_violins(ax, MK, k, title):
    d = MK[MK.k == k]
    T.tcl_violin_points(ax, [d[d.scheme == s].mean_class.values for s in SCHEME],
                        ["oracle", "ours", "ours"], [SCHEME_SHORT[s] for s in SCHEME], point_size=5)
    T.tcl_nominal(ax, NOM, label=None)
    ax.set_ylim(0.885, 0.915); ax.set_ylabel("mean per-class coverage")
    ax.set_title(title)


def shift_sweep(ax, MF, title):
    T.tcl_regions(ax, [(NOM, 1.0, "ours", None), (0.0, NOM, "failing_default", None)])
    for s in SCHEME:
        g = MF[MF.scheme == s].groupby("replay_slots").agg(acc=("acc", "mean"), m=("cov_old100", "mean"),
                                                         ci=("cov_old100", ci95)).reset_index()
        sty = scheme_style(s)
        ax.errorbar(g.acc, g.m, yerr=g.ci, color=sty["c"], lw=1.0, marker=sty["marker"], ms=2.6,
                    mfc="white" if sty["hollow"] else sty["c"], mec=sty["c"], capsize=1.2, elinewidth=0.6,
                    label=SCHEME_LABEL[s])
    T.tcl_nominal(ax, NOM, label=None)
    ax.set_xlabel("final accuracy (replay 2 → 60 slots)"); ax.set_ylabel("coverage, old-class mix")
    ax.set_ylim(0.855, 0.925)
    ax.set_title(title)


def two_regimes_drop(ax, MF, EM, title, replay=20):
    out = {}
    for j, (lab, D) in enumerate([("frozen ViT", MF[MF.replay_slots == replay]), ("end-to-end", EM)]):
        for i, s in enumerate(SCHEME):
            v = (D[D.scheme == s].cov_uniform - D[D.scheme == s].cov_old100).values
            x = j + (i - 1) * 0.24; sty = scheme_style(s)
            ax.errorbar([x], [v.mean()], yerr=[ci95(v)], fmt=sty["marker"], ms=3.4, color=sty["c"],
                        mfc="white" if sty["hollow"] else sty["f"], mec=sty["c"], capsize=1.5, elinewidth=0.7,
                        label=SCHEME_LABEL[s] if j == 0 else None)
            ax.scatter(np.full(len(v), x) + 0.07, v, s=2.5, color=sty["c"], zorder=3)
            out[(lab, s)] = (v.mean(), ci95(v))
    ax.axhline(0, color=T.TCL_NEUTRAL["ink"], lw=0.6)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["frozen ViT", "end-to-end ResNet-18"]); ax.set_xlim(-0.45, 1.45)
    ax.set_ylabel("coverage lost to shift")
    ax.set_title(title)
    return out


# ============================================================ figure 5
def budget_curve(ax, EC, title, chosen=150):
    T.tcl_curve(ax, EC.epochs.values, EC.acc.values, role="oracle", tone="text", label=None, ms=2.4, end_label=False)
    ax.axvline(chosen, color=T.tcl_color("ours", "stroke"), lw=0.8, ls=(0, (3, 2)))
    ax.text(chosen, 0.04, f" {chosen} epochs", transform=ax.get_xaxis_transform(), fontsize=5.2,
            color=T.tcl_color("ours", "text"))
    ax.set_xlabel("epochs per task"); ax.set_ylabel("single-task accuracy"); ax.set_title(title)


def e2e_invisible(ax, ES, title):
    E = ES.copy(); E["m"] = E.method.str.replace("E2E-", "")
    order = [m for m in ["RBCC", "Replay", "LwF", "Seq-FT"] if m in set(E.m)]
    E = E.set_index("m").loc[order]
    T.tcl_dumbbell(ax, E.CovEarlier.values, E.CovCurrent.values, order, role_a="failing_default", role_b="oracle",
                   label_a="earlier tasks", label_b="just learned", size=16)
    ax.set_ylim(len(order) - 0.5, -0.5)
    T.tcl_nominal(ax, NOM, axis="x"); ax.set_xlim(0, 1.02); ax.set_xticks([0, 0.5, 0.9]); ax.set_xlabel("coverage")
    ax.legend(loc="lower left", fontsize=5.0, handletextpad=0.1, borderaxespad=0.1)
    ax.set_title(title)


def frozen_vs_e2e(ax, EV, title):
    E = EV.copy(); E["m"] = E.method.str.replace("E2E-", ""); ends = []
    for _, r in E.iterrows():
        role = "ours" if r.m == "RBCC" else ROLE.get(FAM.get(r.m, "Gradient"), "gradient_family")
        ax.plot([0, 1], [r.frozen_gap, r.e2e_gap], color=T.tcl_color(role, "stroke"), lw=1.0, marker="o", ms=2.8)
        ends.append((1, r.e2e_gap, r.m, role))
    ax.set_xticks([0, 1]); ax.set_xticklabels(["frozen", "end-to-end"]); ax.set_xlim(-0.2, 1.75)
    ax.set_ylim(0.1, 1.0); T.tcl_end_labels(ax, ends, fontsize=5.0)
    ax.set_ylabel("earlier-task coverage gap"); ax.set_title(title)


def e2e_protocols(ax, EP, title):
    E = EP.set_index("protocol").loc[PROTO]
    for i, p in enumerate(PROTO):
        role = PROTO_ROLE[p]
        ax.errorbar([i], [E.loc[p, "mean"]], yerr=[E.loc[p, "ci95"]], fmt="o", ms=3.6, color=T.tcl_color(role, "stroke"),
                    mfc=T.tcl_color(role, "fill"), capsize=1.5, elinewidth=0.7)
    T.tcl_nominal(ax, NOM, label=None)
    ax.set_xticks(range(4)); ax.set_xticklabels(["cur", "rep", "res", "all"], fontsize=4.8)
    for t, p in zip(ax.get_xticklabels(), PROTO): t.set_color(T.tcl_color(PROTO_ROLE[p], "text"))
    ax.set_xlim(-0.5, 3.5); ax.set_ylabel("final coverage"); ax.set_title(title)


def backbone_pairs(ax, SA, title):
    W = SA.pivot_table(index=["dataset", "method", "family"], columns="backbone", values="FinalCov").reset_index()
    _fam_points(ax, W.rename(columns={"vit_b16_in21k": "x", "dinov2_base": "y"}), "x", "y")
    ax.plot([0, 1], [0, 1], color=T.TCL_NEUTRAL["grey"], lw=0.8, ls=(0, (4, 2.5)))
    T.tcl_nominal(ax, NOM, label=None); T.tcl_nominal(ax, NOM, axis="x")
    ax.set_xlim(0.1, 1.0); ax.set_ylim(0.1, 1.0); ax.set_xlabel("final coverage, ViT"); ax.set_ylabel("final coverage, DINOv2")
    r = st.spearmanr(W.vit_b16_in21k, W.dinov2_base)
    ax.text(0.96, 0.05, f"ρ = {r.statistic:.2f}", transform=ax.transAxes, fontsize=5.4, ha="right")
    ax.set_title(title)
    return r


def score_pairs(ax, SC, col, xlabel, title, logx=False, nominal=False):
    g = SC.groupby(["method", "score"])[col].mean().unstack()
    m = METHOD_ORDER; g = g.loc[m]
    for i, mm in enumerate(m):
        role = ROLE[FAM[mm]]
        ax.plot([g.loc[mm, "lac"], g.loc[mm, "aps"]], [i, i], color=T.TCL_NEUTRAL["light"], lw=2, zorder=1)
        ax.scatter(g.loc[mm, "lac"], i, s=14, marker="o", fc="white", ec=T.tcl_color(role, "stroke"), lw=0.8, zorder=3,
                   label="LAC" if i == 0 else None)
        ax.scatter(g.loc[mm, "aps"], i, s=14, marker="o", fc=T.tcl_color(role, "stroke"), ec=T.tcl_color(role, "stroke"),
                   lw=0.8, zorder=3, label="APS" if i == 0 else None)
    ax.set_yticks(range(len(m))); ax.set_yticklabels(m); ax.set_ylim(len(m) - 0.5, -0.5)
    for t in ax.get_yticklabels(): t.set_color(T.tcl_color(ROLE[FAM[t.get_text()]], "text"))
    if nominal: T.tcl_nominal(ax, NOM, axis="x")
    if logx: ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    if logx: ax.set_xlim(right=ax.get_xlim()[1] * 3)
    leg = ax.legend(loc="upper left" if not logx else "lower right", fontsize=5.0, handletextpad=0.1, borderaxespad=0.1)
    for h in leg.legend_handles: h.set_edgecolor(T.TCL_NEUTRAL["ink"])
    leg.legend_handles[1].set_facecolor(T.TCL_NEUTRAL["ink"])
    ax.set_title(title)
    return g


def dataset_boxes(ax, SA, title):
    ds = ["cifar100", "cub200", "imagenet_r", "medmix26"]; w = 0.24
    for j, f in enumerate(FAM_ORDER):
        groups = [SA[(SA.dataset == d) & (SA.family == f)].CovGap.values for d in ds]
        pos = np.arange(4) + (j - 1) * w
        bp = ax.boxplot(groups, positions=pos, widths=w * 0.85, patch_artist=True, showfliers=False,
                        medianprops=dict(color=T.tcl_color(ROLE[f], "text"), lw=0.9))
        for b in bp["boxes"]: b.set(fc=T.tcl_color(ROLE[f], "fill"), ec=T.tcl_color(ROLE[f], "stroke"), lw=0.6)
        for part in ("whiskers", "caps"):
            for l in bp[part]: l.set(color=T.tcl_color(ROLE[f], "stroke"), lw=0.6)
        for p_, g in zip(pos, groups):
            ax.scatter(np.full(len(g), p_), g, s=3, color=T.tcl_color(ROLE[f], "stroke"), zorder=3)
    ax.set_xticks(range(4)); ax.set_xticklabels(["C100", "CUB", "IN-R", "Med"], fontsize=5.0)
    ax.set_ylabel("mean coverage gap"); ax.set_title(title)
