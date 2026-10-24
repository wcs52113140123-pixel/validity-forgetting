import zlib
"""CNS-style panel builders, second design: richer chart types.

Each builder draws one lettered panel (possibly with child small-multiples or marginal
axes inside its own cell), sets a short descriptive title, and returns the numbers the
caption quotes. Encodings come from tcl_kernel only; family colour is fixed paper-wide.
"""
import numpy as np
import pandas as pd
from scipy import stats as st
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Rectangle, Wedge
import matplotlib.colors as mcolors
import tcl_kernel as T
from figs_tcl import (FAM, ROLE, FAM_ORDER, METHOD_ORDER, NOM, PROTO, PROTO_ROLE, PROTO_SHORT, SCHEME,
                      SCHEME_SHORT, scheme_style, ci95, DS_LABEL, BB_LABEL)

INK = T.TCL_NEUTRAL["ink"]
GREY = T.TCL_NEUTRAL["grey"]
LIGHT = T.TCL_NEUTRAL["light"]


def fc(f, tone="fill"):
    return T.tcl_color(ROLE[f], tone)


def _kde(v, grid, bw=None):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if len(v) < 3 or np.std(v) < 1e-9:
        d = np.zeros_like(grid)
        if len(v):
            d[np.argmin(np.abs(grid - v.mean()))] = 1.0
        return d
    return st.gaussian_kde(v, bw_method=bw)(grid)


def _boot(fn, n, B=3000, seed=0):
    rng = np.random.default_rng(seed)
    return np.array([fn(rng.integers(0, n, n)) for _ in range(B)])


def _shrink(ax, fx=0.80, fy=0.80):
    """Shrink the main axes inside its own grid cell so marginal axes fit in the cell
    (keeps the panel's footprint, and therefore the letter grid, unchanged)."""
    p = ax.get_position()
    ax.set_position([p.x0, p.y0, p.width * fx, p.height * fy])


# ================================================================= figure 1
def divergence_multiples(ax, TR, title):
    """Small multiples, one per family: accuracy and coverage on the same axis with the
    space between them filled. The filled area is the divergence the paper is about."""
    ax.axis("off")
    ax.set_title(title)
    w, gap = 0.31, 0.035
    out = {}
    for i, f in enumerate(FAM_ORDER):
        a = ax.inset_axes([i * (w + gap), 0.0, w, 1.0])
        d = TR[TR.family == f].sort_values("step")
        x = d.step.values + 1
        a.fill_between(x, d["cov"], d["acc"], where=d["acc"] >= d["cov"], color=T.tcl_color("failing_default", "tint"),
                       lw=0, zorder=1, interpolate=True)
        a.fill_between(x, d["cov"], d["acc"], where=d["acc"] < d["cov"], color=T.tcl_color("ours", "tint"),
                       lw=0, zorder=1, interpolate=True)
        a.plot(x, d["acc"], color=fc(f, "stroke"), lw=1.2, ls=(0, (3, 1.5)), zorder=3)
        a.plot(x, d["cov"], color=fc(f, "stroke"), lw=1.4, zorder=3)
        a.axhline(NOM, color=GREY, lw=0.7, ls=(0, (4, 2.5)), zorder=2)
        ya, yc = float(d["acc"].values[-1]), float(d["cov"].values[-1])
        pos = ax.get_position(); h_pt = pos.height * ax.figure.get_size_inches()[1] * 72
        sep = 0.75 * 8.4 / h_pt            # one 7 pt line, in data units (ylim span 0.75)
        if abs(ya - yc) < sep:             # push the two end labels apart symmetrically
            mid = (ya + yc) / 2; hi, lo = mid + sep / 2, mid - sep / 2
            ya, yc = (hi, lo) if ya >= yc else (lo, hi)
        a.text(10.3, ya, "accuracy", fontsize=5.0, va="center", color=fc(f, "text"))
        a.text(10.3, yc, "coverage", fontsize=5.0, va="center", color=fc(f, "text"))
        a.set_xlim(0.6, 14.6); a.set_ylim(0.25, 1.0); a.set_xticks([1, 5, 10])
        a.text(0.03, 0.04, f.lower(), transform=a.transAxes, fontsize=6.0, color=fc(f, "text"), fontweight="bold")
        if i:
            a.set_yticklabels([])
        else:
            a.set_ylabel("seen classes")
        a.set_xlabel("tasks learned" if i == 1 else "")
        out[f] = float(d["acc"].values[-1] - d["cov"].values[-1])
    return out


def bubble_matrix(ax, MAT, title, nom=NOM):
    """Per-task coverage of the final model as a dot matrix: colour = coverage (diverging
    around the promise), area = shortfall below it. Methods x tasks."""
    cmap = T.tcl_cmap("div_coverage")
    norm = mcolors.TwoSlopeNorm(vmin=0.0, vcenter=nom, vmax=1.0)
    M = MAT.values
    yy, xx = np.mgrid[0:M.shape[0], 0:M.shape[1]]
    short = np.clip(nom - M, 0, None)
    size = 6 + 120 * short / nom
    ax.scatter(xx.ravel(), yy.ravel(), s=size.ravel(), c=M.ravel(), cmap=cmap, norm=norm,
               ec=INK, lw=0.25, zorder=3)
    ax.set_xticks(range(M.shape[1])); ax.set_xticklabels([f"T{c + 1}" for c in MAT.columns])
    ax.set_yticks(range(M.shape[0])); ax.set_yticklabels(list(MAT.index))
    for t in ax.get_yticklabels():
        t.set_color(fc(FAM[t.get_text()], "text"))
    ax.set_xlim(-0.6, M.shape[1] - 0.4); ax.set_ylim(M.shape[0] - 0.5, -0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    cax = ax.inset_axes([0.0, -0.30, 0.5, 0.045])
    cb = ax.figure.colorbar(ax.collections[0], cax=cax, orientation="horizontal")
    cb.set_ticks([0, nom, 1.0]); cb.ax.tick_params(labelsize=5, length=1.5)
    cb.outline.set_linewidth(0.4)
    ax.text(0.56, -0.28, "dot area = shortfall below 0.90", transform=ax.transAxes, fontsize=5.0, va="center", color=INK)
    ax.set_title(title)
    return int((M[:, :3] == 0).sum())


def paired_swarm(ax, MC, title, seed=0):
    """Each of the 56 method-configuration cells measured twice: on the task just learned
    and on earlier tasks. Beeswarm + faint pairing lines."""
    d = MC.copy(); d["family"] = d.method.map(FAM)
    rng = np.random.default_rng(seed)
    jit = {}
    for _, r in d.iterrows():
        jit[(r.dataset, r.backbone, r.method)] = rng.uniform(-0.16, 0.16)
    for _, r in d.iterrows():
        j = jit[(r.dataset, r.backbone, r.method)]
        ax.plot([0 + j, 1 + j], [r.cov_cur, r.cov_old], color=fc(r.family, "fill"), lw=0.4, alpha=0.8, zorder=1)
    for x, col in [(0, "cov_cur"), (1, "cov_old")]:
        for f in FAM_ORDER:
            s = d[d.family == f]
            xs = [x + jit[(r.dataset, r.backbone, r.method)] for _, r in s.iterrows()]
            ax.scatter(xs, s[col], s=7, fc=fc(f), ec=fc(f, "stroke"), lw=0.4, zorder=3,
                       label=f.lower() if x == 0 else None)
    T.tcl_nominal(ax, NOM, label=None)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["just learned", "earlier"])
    ax.set_xlim(-0.45, 1.45); ax.set_ylim(-0.02, 1.02); ax.set_ylabel("coverage")
    ax.set_title(title)
    return float(d.cov_cur.min()), float(d.cov_old.min())


def joint_size(ax, SA, title):
    """Set size against coverage with marginal densities per family (joint plot)."""
    for f in FAM_ORDER:
        s = SA[SA.family == f]
        ax.scatter(s.FinalCov, s.FinalSize, s=10, fc=fc(f), ec=fc(f, "stroke"), lw=0.45, zorder=3, label=f.lower())
    ax.set_yscale("log")
    T.tcl_nominal(ax, NOM, axis="x")
    ax.set_xlim(0.1, 1.0)
    ax.set_xlabel("final coverage"); ax.set_ylabel("set size (labels)")
    _shrink(ax, 0.80, 0.80)
    top = ax.inset_axes([0, 1.04, 1, 0.22], sharex=ax)
    right = ax.inset_axes([1.04, 0, 0.22, 1], sharey=ax)
    gx = np.linspace(0.05, 1.05, 200)
    gy = np.logspace(np.log10(SA.FinalSize.min() * 0.7), np.log10(SA.FinalSize.max() * 1.4), 200)
    for f in FAM_ORDER:
        s = SA[SA.family == f]
        kx = _kde(s.FinalCov, gx, 0.6)
        top.fill_between(gx, kx, color=fc(f), alpha=0.55, lw=0); top.plot(gx, kx, color=fc(f, "stroke"), lw=0.7)
        ky = _kde(np.log10(s.FinalSize), np.log10(gy), 0.45)
        right.fill_betweenx(gy, ky, color=fc(f), alpha=0.55, lw=0); right.plot(ky, gy, color=fc(f, "stroke"), lw=0.7)
    for m in (top, right):
        m.axis("off")
    ax.legend(loc="upper left", fontsize=5.0, handletextpad=0.1, borderaxespad=0.15)
    if ax.get_yscale() == "log":
        lo_, hi_ = ax.get_ylim(); ax.set_ylim(lo_, hi_ * 2.5)   # headroom: keep points clear of the 7 pt legend
    ax.set_title(title)
    lo = SA[SA.FinalCov < 0.5].FinalSize.median(); hi = SA[SA.FinalCov >= 0.85].FinalSize.median()
    return float(lo), float(hi)


def waffle(ax, SA, title):
    """56 method-configuration cells as a waffle: colour = family, filled = below the
    promised 0.90, open = meets it."""
    d = SA.copy()
    d["fo"] = d.family.map({f: i for i, f in enumerate(FAM_ORDER)})
    d = d.sort_values(["fo", "FinalCov"], ascending=[True, False]).reset_index(drop=True)
    ncol = 8
    for i, r in d.iterrows():
        x, y = i % ncol, i // ncol
        below = r.FinalCov < NOM
        ax.add_patch(Rectangle((x + 0.12, -y - 0.88), 0.76, 0.76, fc=fc(r.family) if below else "white",
                               ec=fc(r.family, "stroke"), lw=0.6))
    n = int((d.FinalCov < NOM).sum())
    ax.set_xlim(-0.1, ncol + 0.1); ax.set_ylim(-(len(d) // ncol) - 0.1, 0.1)
    ax.set_aspect("equal"); ax.axis("off")
    ax.text(ncol + 0.5, -1.0, f"{n}", fontsize=15, fontweight="bold", color=T.tcl_color("failing_default", "text"),
            va="center")
    ax.text(ncol + 0.5, -2.2, f"of {len(d)} configurations\nbelow 0.90", fontsize=5.6, color=INK, va="top")
    ax.text(ncol + 0.5, -4.6, "filled = below\nopen = meets", fontsize=5.0, color=GREY, va="top")
    ax.set_xlim(-0.1, ncol + 4.2)
    ax.set_title(title)
    return n


# ================================================================= figure 2
def loss_forest(ax, DPx, title):
    """All 56 cells sorted by coverage loss: open marker = accuracy loss, filled =
    coverage loss, segment = the excess. A forest of dumbbells."""
    d = DPx.copy().sort_values("cov_loss").reset_index(drop=True)
    y = np.arange(len(d))
    for i, r in d.iterrows():
        c = fc(r.family, "stroke")
        ax.plot([r.acc_loss, r.cov_loss], [i, i], color=fc(r.family), lw=1.0, zorder=1, solid_capstyle="round")
        ax.scatter(r.acc_loss, i, s=7, fc="white", ec=c, lw=0.6, zorder=3)
        ax.scatter(r.cov_loss, i, s=8, fc=c, ec=c, lw=0.4, zorder=3)
    ax.axvline(0, color=LIGHT, lw=0.7, zorder=0)
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.set_ylim(-1, len(d)); ax.set_xlim(-0.04, max(d.cov_loss.max(), d.acc_loss.max()) * 1.05)
    ax.set_xlabel("loss vs best method")
    ax.scatter([], [], s=7, fc="white", ec=INK, lw=0.6, label="accuracy")
    ax.scatter([], [], s=8, fc=INK, ec=INK, label="coverage")
    ax.legend(loc="lower right", fontsize=5.0, handletextpad=0.1, borderaxespad=0.1)
    for f in FAM_ORDER:
        yy = y[d.family.values == f]
        if len(yy):
            ax.text(-0.035, yy.mean(), f.lower(), rotation=90, ha="right", va="center", fontsize=5.4,
                    color=fc(f, "text"), transform=ax.get_yaxis_transform() if False else ax.transData)
    ax.set_title(title)
    return int((d.cov_loss > d.acc_loss).sum()), len(d)


def ratio_raincloud(ax, DPx, title, methods=("EWC", "LwF", "Seq-FT", "Replay"), B=4000):
    """Bootstrap distribution of the ratio of mean losses per method (half-violin),
    with the point estimate and 95% interval beneath."""
    out = {}
    for i, m in enumerate(methods):
        d = DPx[DPx.method == m]
        a, c = d.acc_loss.values, d.cov_loss.values
        bs = _boot(lambda k: c[k].mean() / max(a[k].mean(), 1e-6), len(a), B, seed=i)
        bs = bs[np.isfinite(bs) & (bs < 50)]
        lb = np.log(bs)
        g = np.linspace(np.log(0.8), np.log(14), 300)
        k = _kde(lb, g, 0.3); k = 0.38 * k / k.max()
        f = FAM[m]
        ax.fill_between(np.exp(g), i - k, i, color=fc(f), alpha=0.85, lw=0)
        ax.plot(np.exp(g), i - k, color=fc(f, "stroke"), lw=0.6)
        rom = c.mean() / a.mean(); lo, hi = np.percentile(bs, [2.5, 97.5])
        ax.plot([lo, hi], [i + 0.12, i + 0.12], color=fc(f, "stroke"), lw=1.0)
        ax.scatter(rom, i + 0.12, s=14, fc="white", ec=fc(f, "stroke"), lw=0.9, zorder=3)
        ax.text(hi * 1.07, i + 0.12, f"{rom:.2f}", va="center", fontsize=5.2, color=fc(f, "text"))
        out[m] = (rom, lo, hi)
    ax.axvline(1, color=GREY, lw=0.7, ls=(0, (4, 2.5)))
    ax.set_xscale("log"); ax.set_xticks([1, 2, 4, 8]); ax.set_xticklabels(["1", "2", "4", "8"])
    ax.minorticks_off(); ax.set_xlim(0.8, 16)
    ax.set_yticks(range(len(methods))); ax.set_yticklabels(methods); ax.set_ylim(-0.6, len(methods) - 0.6)
    for t in ax.get_yticklabels():
        t.set_color(fc(FAM[t.get_text()], "text"))
    ax.set_xlabel("coverage loss / accuracy loss")
    ax.set_title(title)
    return out


def ridgeline(ax, SR, title):
    """Ridgeline of per-task coverage of the final model, one ridge per method."""
    A = SR[SR.protocol == "A_current"]
    fin = A[A.step == A.step.max()].groupby(["dataset", "backbone", "method", "task"])["cov"].mean().reset_index()
    g = np.linspace(-0.05, 1.05, 300)
    order = METHOD_ORDER
    for i, m in enumerate(order):
        v = fin[fin.method == m]["cov"].values
        k = _kde(v, g, 0.12); k = 1.35 * k / k.max()
        base = len(order) - 1 - i
        f = FAM[m]
        ax.fill_between(g, base, base + k, color=fc(f), alpha=0.9, lw=0, zorder=10 - i)
        ax.plot(g, base + k, color=fc(f, "stroke"), lw=0.6, zorder=10 - i)
        ax.plot([np.median(v)] * 2, [base, base + 0.3], color=fc(f, "text"), lw=1.0, zorder=20)
        ax.text(-0.07, base + 0.15, m, ha="right", va="bottom", fontsize=5.2, color=fc(f, "text"))
    ax.axvline(NOM, color=GREY, lw=0.7, ls=(0, (4, 2.5)), zorder=30)
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.1, len(order) + 0.6)
    ax.set_xlabel("per-task coverage, final model")
    ax.set_title(title)


def null_pmf(ax, NU, title, n=10):
    """Observed number of tasks meeting 0.90 (of 10) per family as stacked bars against
    the binomial distribution a valid predictor would produce."""
    d = NU.copy()
    ks = np.arange(n + 1)
    bottom = np.zeros(n + 1)
    tot = len(d)
    for f in FAM_ORDER:
        c = np.array([((d.family == f) & (d.k == k)).sum() for k in ks]) / tot
        ax.bar(ks, c, 0.78, bottom=bottom, color=fc(f), ec=fc(f, "stroke"), lw=0.5, label=f.lower(), zorder=2)
        bottom += c
    p0 = d.p0.mean()
    pm = st.binom.pmf(ks, n, p0)
    ax.plot(ks, pm, color=INK, lw=0.9, marker="o", ms=2.2, mfc="white", zorder=4, label="valid predictor")
    ax.set_xticks([0, 2, 4, 6, 8, 10]); ax.set_xlabel("tasks meeting 0.90 (of 10)")
    ax.set_ylabel("fraction of configurations")
    ax.legend(loc="upper right", fontsize=4.9, handletextpad=0.3, borderaxespad=0.1, handlelength=1.2)
    ax.set_ylim(0, 0.5)
    ax.set_title(title)
    return {f: int(d[d.family == f].sig.sum()) for f in FAM_ORDER}


def ecdf(ax, SR, title):
    """ECDF of per-task coverage, one curve per family, with the mean marked: the mean
    sits well above the lower tail it averages away."""
    A = SR[SR.protocol == "A_current"]
    fin = A[A.step == A.step.max()].groupby(["dataset", "backbone", "method", "family", "task"])["cov"].mean().reset_index()
    out = {}
    for f in FAM_ORDER:
        v = np.sort(fin[fin.family == f]["cov"].values)
        y = np.arange(1, len(v) + 1) / len(v)
        ax.step(v, y, where="post", color=fc(f, "stroke"), lw=1.2, label=f.lower())
        mu = v.mean()
        ax.scatter([mu], [np.searchsorted(v, mu) / len(v)], s=16, marker="D", fc=fc(f), ec=fc(f, "stroke"), lw=0.6,
                   zorder=4)
        out[f] = (float(mu), float((v < NOM).mean()))
    ax.axvline(NOM, color=GREY, lw=0.7, ls=(0, (4, 2.5)))
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(0, 1.02)
    ax.set_xlabel("per-task coverage"); ax.set_ylabel("cumulative fraction")
    ax.plot([0.045], [0.925], marker="D", ms=3.2, mfc="white", mec=INK, mew=0.7, ls="none", transform=ax.transAxes, clip_on=False)
    ax.text(0.085, 0.95, "mean", transform=ax.transAxes, fontsize=5.0, va="top", color=INK)
    ax.set_title(title)
    return out


def _ribbon(ax, y0a, y0b, y1a, y1b, x0, x1, color, alpha=0.55):
    xm = (x0 + x1) / 2
    verts = [(x0, y0a), (xm, y0a), (xm, y1a), (x1, y1a), (x1, y1b), (xm, y1b), (xm, y0b), (x0, y0b), (x0, y0a)]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
             Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts, codes), fc=color, ec="none", alpha=alpha, zorder=1))


def rank_alluvial(ax, SA, title):
    """Alluvial (Sankey) flow from accuracy rank to validity rank, 8 configurations x 7
    methods; ribbon width = number of cells making that transition."""
    rows = []
    for _, g in SA.groupby(["dataset", "backbone"]):
        g = g.assign(ra=g.ACC.rank(ascending=False, method="first").astype(int),
                     rv=g.CovGap.rank(ascending=True, method="first").astype(int))
        rows.append(g[["method", "family", "ra", "rv"]])
    R = pd.concat(rows)
    n_cfg = R.groupby("ra").size().max()
    h = 1.0 / n_cfg * 0.86
    offL = {r: 0 for r in range(1, 8)}; offR = {r: 0 for r in range(1, 8)}
    for f in FAM_ORDER:
        for (ra, rv), c in R[R.family == f].groupby(["ra", "rv"]).size().items():
            y0 = ra - 0.43 + offL[ra] * h; y1 = rv - 0.43 + offR[rv] * h
            _ribbon(ax, y0, y0 + c * h, y1, y1 + c * h, 0.08, 0.92, fc(f))
            offL[ra] += c; offR[rv] += c
    for x in (0, 1):
        for r in range(1, 8):
            ax.add_patch(Rectangle((x - 0.08 if x == 1 else 0, r - 0.43), 0.08, 0.86, fc=INK, ec="none", zorder=3))
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(7.6, 0.4)
    ax.set_yticks(range(1, 8)); ax.set_xticks([0.04, 0.96]); ax.set_xticklabels(["accuracy\nrank", "validity\nrank"])
    ax.tick_params(axis="x", length=0)
    for s in ("bottom",):
        ax.spines[s].set_visible(False)
    ax.set_title(title)
    return float((R.ra == R.rv).mean())


def rho_lollipop(ax, SA, RR, title, B=3000):
    """Per-configuration Spearman rho between accuracy and validity ranks, bootstrap 95%
    interval over methods; filled = p < 0.05."""
    Rr = RR.rename(columns={"数据集": "dataset", "骨干": "backbone", "Spearman_ρ": "rho", "p_ρ": "p",
                            "spearman_rho": "rho", "p_rho": "p"})
    rows = []
    for (ds, bb), g in SA.groupby(["dataset", "backbone"]):
        a, v = g.ACC.values, -g.CovGap.values
        bs = _boot(lambda k: st.spearmanr(a[k], v[k]).statistic if len(set(k)) > 2 else np.nan, len(a), B,
                   seed=zlib.crc32(f"{ds}|{bb}".encode()) % 1000)   # deterministic across processes (str hash is salted)
        r_ = Rr[(Rr.dataset == ds) & (Rr.backbone == bb)].iloc[0]
        SH = {"cifar100": "C100", "cub200": "CUB", "imagenet_r": "IN-R", "medmix26": "Med"}
        rows.append(dict(lab=f"{SH[ds]} {BB_LABEL[bb]}", rho=r_.rho, p=r_.p,
                         lo=np.nanpercentile(bs, 2.5), hi=np.nanpercentile(bs, 97.5)))
    D = pd.DataFrame(rows).sort_values("rho").reset_index(drop=True)
    c = T.tcl_color("oracle", "stroke")
    for i, r in D.iterrows():
        ax.plot([0, r.rho], [i, i], color=LIGHT, lw=1.6, zorder=1, solid_capstyle="round")
        ax.plot([r.lo, r.hi], [i, i], color=c, lw=0.7, zorder=2)
        ax.scatter(r.rho, i, s=16, fc=c if r.p < 0.05 else "white", ec=c, lw=0.8, zorder=3)
    ax.set_yticks(range(len(D))); ax.set_yticklabels(D.lab, fontsize=5.0)
    ax.set_xlim(0, 1.05); ax.set_xticks([0, 0.5, 1]); ax.set_xlabel("Spearman ρ")
    ax.scatter([], [], s=16, fc=c, ec=c, label="p < 0.05"); ax.scatter([], [], s=16, fc="white", ec=c, label="n.s.")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, fontsize=5.0, handletextpad=0.1,
              borderaxespad=0.1, columnspacing=0.6)
    ax.set_title(title)
    return D


# ================================================================= figure 3
def joint_mechanism(ax, MC, title):
    """Hero: coverage gap against ranking degradation, regression with 95% band, marginal
    densities per family on both axes."""
    d = MC.copy(); d["family"] = d.method.map(FAM)
    x, y = d.dAURC.values, d.covgap.values
    r = st.linregress(x, y)
    xs = np.linspace(x.min() - 0.02, x.max() + 0.02, 100)
    bs = _boot(lambda k: np.polyfit(x[k], y[k], 1), len(x), 2000, seed=3)
    band = np.array([b[0] * xs + b[1] for b in bs])
    ax.fill_between(xs, *np.percentile(band, [2.5, 97.5], axis=0), color=LIGHT, lw=0, zorder=1)
    ax.plot(xs, r.intercept + r.slope * xs, color=INK, lw=0.9, zorder=2)
    for f in FAM_ORDER:
        s = d[d.family == f]
        ax.scatter(s.dAURC, s.covgap, s=14, fc=fc(f), ec=fc(f, "stroke"), lw=0.5, zorder=3, label=f.lower())
    ax.text(0.04, 0.95, f"R² = {r.rvalue ** 2:.2f}\nslope = {r.slope:.2f}", transform=ax.transAxes, fontsize=5.6,
            va="top", color=INK)
    ax.set_xlabel("ΔAURC, earlier − current task (ranking)"); ax.set_ylabel("coverage gap")
    ax.legend(loc="lower right", fontsize=5.2, handletextpad=0.1, borderaxespad=0.15)
    _shrink(ax, 0.84, 0.84)
    top = ax.inset_axes([0, 1.04, 1, 0.17], sharex=ax)
    right = ax.inset_axes([1.04, 0, 0.17, 1], sharey=ax)
    gx = np.linspace(x.min() - 0.1, x.max() + 0.1, 200); gy = np.linspace(y.min() - 0.1, y.max() + 0.1, 200)
    for f in FAM_ORDER:
        s = d[d.family == f]
        kx = _kde(s.dAURC, gx, 0.4); ky = _kde(s.covgap, gy, 0.4)
        top.fill_between(gx, kx, color=fc(f), alpha=0.55, lw=0); top.plot(gx, kx, color=fc(f, "stroke"), lw=0.6)
        right.fill_betweenx(gy, ky, color=fc(f), alpha=0.55, lw=0); right.plot(ky, gy, color=fc(f, "stroke"), lw=0.6)
    for m in (top, right):
        m.axis("off")
    ax.set_title(title)
    return r


def commonality(ax, MC, title):
    """Variance in the coverage gap partitioned into what ranking explains alone, what
    scale explains alone, and what they share (commonality analysis)."""
    import statsmodels.api as sm
    d = MC.copy(); d["family"] = d.method.map(FAM)
    groups = [("all", d)] + [(f.lower(), d[d.family == f]) for f in FAM_ORDER]
    out = {}
    for i, (lab, g) in enumerate(groups):
        r2 = lambda cols: sm.OLS(g.covgap, sm.add_constant(g[cols])).fit().rsquared
        both, ra, re = r2(["dAURC", "dECE"]), r2(["dAURC"]), r2(["dECE"])
        ua, ue = both - re, both - ra; sh = both - ua - ue
        parts = [(ua, T.tcl_color("ours", "stroke"), "ranking only"), (sh, T.tcl_color("ours", "fill"), "shared"),
                 (ue, T.tcl_color("replay", "fill"), "scale only"), (1 - both, "white", "unexplained")]
        left = 0
        for v, col, nm in parts:
            v = max(v, 0)
            ax.barh(i, v, left=left, height=0.62, color=col, ec=INK if col == "white" else "none", lw=0.4,
                    label=nm if i == 0 else None)
            left += v
        out[lab] = dict(both=both, ranking_only=ua, scale_only=ue, shared=sh)
    ax.set_yticks(range(len(groups))); ax.set_yticklabels([g[0] for g in groups])
    for t, (lab, _) in zip(ax.get_yticklabels(), groups):
        if lab != "all":
            t.set_color(fc(lab.capitalize(), "text"))
    ax.set_ylim(len(groups) - 0.5, -0.5); ax.set_xlim(0, 1); ax.set_xlabel("share of variance in gap")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2, fontsize=4.9, handlelength=1.0,
              columnspacing=0.8, handletextpad=0.3)
    ax.set_title(title)
    return out


def beta_densities(ax, MC, title, B=3000):
    """Bootstrap distributions of the standardised coefficients of ranking and scale in
    the joint regression; scale's straddles zero."""
    import statsmodels.api as sm
    d = MC.dropna(subset=["covgap", "dAURC", "dECE"])
    Z = (d[["dAURC", "dECE"]] - d[["dAURC", "dECE"]].mean()) / d[["dAURC", "dECE"]].std()
    y = (d.covgap - d.covgap.mean()) / d.covgap.std()
    X = sm.add_constant(Z).values; yv = y.values
    bs = _boot(lambda k: np.linalg.lstsq(X[k], yv[k], rcond=None)[0][1:], len(yv), B, seed=5)
    g = np.linspace(-1.0, 2.0, 300)
    out = {}
    for j, (nm, role) in enumerate([("ranking (ΔAURC)", "ours"), ("scale (ΔECE)", "replay")]):
        k = _kde(bs[:, j], g, 0.25)
        ax.fill_between(g, k, color=T.tcl_color(role, "fill"), alpha=0.8, lw=0)
        ax.plot(g, k, color=T.tcl_color(role, "stroke"), lw=0.8)
        lo, hi = np.percentile(bs[:, j], [2.5, 97.5])
        ax.text(np.median(bs[:, j]) + (0.12 if j == 0 else -0.12), k.max() * 1.04, nm, ha="left" if j == 0 else "right",
                fontsize=5.2, color=T.tcl_color(role, "text"))
        out[nm] = (float(np.median(bs[:, j])), float(lo), float(hi))
    ax.axvline(0, color=INK, lw=0.7)
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.set_xlabel("standardised coefficient (bootstrap)")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    ax.set_title(title)
    return out


def strip_heat(ax, SR, title):
    """Heat strips: AURC and ECE of seen classes along the stream, one strip per family
    and metric, sequential colour."""
    A = SR[(SR.protocol == "A_current") & (SR.task <= SR.step)]
    rows, labs = [], []
    for met in ("aurc", "ece"):
        g = A.groupby(["family", "dataset", "backbone", "step"])[met].mean().groupby(["family", "step"]).mean()
        for f in FAM_ORDER:
            rows.append(g.loc[f].values); labs.append((met.upper(), f))
    M = np.array(rows)
    im = ax.imshow(M, aspect="auto", cmap=T.tcl_cmap("seq_blue"), vmin=0, vmax=np.nanmax(M), interpolation="nearest")
    ax.set_yticks(range(len(labs))); ax.set_yticklabels([f.lower() for _, f in labs])
    for t, (_, f) in zip(ax.get_yticklabels(), labs):
        t.set_color(fc(f, "text"))
    ax.axhline(2.5, color="white", lw=2.0)
    for i, met in [(1, "AURC"), (4, "ECE")]:
        ax.text(1.03, i, met, transform=ax.get_yaxis_transform(), rotation=-90, ha="center", va="center",
                fontsize=5.6, color=INK)
    ax.set_xticks([0, 3, 6, 9]); ax.set_xticklabels(["1", "4", "7", "10"]); ax.set_xlabel("tasks learned")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cax = ax.inset_axes([1.08, 0.0, 0.03, 1.0])
    cb = ax.figure.colorbar(im, cax=cax); cb.ax.tick_params(labelsize=5, length=1.5); cb.outline.set_linewidth(0.4)
    ax.set_title(title)


def split_violin(ax, MC, title):
    """Split violins per family: left half ΔAURC, right half ΔECE (earlier − current)."""
    d = MC.copy(); d["family"] = d.method.map(FAM)
    g = np.linspace(-0.3, 1.0, 300)
    for i, f in enumerate(FAM_ORDER):
        s = d[d.family == f]
        for side, col, role in [(-1, "dAURC", "ours"), (1, "dECE", "replay")]:
            k = _kde(s[col], g, 0.35); k = 0.42 * k / k.max()
            msk = k > 0.004
            gg, kk = g[msk], k[msk]
            ax.fill_betweenx(gg, i, i + side * kk, color=T.tcl_color(role, "fill"), alpha=0.9, lw=0)
            ax.plot(i + side * kk, gg, color=T.tcl_color(role, "stroke"), lw=0.5)
            ax.plot([i, i + side * 0.3], [s[col].median()] * 2, color=T.tcl_color(role, "text"), lw=1.0)
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_xticks(range(3))
    # middle label one line lower: the three family names are wider than their slots
    ax.set_xticklabels([f.lower() if i != 1 else "\n" + f.lower() for i, f in enumerate(FAM_ORDER)])
    for t, f in zip(ax.get_xticklabels(), FAM_ORDER):
        t.set_color(fc(f, "text"))
    ax.set_xlim(-0.6, 2.6); ax.set_ylim(-0.3, 1.0)
    ax.set_ylabel("earlier − current")
    ax.plot([0.04], [0.945], marker="s", ms=4.2, fillstyle="left", mfc=T.tcl_color("ours", "fill"), mfcalt="white", mec=T.tcl_color("ours", "stroke"), mew=0.6, ls="none", transform=ax.transAxes, clip_on=False)
    ax.text(0.075, 0.97, "ΔAURC", transform=ax.transAxes, fontsize=5.0, va="top", color=T.tcl_color("ours", "text"))
    ax.plot([0.04], [0.855], marker="s", ms=4.2, fillstyle="right", mfc=T.tcl_color("replay", "fill"), mfcalt="white", mec=T.tcl_color("replay", "stroke"), mew=0.6, ls="none", transform=ax.transAxes, clip_on=False)
    ax.text(0.075, 0.88, "ΔECE", transform=ax.transAxes, fontsize=5.0, va="top", color=T.tcl_color("replay", "text"))
    ax.set_title(title)


def polar_heat(ax, MAT, title, nom=NOM, gap_deg=48):
    """Radial heatmap: angle = task (clockwise from the top), ring = method (inner =
    prototype), colour = per-task coverage of the final model, diverging around the
    promise. A gap at the top holds the ring labels."""
    cmap = T.tcl_cmap("div_coverage")
    norm = mcolors.TwoSlopeNorm(vmin=0.0, vcenter=nom, vmax=1.0)
    M = MAT.values; nT = M.shape[1]; nR = M.shape[0]
    g0 = np.deg2rad(gap_deg / 2)
    th = np.linspace(g0, 2 * np.pi - g0, nT + 1)
    w = th[1] - th[0]
    r0 = 1.6
    for i in range(nR):
        for j in range(nT):
            ax.bar((th[j] + th[j + 1]) / 2, 1.0, width=w * 0.94, bottom=i + r0,
                   color=cmap(norm(M[i, j])), ec="white", lw=0.3)
    ax.set_ylim(0, nR + r0 + 0.1)
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
    ax.set_xticks((th[:-1] + th[1:]) / 2); ax.set_xticklabels([f"T{j + 1}" for j in range(nT)], fontsize=5.0)
    ax.tick_params(axis="x", pad=-1)
    ax.set_yticks([]); ax.spines["polar"].set_visible(False); ax.grid(False)
    # ring key beside the wheel: rings are too thin for 7 pt names inside them
    names = list(MAT.index)[::-1]                    # outermost ring first
    ax.text(1.50, 0.98, "ring, outer\nto inner", transform=ax.transAxes, ha="left", va="top",
            fontsize=5.0, color=GREY, linespacing=1.0)
    for q, m in enumerate(names):
        ax.text(1.50, 0.74 - q * 0.115, m, transform=ax.transAxes, ha="left", va="top", fontsize=5.0,
                color=fc(FAM[m], "text"))
    ax.set_title(title, pad=10)
    sm_ = __import__("matplotlib.cm", fromlist=["ScalarMappable"]).ScalarMappable(norm=norm, cmap=cmap)
    cax = ax.inset_axes([1.16, 0.12, 0.035, 0.76])
    cb = ax.figure.colorbar(sm_, cax=cax); cb.set_ticks([0, nom, 1]); cb.ax.tick_params(labelsize=5, length=1.5)
    cb.outline.set_linewidth(0.4)


# ================================================================= figure 4
def protocol_raincloud(ax, PCm, title):
    """Raincloud per calibration set: half-density above, jittered cells below, box."""
    g = np.linspace(-0.02, 0.55, 300)
    rng = np.random.default_rng(1)
    for i, p in enumerate(PROTO):
        v = PCm[PCm.protocol == p].CovGap.values; role = PROTO_ROLE[p]
        k = _kde(v, g, 0.35); k = 0.42 * k / k.max()
        y = len(PROTO) - 1 - i
        ax.fill_between(g, y, y + k, color=T.tcl_color(role, "fill"), alpha=0.85, lw=0)
        ax.plot(g, y + k, color=T.tcl_color(role, "stroke"), lw=0.6)
        q1, q2, q3 = np.percentile(v, [25, 50, 75])
        ax.add_patch(Rectangle((q1, y - 0.11), q3 - q1, 0.08, fc=T.tcl_color(role, "stroke"), ec="none"))
        ax.plot([q2, q2], [y - 0.13, y - 0.01], color="white", lw=1.0)
        ax.scatter(v, y - 0.28 + rng.uniform(-0.07, 0.07, len(v)), s=3.5, color=T.tcl_color(role, "stroke"), zorder=3)
    ax.set_yticks(range(len(PROTO))); ax.set_yticklabels([PROTO_SHORT[p] for p in PROTO][::-1])
    for t, p in zip(ax.get_yticklabels(), PROTO[::-1]):
        t.set_color(T.tcl_color(PROTO_ROLE[p], "text"))
    ax.set_ylim(-0.5, len(PROTO) - 0.45); ax.set_xlim(-0.02, 0.55)
    ax.set_xlabel("mean coverage gap"); ax.set_title(title)


def estimation(ax, PCm, p_from, p_to, title, B=5000):
    """Gardner-Altman estimation plot: paired cells on the left, bootstrap distribution
    of the mean paired difference on a floating axis to the right."""
    W = PCm.pivot_table(index=["dataset", "backbone", "method"], columns="protocol", values="CovGap").reset_index()
    a, b = W[p_from].values, W[p_to].values
    for i in range(len(a)):
        ax.plot([0, 1], [a[i], b[i]], color=LIGHT, lw=0.6, zorder=1)
    for x, v, p in [(0, a, p_from), (1, b, p_to)]:
        ax.scatter(np.full(len(v), x), v, s=9, fc=T.tcl_color(PROTO_ROLE[p], "fill"),
                   ec=T.tcl_color(PROTO_ROLE[p], "stroke"), lw=0.5, zorder=3)
    ax.set_xticks([0, 1]); ax.set_xticklabels([PROTO_SHORT[p_from], PROTO_SHORT[p_to]])
    for t, p in zip(ax.get_xticklabels(), [p_from, p_to]):
        t.set_color(T.tcl_color(PROTO_ROLE[p], "text"))
    ax.set_xlim(-0.35, 1.35); ax.set_ylabel("mean coverage gap")
    diff = b - a
    bs = _boot(lambda k: diff[k].mean(), len(diff), B, seed=7)
    _shrink(ax, 0.66, 1.0)
    e = ax.inset_axes([1.10, 0, 0.40, 1])
    e.sharey(ax) if False else None
    m0 = b.mean()
    gy = np.linspace(bs.min(), bs.max(), 200)
    k = _kde(bs, gy, 0.3); k = 0.45 * k / k.max()
    e.fill_betweenx(gy + m0 - 0 if False else gy, 0, k, color=T.tcl_color("ours", "fill"), lw=0)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    e.plot([0, 0], [lo, hi], color=T.tcl_color("ours", "stroke"), lw=1.0)
    e.scatter([0], [diff.mean()], s=14, color=T.tcl_color("ours", "stroke"), zorder=3)
    e.axhline(0, color=INK, lw=0.6)
    e.set_xlim(-0.15, 0.6); e.set_xticks([])
    e.spines["bottom"].set_visible(False); e.spines["left"].set_visible(False); e.spines["right"].set_visible(True)
    e.yaxis.tick_right(); e.yaxis.set_label_position("right"); e.set_ylabel("paired Δ", fontsize=5.4)
    e.tick_params(labelsize=5)
    e.set_ylim(min(lo, -0.02) * 1.25, max(0.02, hi) * 1.6 if hi > 0 else 0.03)
    ax.set_title(title)
    return float(diff.mean()), float(lo), float(hi), int((diff < 0).sum()), len(diff)


def tradeoff(ax, KS, title):
    """Accuracy against coverage gap as k (reserved slots of m = 20) grows, one path per
    calibration set; the reserved-slot path is flat in gap while accuracy falls."""
    out = {}
    for p in ("A_current", "Cn_replayed", "C_rbcc"):
        g = KS[KS.protocol == p].groupby("k")[["ACC", "CovGap"]].mean().reset_index()
        if p == "C_rbcc":
            g = g[g.k > 0]
        role = PROTO_ROLE[p]
        ax.plot(g.CovGap, g.ACC, color=T.tcl_color(role, "stroke"), lw=1.0, zorder=2)
        ax.scatter(g.CovGap, g.ACC, s=10 + 3 * g.k, fc=T.tcl_color(role, "fill"), ec=T.tcl_color(role, "stroke"),
                   lw=0.6, zorder=3)
        top = g.loc[g.ACC.idxmax()]
        dx, ha = {"Cn_replayed": (-3, "right"), "A_current": (3, "left"), "C_rbcc": (2, "left")}.get(p, (0, "center"))
        ax.annotate(PROTO_SHORT[p], (top.CovGap, top.ACC), xytext=(dx, 4), textcoords="offset points", fontsize=5.2,
                    ha=ha, va="bottom", color=T.tcl_color(role, "text"))
        if p == "C_rbcc":
            for _, r in g[g.k.isin([1, 10])].iterrows():
                ax.annotate(f"k = {int(r.k)}", (r.CovGap, r.ACC), xytext=(6, 0), textcoords="offset points",
                            fontsize=4.8, va="center", color=T.tcl_color(role, "text"))
        out[p] = g
    ax.set_xlabel("mean coverage gap"); ax.set_ylabel("final accuracy")
    allg = pd.concat(out.values()); a0, a1 = allg.ACC.min(), allg.ACC.max(); pad = 0.12 * (a1 - a0)
    ax.set_xlim(-0.01, allg.CovGap.max() * 1.12); ax.set_ylim(a0 - pad, a1 + 2.2 * pad)

    ax.set_title(title)
    return out


def k_bounds(ax, MK, title, kmin=9, kcons=19):
    ax.axvspan(0, kmin, color=T.TCL_NEUTRAL["rule"], lw=0, zorder=0)
    ax.axvspan(kmin, kcons, color=T.tcl_color("replay", "tint"), lw=0, zorder=0)
    ax.text(kmin / 2, 0.97, "fall-\nback", transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=4.8, color=INK)
    ax.text((kmin + kcons) / 2, 0.97, "q =\nmax", transform=ax.get_xaxis_transform(), ha="center",
            va="top", fontsize=4.8, color=T.tcl_color("replay", "text"))
    ends = []
    for s in SCHEME:
        g = MK[MK.scheme == s].groupby("k")["size"].agg(["mean", ci95]).reset_index(); sty = scheme_style(s)
        ax.fill_between(g.k, g["mean"] - g.ci95, g["mean"] + g.ci95, color=sty["c"], alpha=0.15, lw=0)
        ax.plot(g.k, g["mean"], color=sty["c"], lw=1.1, marker=sty["marker"], ms=2.6,
                mfc="white" if sty["hollow"] else sty["c"], mec=sty["c"], label=SCHEME_SHORT[s])
    ax.set_xlim(0, 42); ax.set_xticks([3, 9, 19, 30, 40]); ax.set_ylim(1.2, 5.4)
    ax.legend(loc="upper right", fontsize=4.9, handletextpad=0.3, borderaxespad=0.1, handlelength=1.4)
    ax.set_xlabel("reserved slots per class, k"); ax.set_ylabel("mean set size")
    ax.set_title(title)


def scheme_k_heat(ax, MK, title, col="mean_class", nom=NOM):
    """Scheme x k heatmap of mean per-class coverage on a balanced test set, centred on
    the promise; numbers printed."""
    P = MK.groupby(["scheme", "k"])[col].mean().unstack().loc[SCHEME]
    cmap = T.tcl_cmap("div_coverage")
    norm = mcolors.TwoSlopeNorm(vmin=0.85, vcenter=nom, vmax=0.95)
    im = ax.imshow(P.values, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(range(0, P.shape[1], 3)); ax.set_xticklabels(P.columns[::3])
    ax.set_yticks(range(P.shape[0])); ax.set_yticklabels(["all past", "one thr.", "per class"])
    ax.set_xlabel("reserved slots per class, k"); ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cax = ax.inset_axes([1.02, 0, 0.03, 1])
    cb = ax.figure.colorbar(im, cax=cax); cb.set_ticks([0.85, 0.90, 0.95]); cb.ax.tick_params(labelsize=5, length=1.5)
    cb.outline.set_linewidth(0.4)
    ax.set_title(title)
    return P


def separation_area(ax, MF, title):
    """Coverage under the old-class mixture vs replay budget; the area between the
    single-threshold schemes and the per-class rule is shaded."""
    G = {}
    for s in SCHEME:
        G[s] = MF[MF.scheme == s].groupby("replay_slots").agg(acc=("acc", "mean"), m=("cov_old100", "mean"),
                                                            ci=("cov_old100", ci95)).reset_index()
    x = G["rbcc_mondrian"].replay_slots.values
    xl = np.log2(x)
    ax.fill_between(xl, G["rbcc_pooled"].m, G["rbcc_mondrian"].m, color=T.tcl_color("ours", "tint"), lw=0, zorder=0)
    for s in SCHEME:
        g = G[s]; sty = scheme_style(s)
        ax.errorbar(np.log2(g.replay_slots), g.m, yerr=g.ci, color=sty["c"], lw=1.1, marker=sty["marker"], ms=2.8,
                    mfc="white" if sty["hollow"] else sty["c"], mec=sty["c"], capsize=1.2, elinewidth=0.6)
    T.tcl_nominal(ax, NOM, label=None)
    ax.set_xticks(xl); ax.set_xticklabels([str(v) for v in x])
    ax.set_xlabel("replay slots per class"); ax.set_ylabel("coverage, old-class mixture")
    ends = [(xl[-1], G[s].m.values[-1], SCHEME_SHORT[s], "oracle" if s == "pooled_oracle" else "ours") for s in SCHEME]
    ax.set_xlim(xl[0] - 0.3, xl[-1] + 0.3); ax.set_ylim(0.855, 0.925)
    for s in SCHEME:
        sty = scheme_style(s)
        ax.plot([], [], color=sty["c"], marker=sty["marker"], ms=2.8, mfc="white" if sty["hollow"] else sty["c"],
                mec=sty["c"], lw=1.1, label=SCHEME_SHORT[s])
    ax.legend(loc="lower right", fontsize=4.9, handletextpad=0.3, borderaxespad=0.1, handlelength=1.4)
    lo_, hi_ = ax.get_ylim(); ax.set_ylim(min(lo_, 0.835), hi_)   # room under the curves for the 7 pt legend
    ax.set_yticks([t for t in (0.84, 0.86, 0.88, 0.90, 0.92) if min(lo_, 0.835) <= t <= hi_])
    ax.yaxis.set_major_formatter(__import__("matplotlib.ticker", fromlist=["FormatStrFormatter"]).FormatStrFormatter("%.2f"))
    ax.set_title(title)
    return G


def halfeye(ax, MF, EM, title, replay=20):
    """Half-eye plots (density + interval) of the coverage lost to the shift, per scheme,
    for the frozen and end-to-end regimes."""
    out = {}
    g = np.linspace(-0.05, 0.06, 300)
    for j, (lab, D) in enumerate([("frozen ViT", MF[MF.replay_slots == replay]), ("end-to-end ResNet-18", EM)]):
        for i, s in enumerate(SCHEME):
            v = (D[D.scheme == s].cov_uniform - D[D.scheme == s].cov_old100).values
            y = j * 3.6 + i; sty = scheme_style(s)
            k = _kde(v, g, 0.6); k = 0.75 * k / k.max()
            ax.fill_between(g, y, y + k, color=sty["c"], alpha=0.25, lw=0)
            ax.plot(g, y + k, color=sty["c"], lw=0.6)
            m, c = v.mean(), ci95(v)
            ax.plot([m - c, m + c], [y, y], color=sty["c"], lw=1.4)
            ax.scatter(m, y, s=14, marker=sty["marker"], fc="white" if sty["hollow"] else sty["c"], ec=sty["c"], lw=0.8,
                       zorder=3)
            out[(lab, s)] = (float(m), float(c))
        ax.text(0.06, j * 3.6 + 2.75, lab, fontsize=5.4, color=INK, ha="right", fontweight="bold")
    ax.axvline(0, color=INK, lw=0.6)
    yt = [j * 3.6 + i for j in range(2) for i in range(3)]
    ax.set_yticks(yt); ax.set_yticklabels(["all past", "one thr.", "per class"] * 2, fontsize=5.0)
    ax.set_ylim(-0.5, 7.2 + 0.2); ax.set_xlim(-0.05, 0.06)
    ax.set_xlabel("coverage lost to the shift")
    ax.set_title(title)
    return out


# ================================================================= figure 5
def parallel_coords(ax, EV, title):
    """Parallel coordinates across the two regimes: coverage on the task just learned
    and on earlier tasks, frozen then end-to-end, one line per method."""
    E = EV.copy(); E["m"] = E.method.str.replace("E2E-", "")
    cols = ["frozen_cov_current", "frozen_cov_earlier", "e2e_cov_current", "e2e_cov_earlier"]
    labs = ["just\nlearned", "earlier", "just\nlearned", "earlier"]
    for x in range(4):
        ax.axvline(x, color=LIGHT, lw=0.8, zorder=0)
    ends = []
    for _, r in E.iterrows():
        role = "ours" if r.m == "RBCC" else ROLE.get(FAM.get(r.m, "Gradient"))
        ax.plot(range(4), [r[c] for c in cols], color=T.tcl_color(role, "stroke"), lw=1.1, marker="o", ms=2.6, zorder=2)
        ends.append((3, r[cols[-1]], r.m, role))
    ax.axvspan(1.5, 3.5, color=T.TCL_NEUTRAL["rule"], alpha=0.35, lw=0, zorder=-1)
    T.tcl_nominal(ax, NOM, label=None)
    ax.set_xticks(range(4)); ax.set_xticklabels(labs, fontsize=5.0)
    ax.set_xlim(-0.2, 3.9); ax.set_ylim(-0.02, 1.02); ax.set_ylabel("coverage")
    from matplotlib.transforms import blended_transform_factory as _btf
    _tr = _btf(ax.transData, ax.transAxes)
    for _x, _g in ((0.5, "frozen backbone"), (2.5, "end-to-end")):
        ax.text(_x, -0.215, _g, transform=_tr, ha="center", va="top", fontsize=5.2, color=INK)
    T.tcl_end_labels(ax, ends, fontsize=5.0)
    ax.set_title(title)


def protocol_lollipop_inset(ax, EP, EC, title):
    """End-to-end final coverage per calibration set (lollipops from 0); inset: the fit
    adequacy curve that fixes the training budget."""
    E = EP.set_index("protocol").loc[PROTO]
    for i, p in enumerate(PROTO):
        role = PROTO_ROLE[p]
        ax.plot([0, E.loc[p, "mean"]], [i, i], color=T.tcl_color(role, "fill"), lw=2.2, solid_capstyle="round")
        ax.errorbar(E.loc[p, "mean"], i, xerr=E.loc[p, "ci95"], fmt="o", ms=3.8, color=T.tcl_color(role, "stroke"),
                    mfc=T.tcl_color(role, "fill"), capsize=1.2, elinewidth=0.7)
    T.tcl_nominal(ax, NOM, axis="x")
    ax.set_yticks(range(4)); ax.set_yticklabels([PROTO_SHORT[p] for p in PROTO]); ax.set_ylim(3.6, -0.6)
    for t, p in zip(ax.get_yticklabels(), PROTO):
        t.set_color(T.tcl_color(PROTO_ROLE[p], "text"))
    ax.set_xlim(0, 1.0); ax.set_xlabel("final coverage, end-to-end")
    if EC is None:
        ax.set_title(title); return None
    ins = ax.inset_axes([0.08, 0.55, 0.42, 0.38])
    ins.plot(EC.epochs, EC.acc, color=INK, lw=0.9, marker="o", ms=1.8)
    ins.axvline(150, color=T.tcl_color("ours", "stroke"), lw=0.7, ls=(0, (3, 2)))
    ins.set_xticks([150, 450]); ins.set_yticks([0.90, 0.92]); ins.tick_params(labelsize=4.6, length=1.5, pad=1)
    ins.set_title("fit, epochs", fontsize=4.8, pad=1.5)
    ax.set_title(title)


def bland_altman(ax, SA, title):
    """Bland-Altman agreement between backbones: difference against mean of the final
    coverage of each cell, with bias and 95% limits of agreement."""
    W = SA.pivot_table(index=["dataset", "method", "family"], columns="backbone", values="FinalCov").reset_index()
    m = (W.vit_b16_in21k + W.dinov2_base) / 2; dlt = W.dinov2_base - W.vit_b16_in21k
    bias, sd = dlt.mean(), dlt.std(ddof=1)
    ax.axhspan(bias - 1.96 * sd, bias + 1.96 * sd, color=T.TCL_NEUTRAL["rule"], lw=0, zorder=0)
    ax.axhline(bias, color=INK, lw=0.8); ax.axhline(0, color=GREY, lw=0.6, ls=(0, (4, 2.5)))
    for f in FAM_ORDER:
        s = W.family == f
        ax.scatter(m[s], dlt[s], s=11, fc=fc(f), ec=fc(f, "stroke"), lw=0.5, zorder=3)
    ax.text(0.02, bias + 1.96 * sd, "+1.96 SD", transform=ax.get_yaxis_transform(), ha="left", va="bottom",
            fontsize=4.8, color=GREY)
    ax.text(0.02, bias - 1.96 * sd, "−1.96 SD", transform=ax.get_yaxis_transform(), ha="left", va="top",
            fontsize=4.8, color=GREY)
    ax.set_xlabel("mean coverage, two backbones"); ax.set_ylabel("DINOv2 − ViT")
    ax.set_title(title)
    return float(bias), float(sd)


def connected_scores(ax, SC, title):
    """LAC to APS as arrows in (set size, coverage) space per method: APS buys coverage
    with larger sets."""
    g = SC.groupby(["method", "score"])[["final_cov", "set_size"]].mean().unstack()
    for m in METHOD_ORDER:
        f = FAM[m]
        x0, y0 = g.loc[m, ("set_size", "lac")], g.loc[m, ("final_cov", "lac")]
        x1, y1 = g.loc[m, ("set_size", "aps")], g.loc[m, ("final_cov", "aps")]
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=fc(f, "stroke"), lw=0.9, mutation_scale=6,
                                    shrinkA=2, shrinkB=2))
        ax.scatter(x0, y0, s=12, fc="white", ec=fc(f, "stroke"), lw=0.8, zorder=3)
        ax.scatter(x1, y1, s=12, fc=fc(f, "stroke"), ec=fc(f, "stroke"), lw=0.8, zorder=3)
        pass
    ax.set_xscale("log"); T.tcl_nominal(ax, NOM, label=None)
    def P(m, s): return g.loc[m, ("set_size", s)], g.loc[m, ("final_cov", s)]
    ax.set_ylim(0.47, 0.955)
    ax.annotate("prototype", P("iCaRL", "aps"), xytext=(0, 5), textcoords="offset points", ha="center",
                va="bottom", fontsize=5.0, color=fc("Prototype", "text"))
    ax.annotate("replay", P("Replay", "lac"), xytext=(0, 5), textcoords="offset points", ha="center",
                va="bottom", fontsize=5.0, color=fc("Replay", "text"))
    ax.annotate("gradient", P("LwF", "aps"), xytext=(4, -6), textcoords="offset points", ha="left",
                va="top", fontsize=5.0, color=fc("Gradient", "text"))
    ax.set_xlabel("set size (labels)"); ax.set_ylabel("final coverage")
    ax.scatter([], [], s=12, fc="white", ec=INK, label="LAC"); ax.scatter([], [], s=12, fc=INK, ec=INK, label="APS")
    ax.legend(loc="lower right", fontsize=5.0, handletextpad=0.1, borderaxespad=0.1)
    ax.set_xlim(right=ax.get_xlim()[1] * 2.2)
    ax.set_title(title)
    return g


def radar(ax, SA, title):
    """Radar of the mean coverage gap per family across the four benchmarks."""
    ds = ["cifar100", "cub200", "imagenet_r", "medmix26"]
    th = np.linspace(0, 2 * np.pi, len(ds), endpoint=False) + np.pi / 4   # spokes on the diagonals: top stays clear
    thc = np.r_[th, th[0]]
    out = {}
    for f in FAM_ORDER:
        v = [SA[(SA.dataset == d) & (SA.family == f)].CovGap.mean() for d in ds]
        vc = np.r_[v, v[0]]
        ax.fill(thc, vc, color=fc(f), alpha=0.45, lw=0)
        ax.plot(thc, vc, color=fc(f, "stroke"), lw=1.0, marker="o", ms=2.2)
        out[f] = dict(zip(ds, v))
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
    ax.set_xticks(th); ax.set_xticklabels(["C100", "CUB", "IN-R", "Med"], fontsize=5.2)
    ax.tick_params(axis="x", pad=0)
    ax.set_ylim(0, 0.8); ax.set_yticks([0.2, 0.4, 0.6]); ax.set_yticklabels(["0.2", "0.4", "0.6"], fontsize=4.4,
                                                                           color=GREY)
    ax.set_rlabel_position(0)
    ax.grid(color=LIGHT, lw=0.5); ax.spines["polar"].set_color(LIGHT)
    ax.set_title(title, pad=14)
    return out


from matplotlib.ticker import FormatStrFormatter


def fit_curve(ax, EC, title, chosen=150):
    ceil = EC.acc.max()
    ax.axhspan(ceil - 0.015, ceil, color=T.tcl_color("ours", "tint"), lw=0, zorder=0)
    ax.text(EC.epochs.max(), ceil - 0.0155, "within 0.015\nof ceiling", ha="right", va="top", fontsize=4.8,
            color=T.tcl_color("ours", "text"))
    ax.plot(EC.epochs, EC.acc, color=INK, lw=1.0, marker="o", ms=2.6)
    ax.axvline(chosen, color=T.tcl_color("ours", "stroke"), lw=0.8, ls=(0, (3, 2)))
    lo_, hi_ = np.floor(EC.acc.min() * 100) / 100, np.ceil(ceil * 100) / 100
    ax.set_ylim(lo_ - 0.002, hi_ + 0.002); ax.set_yticks(np.round(np.arange(lo_, hi_ + 1e-9, 0.01), 2))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_xlabel("epochs per task"); ax.set_ylabel("single-task accuracy")
    ax.set_title(title)


def shift_dose(ax, MF, title, replay=20):
    """Dose-response to the deployment shift at a fixed replay budget: coverage as the
    test mixture moves from class-balanced to 80% and 100% earlier classes (slopegraph)."""
    D = MF[MF.replay_slots == replay]
    cols = [("cov_uniform", "bal."), ("cov_old80", "80%"), ("cov_old100", "100%")]
    ends = []
    for s in SCHEME:
        sty = scheme_style(s); d = D[D.scheme == s]
        m = [d[c].mean() for c, _ in cols]; ci = [ci95(d[c]) for c, _ in cols]
        ax.errorbar(range(3), m, yerr=ci, color=sty["c"], lw=1.1, marker=sty["marker"], ms=3.0,
                    mfc="white" if sty["hollow"] else sty["c"], mec=sty["c"], capsize=1.2, elinewidth=0.6)
        ends.append((2, m[-1], SCHEME_SHORT[s], "oracle" if s == "pooled_oracle" else "ours"))
    T.tcl_nominal(ax, NOM, label=None)
    ax.set_xticks(range(3)); ax.set_xticklabels([l for _, l in cols], fontsize=5.0)
    ax.set_xlim(-0.3, 2.3); ax.set_xlabel("earlier classes in test mix")
    ax.set_title(title)


def gauge_matrix(ax, SA, title, nom=NOM):
    """56 method-configuration cells as a matrix of gauges: row = method, column =
    dataset x backbone. Each cell is filled up to its final coverage; the tick inside
    every cell marks the promised 0.90. Cells that meet it are outlined in ink."""
    ds = ["cifar100", "cub200", "imagenet_r", "medmix26"]; bbs = ["vit_b16_in21k", "dinov2_base"]
    cols = [(d, b) for d in ds for b in bbs]
    S = SA.set_index(["method", "dataset", "backbone"])
    w, h, gx = 0.86, 0.72, 0.0
    OX = 1.55          # room for the method labels inside the axes, so the panel letter sits left of them
    n_ok = 0
    for i, m in enumerate(METHOD_ORDER):
        f = FAM[m]
        for j, (d, b) in enumerate(cols):
            x = OX + j + 0.18 * (j // 2)
            y = len(METHOD_ORDER) - 1 - i
            if (m, d, b) not in S.index:
                continue
            v = float(S.loc[(m, d, b), "FinalCov"])
            ok = v >= nom
            n_ok += ok
            ax.add_patch(Rectangle((x, y), w, h, fc=T.tcl_color(ROLE[f], "tint"), ec="none", zorder=1))
            ax.add_patch(Rectangle((x, y), w * v, h, fc=fc(f), ec="none", zorder=2))
            ax.plot([x + w * nom] * 2, [y, y + h], color=INK, lw=0.6, zorder=3)
            ax.add_patch(Rectangle((x, y), w, h, fc="none", ec=INK if ok else fc(f, "stroke"),
                                   lw=1.0 if ok else 0.35, zorder=4))
    xs = [OX + j + 0.18 * (j // 2) + w / 2 for j in range(len(cols))]
    for k, d in enumerate(ds):
        ax.text((xs[2 * k] + xs[2 * k + 1]) / 2, len(METHOD_ORDER) + 0.75,
                {"cifar100": "C100", "cub200": "CUB", "imagenet_r": "IN-R", "medmix26": "Med"}[d],
                ha="center", va="bottom", fontsize=5.2, color=INK)
        for q, b in enumerate(bbs):
            ax.text(xs[2 * k + q], len(METHOD_ORDER) - 0.18, "V" if b.startswith("vit") else "D", ha="center",
                    va="bottom", fontsize=4.6, color=GREY)
    for i, m in enumerate(METHOD_ORDER):
        ax.text(OX - 0.12, len(METHOD_ORDER) - 1 - i + h / 2, m, ha="right", va="center", fontsize=5.5,
                color=fc(FAM[m], "text"))
    right = xs[-1] + w / 2 + 0.45
    nb = len(SA) - n_ok
    ax.text(right, len(METHOD_ORDER) / 2 + 0.55, f"{nb}/{len(SA)}", fontsize=11, fontweight="bold",
            color=T.tcl_color("failing_default", "text"), va="bottom", ha="left")
    ax.text(right, len(METHOD_ORDER) / 2 + 0.35, "below 0.90", fontsize=5.4, color=INK, va="top", ha="left")
    ax.set_xlim(0, right + 2.5); ax.set_ylim(-0.05, len(METHOD_ORDER) + 1.75)
    ax.axis("off")
    ax.set_title(title)
    return nb


def circular_bars(ax, SA, title, nom=NOM):
    """Circular bar plot of the 56 method-configuration cells: bar length = final
    coverage, grouped by method family with a gap between groups, sorted within each
    group; the dashed ring is the promised 0.90. Bars that reach it are drawn in ink."""
    d = SA.copy()
    d["fo"] = d.family.map({f: i for i, f in enumerate(FAM_ORDER)})
    d = d.sort_values(["fo", "FinalCov"], ascending=[True, False]).reset_index(drop=True)
    gap = 3
    pos, cur, prev = [], 0, None
    for f in d.family:
        if prev is not None and f != prev:
            cur += gap
        pos.append(cur); cur += 1; prev = f
    n = cur + gap
    th = np.array(pos) * 2 * np.pi / n
    w = 2 * np.pi / n * 0.82
    r0 = 0.85
    ok = d.FinalCov.values >= nom
    ax.bar(th, d.FinalCov.values, width=w, bottom=r0, color=[fc(f) for f in d.family],
           ec=[INK if o else fc(f, "stroke") for f, o in zip(d.family, ok)], lw=[0.9 if o else 0.3 for o in ok],
           zorder=3)
    tt = np.linspace(0, 2 * np.pi, 400)
    ax.plot(tt, np.full_like(tt, r0 + nom), color=INK, lw=0.6, ls=(0, (3, 2)), zorder=4)
    for rr in (0.5, 1.0):
        ax.plot(tt, np.full_like(tt, r0 + rr), color=LIGHT, lw=0.4, zorder=1)
    for f in FAM_ORDER:
        idx = np.where(d.family.values == f)[0]
        mid = th[idx].mean()
        deg = np.rad2deg(mid) % 360
        ha = "left" if 10 < deg < 170 else ("right" if 190 < deg < 350 else "center")
        ax.text(mid, r0 + 1.58, f.lower(), ha=ha, va="center", fontsize=5.4, color=fc(f, "text"), clip_on=False)
    ax.text(np.deg2rad(4), r0 + nom + 0.02, "0.90", fontsize=4.6, color=INK, ha="left", va="bottom")
    nb = int((~ok).sum())
    ax.text(0, 0.0, f"{nb}/{len(d)}", ha="center", va="bottom", fontsize=8.5, fontweight="bold",
            color=T.tcl_color("failing_default", "text"))
    ax.text(np.pi, 0.08, "below 0.90", ha="center", va="top", fontsize=4.8, color=INK)
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
    ax.set_ylim(0, r0 + 1.62)
    ax.set_anchor("W")
    ax.text(1.10, 0.42, "bar = final coverage\nof one method × dataset\n× backbone\nconfiguration", transform=ax.transAxes,
            fontsize=5.0, color=INK, va="center", ha="left")
    ax.text(1.10, 0.16, "dashed ring = 0.90\nink outline = meets it", transform=ax.transAxes, fontsize=5.0,
            color=GREY, va="center", ha="left"); ax.set_xticks([]); ax.set_yticks([])
    ax.spines["polar"].set_visible(False); ax.grid(False)
    ax.set_title(title)
    return nb, int(ok.sum())
