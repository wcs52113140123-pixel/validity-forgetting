"""Plotting helpers for trustworthy continual-learning papers.

Defaults come from a measured corpus: 806 main-text figures from 148 papers
(77 CCF-A/B, 71 CNS-family, 2022-2026) on continual learning, conformal prediction,
uncertainty calibration and selective prediction. See SKILL.md for the numbers.

Palette: "sea-salt coral" four hues. FILL tones are light (contrast 1.35-1.85 on white)
and may only fill areas; STROKE tones draw lines, edges, error bars; TEXT tones write
words. Stroke lightness is deliberately staggered so the four hues stay apart in
greyscale and under protan/deutan simulation.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch, Rectangle

TCL_PAL = {
    "blue":   {"tint": "#ECF4FF", "fill": "#A5C1E6", "mid": "#789ED0", "stroke": "#4D7CB8", "text": "#284E7E"},
    "pink":   {"tint": "#FFEFF2", "fill": "#F6C4CF", "mid": "#AD6E7E", "stroke": "#793147", "text": "#672C3D"},
    "green":  {"tint": "#E2FAEE", "fill": "#9ED4BA", "mid": "#69B795", "stroke": "#36A278", "text": "#045C40"},
    "yellow": {"tint": "#FFF2DC", "fill": "#FDD99D", "mid": "#CC9B41", "stroke": "#836328", "text": "#654703"},
}
TCL_NEUTRAL = {"ink": "#2B2F36", "grey": "#8A9099", "light": "#D5D9DF", "rule": "#E9ECF0"}
TCL_CMAP_STOPS = {
    "seq_blue": ["#ECF4FF", "#A5C1E6", "#789ED0", "#4D7CB8", "#284E7E"],
    "div_coverage": ["#86364E", "#EDB6C3", "#F6F6F4", "#ABCAF3", "#235594"],
    "pastel_surface": ["#4D7CB8", "#A5C1E6", "#F6C4CF", "#FDD99D"],
}
# One meaning per hue, fixed across the whole paper. Colour encodes ONE dimension per
# figure; a second dimension goes to lightness (stroke vs mid), fill vs hollow, or shape.
TCL_SEMANTICS = {
    "ours": "blue", "reserved_slots": "blue", "per_class_rule": "blue",
    "failing_default": "pink", "current_task_calibration": "pink", "gradient_family": "pink",
    "replay": "yellow", "replay_family": "yellow", "replayed_exemplars": "yellow",
    "prototype_family": "green",
    "oracle": "neutral", "nominal": "neutral",
}
# Measured corpus norms (median [IQR]).
TCL_NORMS = {
    "panels_fig1": {"CCF": (2, 1, 3), "CNS": (4, 2.5, 5.5)},
    "panels_rest": {"CCF": (3, 2, 4), "CNS": (6, 3, 9)},
    "distinct_types": {"CCF": 1, "CNS": 2},
    "share_single_type": {"CCF": 0.774, "CNS": 0.416},
    "fig1_has_schematic": {"CCF": 0.69, "CNS": 0.92},
    "irregular_mosaic": {"CCF": 0.16, "CNS": 0.58},
    "shows_variability": {"CCF": 0.23, "CNS": 0.42},
    "white_no_grid": {"CCF": 0.52, "CNS": 0.81},
    "direct_labels": {"CCF": 0.04, "CNS": 0.18},
    "any_annotation": {"CCF": 0.58, "CNS": 0.85},
    "claim_sentence_titles": {"CCF": 0.02, "CNS": 0.02},
    "saturation_median": {"CCF": 0.42, "CNS": 0.40},
}
TCL_LETTER_GID = "tcl_panel_letter"
TCL_TITLE_GID = "tcl_panel_title"
TCL_WIDTH_IN = {"double": 7.09, "one_half": 4.72, "single": 3.46}


def tcl_style(sizes=(6.5, 6, 5.5), font="DejaVu Sans"):
    """White ground, no grid, open frame, three-size type ladder (titles, legend, ticks).

    81% of CNS figures in the corpus are white with no grid against 52% of CCF; grey
    darkgrid backgrounds appear in 11% of CCF and 3% of CNS figures.
    """
    base, mid, small = sizes
    mpl.rcParams.update({
        "font.family": font, "font.size": base, "axes.titlesize": base,
        "axes.labelsize": base, "xtick.labelsize": small, "ytick.labelsize": small,
        "legend.fontsize": mid, "axes.titlelocation": "left", "axes.titleweight": "normal",
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "axes.edgecolor": TCL_NEUTRAL["ink"], "axes.labelcolor": TCL_NEUTRAL["ink"],
        "xtick.color": TCL_NEUTRAL["ink"], "ytick.color": TCL_NEUTRAL["ink"],
        "axes.linewidth": 0.6, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2.0, "ytick.major.size": 2.0, "xtick.major.pad": 1.5,
        "ytick.major.pad": 1.5, "axes.labelpad": 1.5, "axes.titlepad": 2.0, "lines.linewidth": 1.0,
        "legend.handlelength": 1.2, "legend.handletextpad": 0.4, "legend.borderaxespad": 0.2,
        "legend.labelspacing": 0.25,
        "legend.frameon": False, "savefig.dpi": 300, "figure.dpi": 110,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def tcl_color(role_or_hue, tone="stroke"):
    """Colour for a semantic role ('ours', 'replay', ...) or a hue name, at a tone.

    tone: tint | fill | mid | stroke | text. Neutral roles return neutral greys.
    """
    hue = TCL_SEMANTICS.get(role_or_hue, role_or_hue)
    if hue == "neutral":
        return {"tint": TCL_NEUTRAL["rule"], "fill": TCL_NEUTRAL["light"], "mid": TCL_NEUTRAL["grey"],
                "stroke": TCL_NEUTRAL["grey"], "text": TCL_NEUTRAL["ink"]}[tone]
    return TCL_PAL[hue][tone]


def tcl_cmap(name="seq_blue"):
    """seq_blue (monotone quantities), div_coverage (centre at the nominal level),
    pastel_surface (smooth surfaces). All three are lightness-monotonic (per side for
    the diverging map)."""
    return LinearSegmentedColormap.from_list(name, TCL_CMAP_STOPS[name], N=256)


def tcl_mosaic(layout, figsize=None, per_panel=(2.3, 1.55), **gs):
    """Irregular multi-panel layout at the printed width (default 180 mm).

    Use a fine column grid (12 or 24 columns) so panel widths follow content, and keep
    hspace/wspace small; the spacing that matters is set by tcl_finalize, which aligns
    letters and titles, not by padding the grid.
    Irregular multi-panel layout. 58% of CNS figures use an irregular mosaic against
    16% of CCF, and none of the 24 mined CCF plotting repositories used subplot_mosaic.

    layout: list of row strings ("aab", "cde") or one string with ';' between rows.
    '.' leaves a cell empty. Returns (fig, dict letter -> Axes).
    """
    if isinstance(layout, str):
        layout = [r for r in layout.replace(";", "\n").splitlines() if r.strip()]
    if layout and isinstance(layout[0], str):
        layout = [list(r) for r in layout]
    rows, cols = len(layout), max(len(r) for r in layout)
    if figsize is None:
        figsize = (TCL_WIDTH_IN["double"], per_panel[1] * rows)
    kw = {"hspace": gs.pop("hspace", 0.9), "wspace": gs.pop("wspace", 0.9)}
    kw.update(gs)
    psk = gs_extra.get("per_subplot_kw") if (gs_extra := {k: kw.pop(k) for k in list(kw) if k == "per_subplot_kw"}) else None
    fig, axd = plt.subplot_mosaic(layout, figsize=figsize, gridspec_kw=kw, empty_sentinel=".",
                                  per_subplot_kw=psk)
    # Grid position of each panel, kept separately: a colorbar or divider replaces an
    # axes' subplotspec with a nested one, after which rowspan/colspan no longer refer
    # to this grid and row/column alignment would silently use the wrong cells.
    cells = {}
    for i, row in enumerate(layout):
        for j, k in enumerate(row):
            if k != "." and k not in cells:
                cells[k] = (i, j)
    fig._tcl_cells = cells
    return fig, axd


def tcl_letters(fig, axd, case="lower", size=8, pad=4.0):
    """Bold panel letters placed AFTER drawing, at each panel's tight bounding box.

    Placing letters at creation time puts them on top of tick labels that do not exist
    yet; this measures the drawn extent instead. Call after all content is drawn.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    items = axd.items() if isinstance(axd, dict) else zip("abcdefghijklmnopqrstuvwxyz", axd)
    for k, ax in items:
        if not ax.get_visible():
            continue
        # An axis-off panel (schematic placeholder) has no tick labels, so its tight box
        # is the frame itself; push the letter outside the frame instead of onto it.
        bb = ax.get_tightbbox(r) if ax.axison else ax.get_window_extent(r)
        extra = 0.0 if ax.axison else pad * 2.5
        x0, y1 = inv.transform((bb.x0 - pad - extra, bb.y1 + pad * 0.2 + extra))
        t = fig.text(x0, y1, k.upper() if case == "upper" else k.lower(), fontsize=size,
                     fontweight="bold", va="top", ha="left", color=TCL_NEUTRAL["ink"])
        t.set_gid(TCL_LETTER_GID)


def tcl_placeholder(ax, pid, title, note="schematic placeholder"):
    """Reserve a panel for a schematic that is NOT drawn in code.

    Draws a dashed rounded frame with the placeholder id (e.g. 'S1') so the layout,
    spacing and letters are final; the image is pasted in later. Keep pid identical to
    the prompt id in the prompt document.
    """
    ax.set_axis_off()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.add_patch(FancyBboxPatch((0.01, 0.01), 0.98, 0.98, boxstyle="round,pad=0,rounding_size=0.03",
                                fc=TCL_NEUTRAL["rule"], ec=TCL_NEUTRAL["grey"], lw=0.8, ls=(0, (4, 3))))
    # stacked in points from the centre, so the three lines never overlap at any panel size
    ax.annotate(f"[{pid}]", (0.5, 0.5), xytext=(0, 9), textcoords="offset points", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=TCL_NEUTRAL["grey"])
    ax.annotate(title, (0.5, 0.5), xytext=(0, 6), textcoords="offset points", ha="center", va="top",
                fontsize=5.8, color=TCL_NEUTRAL["ink"])
    nl = title.count("\n") + 1
    ax.annotate(note, (0.5, 0.5), xytext=(0, 6 - 7.6 * nl - 3), textcoords="offset points", ha="center", va="top",
                fontsize=5.0, color=TCL_NEUTRAL["grey"], style="italic")
    return ax


def tcl_nominal(ax, y=0.90, label="nominal 0.90", side="right", axis="y"):
    """The promised level as a thin dashed neutral line, named once on the panel."""
    if axis == "y":
        ax.axhline(y, color=TCL_NEUTRAL["grey"], lw=0.8, ls=(0, (4, 2.5)), zorder=1)
        if label:
            x = {"right": 0.99, "left": 0.01, "center": 0.5}.get(side, side)
            ha = {"right": "right", "left": "left"}.get(side, "center")
            ax.text(x, y + 0.012 * (ax.get_ylim()[1] - ax.get_ylim()[0]), label,
                    transform=ax.get_yaxis_transform(), ha=ha,
                    va="bottom", fontsize=5.2, color=TCL_NEUTRAL["grey"])
    else:
        ax.axvline(y, color=TCL_NEUTRAL["grey"], lw=0.8, ls=(0, (4, 2.5)), zorder=1)


def tcl_regions(ax, spans, axis="y", alpha=1.0):
    """Shade validity regions as tint-tone backgrounds, e.g.
    spans=[(0.9, 1.0, 'ours', 'meets promise'), (0, 0.9, 'failing_default', 'fails')].
    Pattern seen in CNS conformal / deferral figures (validity drawn as background)."""
    for lo, hi, role, lab in spans:
        c = tcl_color(role, "tint")
        if axis == "y":
            ax.axhspan(lo, hi, color=c, alpha=alpha, zorder=0, lw=0)
        else:
            ax.axvspan(lo, hi, color=c, alpha=alpha, zorder=0, lw=0)
        if lab:
            if axis == "y":
                ax.text(0.01, (lo + hi) / 2, lab, transform=ax.get_yaxis_transform(), fontsize=5.2,
                        va="center", color=tcl_color(role, "text"), alpha=0.9)
            else:
                ax.text((lo + hi) / 2, 0.98, lab, transform=ax.get_xaxis_transform(), fontsize=5.2,
                        ha="center", va="top", color=tcl_color(role, "text"), alpha=0.9)


def tcl_curve(ax, x, mean, lo=None, hi=None, role="ours", label=None, tone="stroke",
              marker="o", end_label=True, hollow=False, lw=1.3, ms=2.8):
    """Line with optional band, stroke tone for the line, fill tone for the band, and the
    series name written at the right end (direct labels: CNS 18%, CCF 4%)."""
    c = tcl_color(role, tone)
    if lo is not None and hi is not None:
        ax.fill_between(x, lo, hi, color=tcl_color(role, "fill"), alpha=0.45, lw=0, zorder=1)
    ax.plot(x, mean, color=c, lw=lw, marker=marker, ms=ms, zorder=3,
            mfc="white" if hollow else c, mec=c, mew=0.8, label=label)
    if end_label and label:
        ax.annotate(label, (x[-1], mean[-1]), xytext=(4, 0), textcoords="offset points",
                    fontsize=5.6, color=tcl_color(role, "text"), va="center")


def tcl_box_points(ax, groups, roles, labels, positions=None, width=0.55, paired=False,
                   jitter=0.08, seed=0, point_size=7):
    """Box with raw points on top (distribution shown, not just a summary).

    Variability appears on 42% of CNS data panels against 23% of CCF; raw points on 5%
    vs 0.3%. paired=True joins the k-th point across groups (same seed / ordering).
    """
    rng = np.random.default_rng(seed)
    pos = np.arange(len(groups)) if positions is None else np.asarray(positions, float)
    xs = []
    for p, g, r in zip(pos, groups, roles):
        g = np.asarray(g, float)
        bp = ax.boxplot(g, positions=[p], widths=width, patch_artist=True, showfliers=False,
                        medianprops=dict(color=tcl_color(r, "text"), lw=1.1))
        for b in bp["boxes"]:
            b.set(fc=tcl_color(r, "fill"), ec=tcl_color(r, "stroke"), lw=0.7)
        for part in ("whiskers", "caps"):
            for l in bp[part]:
                l.set(color=tcl_color(r, "stroke"), lw=0.7)
        xj = p + rng.uniform(-jitter, jitter, len(g))
        xs.append((xj, g))
        ax.scatter(xj, g, s=point_size, fc="white", ec=tcl_color(r, "stroke"), lw=0.6, zorder=4)
    if paired and len(xs) > 1:
        n = min(len(v[1]) for v in xs)
        for k in range(n):
            ax.plot([v[0][k] for v in xs], [v[1][k] for v in xs], color=TCL_NEUTRAL["light"], lw=0.5, zorder=2)
    ax.set_xticks(pos)
    ax.set_xticklabels(labels)
    return xs


def tcl_dumbbell(ax, a, b, names, role_a="failing_default", role_b="ours",
                 label_a=None, label_b=None, horizontal=True, size=26):
    """Paired comparison of two states per item (before/after, current/earlier tasks)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    idx = np.arange(len(names))
    for i in idx:
        if horizontal:
            ax.plot([a[i], b[i]], [i, i], color=TCL_NEUTRAL["light"], lw=2.2, zorder=1, solid_capstyle="round")
        else:
            ax.plot([i, i], [a[i], b[i]], color=TCL_NEUTRAL["light"], lw=2.2, zorder=1, solid_capstyle="round")
    xa, ya = (a, idx) if horizontal else (idx, a)
    xb, yb = (b, idx) if horizontal else (idx, b)
    ax.scatter(xa, ya, s=size, fc=tcl_color(role_a, "fill"), ec=tcl_color(role_a, "stroke"), lw=0.8, zorder=3, label=label_a)
    ax.scatter(xb, yb, s=size, fc=tcl_color(role_b, "fill"), ec=tcl_color(role_b, "stroke"), lw=0.8, zorder=3, label=label_b)
    if horizontal:
        ax.set_yticks(idx); ax.set_yticklabels(names)
    else:
        ax.set_xticks(idx); ax.set_xticklabels(names)


def tcl_coverage_matrix(ax, M, row_labels=None, col_labels=None, nominal=0.90, vmin=0.0, vmax=1.0,
                        annotate=True, fmt="{:.2f}", cbar=True, cbar_label=None):
    """Per-task (or per-class) coverage matrix on the diverging map centred AT THE
    NOMINAL LEVEL, not the data midpoint: below nominal reads pink, above reads blue."""
    M = np.asarray(M, float)
    im = ax.imshow(M, cmap=tcl_cmap("div_coverage"), norm=TwoSlopeNorm(vcenter=nominal, vmin=vmin, vmax=vmax),
                   aspect="auto", interpolation="nearest")
    if annotate and M.size <= 220:
        for (r, c), v in np.ndenumerate(M):
            if np.isnan(v):
                continue
            s = fmt.format(v)
            s = s[1:] if s.startswith("0.") else s
            dark = v < vmin + 0.35 * (nominal - vmin) or v > nominal + 0.7 * (vmax - nominal)
            ax.text(c, r, s, ha="center", va="center", fontsize=4.8,
                    color="white" if dark else TCL_NEUTRAL["ink"])
    if row_labels is not None:
        ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels)
    elif row_labels is None:
        ax.set_yticks([])
    if col_labels is not None:
        ax.set_xticks(range(len(col_labels))); ax.set_xticklabels(col_labels)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = None
    if cbar:
        # inset, outside the right edge: leaves the panel's grid cell untouched so the
        # layout and alignment are unaffected; reserve one empty grid column for it.
        if cbar == "below":
            # horizontal key under the tick labels: no horizontal space taken from the
            # neighbouring panel, which is what collides in dense CNS rows
            cax = ax.inset_axes([0.0, -0.17, 1.0, 0.045])
            cb = ax.figure.colorbar(im, cax=cax, orientation="horizontal")
        else:
            cax = ax.inset_axes([1.02, 0.0, 0.035, 1.0])
            cb = ax.figure.colorbar(im, cax=cax)
        cb.set_ticks(sorted({vmin, nominal, vmax}))
        cb.ax.tick_params(labelsize=5, length=2)
        cb.outline.set_linewidth(0.5)
        if cbar_label:
            cb.set_label(cbar_label, fontsize=5.5)
    return im, cb


def tcl_violin_points(ax, groups, roles, labels, width=0.7, seed=0, point_size=5):
    """Half-transparent violin, box-free, with jittered raw points and a median tick."""
    rng = np.random.default_rng(seed)
    pos = np.arange(len(groups))
    parts = ax.violinplot([np.asarray(g, float) for g in groups], positions=pos, widths=width,
                          showextrema=False, showmedians=False)
    for body, r in zip(parts["bodies"], roles):
        body.set_facecolor(tcl_color(r, "fill")); body.set_edgecolor(tcl_color(r, "stroke"))
        body.set_alpha(0.85); body.set_linewidth(0.7)
    for p, g, r in zip(pos, groups, roles):
        g = np.asarray(g, float)
        ax.scatter(p + rng.uniform(-0.07, 0.07, len(g)), g, s=point_size, fc="white",
                   ec=tcl_color(r, "stroke"), lw=0.5, zorder=3)
        ax.plot([p - 0.16, p + 0.16], [np.median(g)] * 2, color=tcl_color(r, "text"), lw=1.4, zorder=4)
    ax.set_xticks(pos); ax.set_xticklabels(labels)


def tcl_overlaps(fig):
    """Text boxes that overlap each other (tick labels of the same axes excluded)."""
    r = fig.canvas.get_renderer()
    tx = [(t, t.get_window_extent(r)) for t in fig.findobj(mpl.text.Text) if t.get_text().strip() and t.get_visible()]
    own = {id(t) for a in fig.axes for t in a.get_xticklabels() + a.get_yticklabels()}
    return [(a.get_text()[:24], b.get_text()[:24]) for i, (a, ba) in enumerate(tx)
            for b, bb in tx[i + 1:] if ba.overlaps(bb) and not (id(a) in own and id(b) in own)]


def tcl_audit(fig, chart_types, is_first=False, shows_variability=None, verbose=True):
    """Score a figure against the measured norms.

    chart_types: list, one entry per panel, e.g. ["placeholder", "curve", "box", "matrix"].
    Placeholders count as schematic panels. Returns dict with findings.
    """
    n = len(chart_types)
    kinds = {c for c in chart_types}
    data_kinds = kinds - {"placeholder", "schematic"}
    lo, hi = (2.5, 5.5) if is_first else (3, 9)
    f = []
    if n < lo:
        f.append(f"{n} panels: below the CNS interquartile range for this position ({lo}-{hi})")
    if n > (hi + 3):
        f.append(f"{n} panels: well above the CNS upper quartile ({hi}); consider splitting")
    if len(kinds) < 2:
        f.append("one chart type only: 77% of CCF figures do this, 42% of CNS figures do")
    if is_first and not (kinds & {"placeholder", "schematic"}):
        f.append("Figure 1 without a schematic: 92% of CNS Figure 1s have one")
    if shows_variability is False:
        f.append("no variability shown: 42% of CNS data panels show error bars, bands, boxes or raw points")
    ov = tcl_overlaps(fig)
    if ov:
        f.append(f"{len(ov)} overlapping text pairs, e.g. {ov[:2]}")
    letters = [t for t in fig.texts if t.get_gid() == TCL_LETTER_GID]
    if len(letters) < n:
        f.append(f"{len(letters)} panel letters for {n} panels")
    out = {"n_panels": n, "n_types": len(kinds), "n_data_types": len(data_kinds), "findings": f}
    if verbose:
        print(f"{n} panels, {len(kinds)} chart types" + ("" if f else " - within corpus norms"))
        for x in f:
            print("  -", x)
    return out


def tcl_finalize(fig, axd, letter_size=8, title_size=None, gap_pt=2.5, letter_pad_pt=3.0,
                 case="lower", align_labels=True):
    """Align panel letters and panel titles on shared rows and columns (CNS convention).

    * Every panel in the same grid row gets its letter AND its title on one baseline,
      placed just above the tallest axes box in that row.
    * Every panel starting in the same grid column gets its letter at one x, just left
      of the widest y-tick/y-label extent among those panels.
    * The title starts at the axes' left edge, the letter sits left of it; titles are
      moved off the axes (ax.set_title is cleared) so tick labels can't push them.
    * y-labels are aligned per column and x-labels per row.
    Call once, after every panel is drawn and before saving. Returns nothing.
    """
    if title_size is None:
        title_size = mpl.rcParams["axes.titlesize"]
    if align_labels:
        fig.align_ylabels([a for a in axd.values() if a.axison])
        fig.align_xlabels([a for a in axd.values() if a.axison])
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    W, H = fig.get_size_inches() * 72.0
    gap_y, pad_x = gap_pt / H, letter_pad_pt / W
    info = {}
    for k, ax in axd.items():
        if not ax.get_visible():
            continue
        cells = getattr(fig, "_tcl_cells", {})
        if k in cells:
            r0, c0 = cells[k]
        else:
            ss = ax.get_subplotspec()
            r0, c0 = ss.rowspan.start, ss.colspan.start
        pos = ax.get_position()
        if ax.axison:
            tb = ax.get_tightbbox(r, bbox_extra_artists=[])
            # tight bbox includes the title; measure without it
            t_old = ax.get_title(loc="left") or ax.get_title()
            ax.set_title(""); ax.set_title("", loc="left")
            tb = ax.get_tightbbox(r)
            left = inv.transform((tb.x0, 0))[0]
        else:
            t_old = ax.get_title(loc="left") or ax.get_title()
            ax.set_title(""); ax.set_title("", loc="left")
            left = pos.x0
        info[k] = dict(ax=ax, row=r0, col=c0, pos=pos, left=left, title=t_old)
    rows, cols = {}, {}
    for k, d in info.items():
        rows.setdefault(d["row"], []).append(k)
        cols.setdefault(d["col"], []).append(k)
    def _top(d):
        # marginal / inset axes drawn above the main box (joint plots) count as the panel top
        tops = [d["pos"].y1] + [c.get_position().y1 for c in d["ax"].child_axes
                                if c.get_visible() and c.get_position().y1 > d["pos"].y1]
        return max(tops)
    base = {rw: max(_top(info[k]) for k in ks) + gap_y for rw, ks in rows.items()}
    lx = {c: min(info[k]["left"] for k in ks) - pad_x for c, ks in cols.items()}
    for k, d in info.items():
        y = base[d["row"]]
        t = fig.text(lx[d["col"]], y, k.upper() if case == "upper" else k.lower(), fontsize=letter_size,
                     fontweight="bold", ha="right", va="baseline", color=TCL_NEUTRAL["ink"])
        t.set_gid(TCL_LETTER_GID)
        if d["title"]:
            tt = fig.text(d["pos"].x0, y, d["title"], fontsize=title_size, ha="left", va="baseline",
                          color=TCL_NEUTRAL["ink"])
            tt.set_gid(TCL_TITLE_GID)


def tcl_alignment_report(fig):
    """Spread (pt) of letter/title baselines per row and letter x per column: all ~0 when aligned."""
    L = [t for t in fig.texts if t.get_gid() == TCL_LETTER_GID]
    T = [t for t in fig.texts if t.get_gid() == TCL_TITLE_GID]
    W, H = fig.get_size_inches() * 72.0
    ys = sorted({round(t.get_position()[1], 4) for t in L})
    xs = sorted({round(t.get_position()[0], 4) for t in L})
    rows = {y: [t for t in L + T if abs(t.get_position()[1] - y) < 1e-3] for y in ys}
    return {"letter_rows": len(ys), "letter_columns": len(xs), "n_letters": len(L), "n_titles": len(T),
            "row_baseline_spread_pt": max((max(t.get_position()[1] for t in v) - min(t.get_position()[1] for t in v)) * H
                                          for v in rows.values()) if rows else 0.0}


def tcl_end_labels(ax, items, min_gap=None, dx_pt=3.0, fontsize=5.6):
    """Series names at the right end of lines, nudged apart so they never overlap.

    items: list of (x_end, y_end, text, role). min_gap defaults to 7% of the y range.
    """
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * 0.10 if min_gap is None else min_gap
    its = sorted(items, key=lambda t: t[1])
    ys = [t[1] for t in its]
    for i in range(1, len(ys)):
        if ys[i] - ys[i - 1] < gap:
            ys[i] = ys[i - 1] + gap
    over = ys[-1] - (hi - gap * 0.3) if ys else 0
    if over > 0:
        ys = [y - over for y in ys]
    for (x, y0, txt, role), y in zip(its, ys):
        ax.annotate(txt, (x, y0), xytext=(dx_pt, (y - y0) / (hi - lo) * ax.bbox.height * 72 / ax.figure.dpi),
                    textcoords="offset points", fontsize=fontsize, va="center",
                    color=tcl_color(role, "text"))


def tcl_title_clashes(fig, pad_pt=2.0):
    """Panel titles that run into the next panel letter on the same row. CNS titles are
    short (dataset or setting); a clash means the title must be shortened, not the
    layout loosened."""
    r = fig._get_renderer()
    L = [t for t in fig.texts if t.get_gid() == TCL_LETTER_GID]
    Tt = [t for t in fig.texts if t.get_gid() == TCL_TITLE_GID]
    out = []
    for t in Tt:
        bt = t.get_window_extent(r)
        for l in L:
            bl = l.get_window_extent(r)
            same_row = abs(bl.y0 - bt.y0) < 3
            if same_row and bl.x0 > bt.x0 and bt.x1 + pad_pt > bl.x0:
                out.append((t.get_text(), l.get_text()))
    return out
