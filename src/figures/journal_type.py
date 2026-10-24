"""Bring a finished figure to Elsevier artwork rules: Arial throughout, 7 pt lettering
at final size (larger call-outs kept), line weights no thinner than 0.25 pt.

Applied after a builder has drawn the figure, so the same builders keep working; the
changes are made on the artists themselves, and tick labels through tick_params so
they survive re-draws.
"""
import matplotlib
import matplotlib.text as mtext
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.collections import Collection

BASE_PT = 7.0          # Elsevier: 7 pt for normal lettering at final size
CALLOUT_PT = 9.0       # text already at or above this is a deliberate call-out; left as is
MIN_LW = 0.25          # Elsevier: no line thinner than 0.25 pt
SKIP_GIDS = ("tcl_panel_letter", "tcl_panel_title")


def use_arial(font_dir="fonts"):
    import os
    from matplotlib import font_manager as fm
    for f in (sorted(os.listdir(font_dir)) if font_dir and os.path.isdir(font_dir) else []):
        if f.lower().endswith(".ttf"):
            fm.fontManager.addfont(os.path.join(font_dir, f))
    matplotlib.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial"],
        "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold", "mathtext.sf": "Arial",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})


def _all_axes(fig):
    out, stack = [], list(fig.axes)
    while stack:
        a = stack.pop()
        if a in out:
            continue
        out.append(a)
        stack.extend(getattr(a, "child_axes", []))
    return out


def enforce(fig, base=BASE_PT, callout=CALLOUT_PT, min_lw=MIN_LW, reset_label_positions=True):
    """Set every ordinary text to `base` pt Arial and floor line widths. Returns counts."""
    n_tick = n_text = n_lw = 0
    for ax in _all_axes(fig):
        ax.tick_params(axis="both", which="both", labelsize=base)
        n_tick += 1
    for t in fig.findobj(mtext.Text):
        if t.get_gid() in SKIP_GIDS or not t.get_text().strip():
            continue
        t.set_fontfamily("sans-serif")
        if t.get_fontsize() < callout and abs(t.get_fontsize() - base) > 1e-6:
            t.set_fontsize(base)
            n_text += 1
    for o in fig.findobj(lambda a: isinstance(a, (Line2D, Patch, Collection))):
        try:
            lw = o.get_linewidth()
        except Exception:
            continue
        if isinstance(lw, (list, tuple)) or hasattr(lw, "__len__"):
            lws = [float(v) for v in lw]
            if lws and min(v for v in lws) < min_lw and any(v > 0 for v in lws):
                o.set_linewidth([max(v, min_lw) if v > 0 else v for v in lws]); n_lw += 1
        elif lw and 0 < lw < min_lw:
            o.set_linewidth(min_lw); n_lw += 1
    if reset_label_positions:
        # composite-level alignment pinned axis labels to shared coordinates; a stand-alone
        # panel wants each label at its default distance from its own tick labels
        for ax in _all_axes(fig):
            for axis in (ax.xaxis, ax.yaxis):
                axis._autolabelpos = True
        import matplotlib.cbook as cbook
        fig._align_label_groups = {"x": cbook.Grouper(), "y": cbook.Grouper()}   # undo fig.align_labels
    fig.canvas.draw()
    return dict(axes=n_tick, texts_resized=n_text, lines_thickened=n_lw)


def text_report(fig):
    """Sizes and font files actually used by visible lettering (letters/titles excluded)."""
    from collections import Counter
    from matplotlib import font_manager as fm
    T = [t for t in fig.findobj(mtext.Text) if t.get_visible() and t.get_text().strip()
         and t.get_gid() not in SKIP_GIDS]
    files = Counter(fm.findfont(t.get_fontproperties()).split("/")[-1] for t in T)
    sizes = Counter(round(t.get_fontsize(), 2) for t in T)
    return dict(n=len(T), sizes=dict(sorted(sizes.items())), font_files=dict(files))


def panel_overlaps(fig, own, tol=0.5):
    """Pairs of visible texts inside one panel whose boxes intersect (in pt)."""
    r = fig.canvas.get_renderer()
    T = []
    for ax0 in own["axes"]:
        for ax in [ax0] + list(getattr(ax0, "child_axes", [])):
            if not ax.get_visible():
                continue
            cand = list(ax.texts)
            if ax.get_legend() is not None:
                cand += list(ax.get_legend().get_texts())
            if ax.axison:
                for axis in ax._axis_map.values():
                    if axis.get_visible():
                        cand.append(axis.label)
                        for tk in axis._update_ticks():   # ticks inside the view interval: what is drawn
                            cand += [tk.label1, tk.label2]
            T += [t for t in cand if t.get_visible() and t.get_text().strip()]
    T += [t for t in own["texts"] if isinstance(t, mtext.Text) and t.get_visible()]
    seen, B = set(), []
    for t in T:
        if id(t) in seen:
            continue
        seen.add(id(t))
        B.append((t, t.get_window_extent(r)))
    k = 72.0 / fig.dpi
    hits = []
    for i, (a, ba) in enumerate(B):
        for b, bb in B[i + 1:]:
            w = min(ba.x1, bb.x1) - max(ba.x0, bb.x0)
            h = min(ba.y1, bb.y1) - max(ba.y0, bb.y0)
            if w * k > tol and h * k > tol:
                hits.append((a.get_text()[:22], b.get_text()[:22], round(w * k, 1), round(h * k, 1)))
    return hits
