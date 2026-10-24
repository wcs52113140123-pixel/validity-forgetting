"""Export every panel of a tcl_mosaic figure as its own file (PDF / SVG / PNG).

Each panel is rendered from the live figure, so sizes, fonts and colours are exactly
those of the composite at print scale (180 mm wide). Everything that belongs to other
panels is hidden before the crop, so neighbouring labels never leak into a panel file.
Panel letters are omitted (the user lays the figure out); panel titles are kept as
editable text.
"""
import numpy as np
from matplotlib.transforms import Bbox

LETTER_GID = "tcl_panel_letter"
TITLE_GID = "tcl_panel_title"


def _gap(a, b):
    """Distance between two Bboxes (0 when they overlap)."""
    dx = max(a.x0 - b.x1, b.x0 - a.x1, 0.0)
    dy = max(a.y0 - b.y1, b.y0 - a.y1, 0.0)
    return float(np.hypot(dx, dy))


def _centre_dist(a, b):
    return float(np.hypot((a.x0 + a.x1 - b.x0 - b.x1) / 2, (a.y0 + a.y1 - b.y0 - b.y1) / 2))


def assign_owners(fig, A):
    """Map every axes and figure-level artist to the panel key it belongs to."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    main = {k: ax.get_window_extent(r) for k, ax in A.items()}
    owners = {k: dict(axes=[A[k]], texts=[], letter=None) for k in A}
    main_axes = set(A.values())

    def nearest(bb):
        return min(main, key=lambda k: (round(_gap(bb, main[k]), 1), _centre_dist(bb, main[k])))

    for ax in fig.axes:                      # colorbars, marginals, side axes
        if ax in main_axes:
            continue
        owners[nearest(ax.get_window_extent(r))]["axes"].append(ax)
    for t in fig.texts:
        if not t.get_text().strip():
            continue
        bb = t.get_window_extent(r)
        if t.get_gid() == TITLE_GID:
            # a title sits directly above its panel and starts at the panel's left edge
            k = min(main, key=lambda k: (abs(bb.x0 - main[k].x0) > 3, max(main[k].y1 - bb.y0, 0) > 3,
                                         bb.y0 - main[k].y1 if bb.y0 >= main[k].y1 - 3 else 1e9))
            owners[k]["texts"].append(t)
        elif t.get_gid() == LETTER_GID:
            owners[nearest(bb)]["letter"] = t
        else:
            owners[nearest(bb)]["texts"].append(t)
    others = list(fig.legends) + list(fig.lines) + list(fig.patches) + list(fig.artists) + list(fig.images)
    for o in others:
        owners[nearest(o.get_window_extent(r))]["texts"].append(o)
    return owners


def panel_bbox_inches(fig, own, pad=0.025):
    r = fig.canvas.get_renderer()
    bbs = [ax.get_tightbbox(r) for ax in own["axes"]]
    bbs += [t.get_window_extent(r) for t in own["texts"] if t.get_visible()]
    bb = Bbox.union([b for b in bbs if b is not None and b.width > 0])
    bb = bb.transformed(fig.dpi_scale_trans.inverted())
    return Bbox.from_extents(bb.x0 - pad, bb.y0 - pad, bb.x1 + pad, bb.y1 + pad)


def export_panels(fig, A, stem, outdir, dpi=600, formats=("pdf", "svg", "png"), keep_titles=True):
    """Write <outdir>/<stem><k>.<fmt> for every panel; return one manifest row per panel."""
    import os
    import matplotlib as mpl
    os.makedirs(outdir, exist_ok=True)
    owners = assign_owners(fig, A)
    every = list(fig.axes) + [t for t in fig.texts] + list(fig.legends) + list(fig.lines) + \
        list(fig.patches) + list(fig.artists) + list(fig.images)
    vis0 = {id(o): o.get_visible() for o in every}
    rows = []
    with mpl.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"}):
        for k in sorted(A):
            own = owners[k]
            keep = set(map(id, own["axes"] + (own["texts"] if keep_titles else
                                                [t for t in own["texts"] if getattr(t, "get_gid", lambda: None)() != TITLE_GID])))
            for o in every:
                o.set_visible(vis0[id(o)] and id(o) in keep)
            fig.canvas.draw()
            bb = panel_bbox_inches(fig, own)
            files = []
            for fmt in formats:
                p = os.path.join(outdir, f"{stem}{k}.{fmt}")
                fig.savefig(p, dpi=dpi, bbox_inches=bb, pad_inches=0, facecolor="white")
                files.append(os.path.basename(p))
            title = next((t.get_text() for t in own["texts"] if getattr(t, "get_gid", lambda: None)() == TITLE_GID), "")
            rows.append(dict(panel=f"{stem}{k}", title=title.replace("\n", " "),
                             width_mm=round(bb.width * 25.4, 1), height_mm=round(bb.height * 25.4, 1),
                             n_axes=len(own["axes"]), files=" ".join(files)))
    for o in every:
        o.set_visible(vis0[id(o)])
    fig.canvas.draw()
    return rows
