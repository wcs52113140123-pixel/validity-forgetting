"""Rebuild the five figures (composites + every panel separately) from results/.

    VF_ROOT=/path/to/validity-forgetting  python src/figures/build_figures.py  [--fonts DIR]

Arial is required (Elsevier artwork rule). Pass --fonts with a folder holding the Arial
TTF files if Arial is not installed system-wide. Outputs go to figures/ and figures/panels/.
"""
import os, sys, argparse
import numpy as np
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)                                  # the prep scripts import the plotting modules from here
ap = argparse.ArgumentParser(); ap.add_argument("--fonts", default=None); ap.add_argument("--panels-only", action="store_true")
args = ap.parse_args()
ROOT = os.environ.setdefault("VF_ROOT", os.path.abspath(os.path.join(HERE, "..", "..")))
import matplotlib; matplotlib.use("Agg")
import journal_type as JT, panel_export as PE
JT.use_arial(args.fonts)                         # registers TTFs from --fonts (if given) and selects Arial
S = {k: Image.open(os.path.join(ROOT, "figures", "schematics", f)).convert("RGB")
     for k, f in [("S2", "S2_Fig3a.png"), ("S3", "S3_Fig4a.png"), ("S4", "S4_Fig5a.png")]}
NS = {}
for n in range(1, 6):
    ns = {"__name__": f"fig{n}"}
    exec(compile(open(os.path.join(HERE, f"fig{n}_prep.py")).read(), f"fig{n}_prep.py", "exec"), ns)
    NS[n] = ns
BUILD = {1: lambda: NS[1]["build_fig1"](),
         2: lambda: NS[2]["build_fig2"]()[:2],
         3: lambda: NS[3]["build_fig3n"](S["S2"])[:2],
         4: lambda: NS[4]["build_fig4h"]([1.35, 1, 1], 5.9, S["S3"]),
         5: lambda: NS[5]["build_fig5"]((1.5, 1), 4.25, S["S4"])}

def journal(n, reset):
    JT.use_arial(args.fonts)                     # the builders reset the style; re-select Arial
    fig, A = BUILD[n](); JT.enforce(fig, reset_label_positions=reset); return fig, A

out = os.path.join(ROOT, "figures"); os.makedirs(os.path.join(out, "panels"), exist_ok=True)
for n in range(1, 6):
    if not args.panels_only:
        fig, A = journal(n, reset=True)
        for t in fig.texts:
            if t.get_gid() == "tcl_panel_title": t.set_visible(False)
            if t.get_gid() == "tcl_panel_letter": t.set_fontsize(8.0); t.set_fontweight("bold")
        fig.savefig(os.path.join(out, f"Figure_{n}.pdf"), bbox_inches="tight", pad_inches=0.02, dpi=1000)
        fig.savefig(os.path.join(out, f"Figure_{n}.png"), bbox_inches="tight", pad_inches=0.02, dpi=600)
    fig, A = journal(n, reset=True)
    PE.export_panels(fig, A, f"Fig{n}", os.path.join(out, "panels", f"Figure_{n}"), dpi=1000,
                     formats=("pdf", "svg", "png"), keep_titles=False)
    print(f"Figure {n} done", flush=True)
