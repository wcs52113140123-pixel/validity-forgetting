"""Frozen-backbone versus end-to-end: does the coverage loss survive full training?

The paper's 280 runs all train a linear head on features from a frozen pre-trained
backbone — the standard PTM-CIL protocol. The obvious objection is that the coverage
collapse might be a property of linear heads on cached features rather than of
continual learning. This script answers it by putting the two settings side by side on
the same benchmark, the same class split, the same conformal machinery and the same
four calibration protocols. Only the trainable surface differs.

What is and is not comparable
-----------------------------
* **Comparable**: the *shape* of the failure — current-task coverage against
  earlier-task coverage, and the ordering of the four calibration protocols. Both arms
  promise the same nominal level on the same benchmark, so a coverage gap means the
  same thing in each.
* **Not comparable**: absolute accuracy. The frozen arm rides a backbone pre-trained on
  ImageNet-21k or LVD-142M; the end-to-end arm is a ResNet-18 trained from scratch on
  CIFAR-100 alone. The frozen arm will be more accurate and that says nothing about
  either setting's validity.
* **Method names differ by design.** The end-to-end arm runs four methods
  (Seq-FT, LwF, Replay, RBCC); EWC, iCaRL, NCM and RanPAC are frozen-feature methods
  whose end-to-end analogues are different algorithms, so they are not transplanted.

Reads e2e_records.csv (end-to-end, this study) and sweep_records.csv (frozen, the
paper's main matrix) and writes a long-format comparison plus the headline contrast.
"""
import numpy as np
import pandas as pd

NOMINAL = 0.90
PAIRS = {"E2E-Seq-FT": "Seq-FT", "E2E-LwF": "LwF",
         "E2E-Replay": "Replay", "E2E-RBCC": "Replay"}


def final_step(d):
    """Rows describing the FINAL model, i.e. after the last incremental step."""
    return d[d.step == d.step.max()]


def arm_summary(d, arm, backbone_filter=None):
    """Per (method, protocol): current-task vs earlier-task coverage of the final model."""
    f = final_step(d)
    if backbone_filter is not None:
        f = f[f.backbone == backbone_filter]
    last = f.step.max()
    f = f.assign(which=np.where(f.task == last, "current", "earlier"))
    g = (f.groupby(["method", "protocol", "which"])
           .agg(cov=("cov", "mean"), acc=("acc", "mean"), size=("size", "mean"),
                n=("cov", "size"))
           .reset_index())
    g["arm"] = arm
    return g


def contrast(E, S, backbone="vit_b16_in21k"):
    """Match each end-to-end method to its frozen counterpart on protocol A."""
    e = arm_summary(E, "end-to-end")
    s = arm_summary(S[S.dataset == "cifar100"], "frozen", backbone_filter=backbone)
    rows = []
    for em, sm in PAIRS.items():
        for prot in ("A_current", "B_pooled", "C_rbcc", "Cn_replayed"):
            ee = e[(e.method == em) & (e.protocol == prot)]
            ss = s[(s.method == sm) & (s.protocol == prot)]
            if ee.empty or ss.empty:
                continue
            def pick(df, w, col="cov"):
                r = df[df.which == w]
                return float(r[col].iloc[0]) if len(r) else np.nan
            rows.append(dict(
                method=em, frozen_counterpart=sm, protocol=prot,
                e2e_cov_current=pick(ee, "current"), e2e_cov_earlier=pick(ee, "earlier"),
                frozen_cov_current=pick(ss, "current"), frozen_cov_earlier=pick(ss, "earlier"),
                e2e_acc_earlier=pick(ee, "earlier", "acc"),
                frozen_acc_earlier=pick(ss, "earlier", "acc"),
                e2e_gap=NOMINAL - pick(ee, "earlier"),
                frozen_gap=NOMINAL - pick(ss, "earlier")))
    return pd.DataFrame(rows)


def verdict(C):
    """State plainly whether the end-to-end arm reproduces the frozen-arm failure.

    Written so that a negative result — the collapse does NOT survive full training —
    is reported as such rather than hedged. That outcome narrows the paper's claim to
    the PTM-CIL setting, which is a finding, not a failure.
    """
    a = C[C.protocol == "A_current"]
    if a.empty:
        return "protocol A rows missing — cannot judge"
    out = []
    out.append(f"protocol A, nominal {NOMINAL:.2f}, {len(a)} matched method pairs")
    out.append(f"  end-to-end: current {a.e2e_cov_current.mean():.4f} | "
               f"earlier {a.e2e_cov_earlier.mean():.4f} | gap {a.e2e_gap.mean():.4f}")
    out.append(f"  frozen    : current {a.frozen_cov_current.mean():.4f} | "
               f"earlier {a.frozen_cov_earlier.mean():.4f} | gap {a.frozen_gap.mean():.4f}")
    e_inv = (a.e2e_cov_current >= NOMINAL - 0.02).sum()
    f_inv = (a.frozen_cov_current >= NOMINAL - 0.02).sum()
    out.append(f"  deployment-invisibility (current task meets the level): "
               f"end-to-end {e_inv}/{len(a)}, frozen {f_inv}/{len(a)}")
    if a.e2e_gap.mean() > 0.10:
        out.append("  VERDICT: the collapse reproduces end-to-end — the claim is not "
                   "an artefact of linear heads on cached features.")
    elif a.e2e_gap.mean() > 0.03:
        out.append("  VERDICT: the collapse is present end-to-end but milder; report "
                   "both magnitudes and do not generalise the frozen-arm numbers.")
    else:
        out.append("  VERDICT: the collapse does NOT reproduce end-to-end. The paper's "
                   "claim must be narrowed to the frozen-backbone (PTM-CIL) setting, "
                   "and this negative result reported.")
    return "\n".join(out)
