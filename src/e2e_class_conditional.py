"""Class-conditional validity in the END-TO-END arm (ResNet-18 trained from scratch).

Why this rather than a second frozen backbone
---------------------------------------------
The frozen-backbone class-conditional arm runs on one feature extractor. A second
*pretrained* backbone could not be obtained (every candidate's weights fail at the
transport layer in this environment), so the robustness question is answered in a
different and arguably stronger way: by repeating the comparison in the regime the
paper already studies separately, where the network is trained from scratch and the
representation is learned rather than frozen. This changes both the architecture
(convolutional rather than transformer) and the representation regime (learned rather
than pretrained-and-frozen).

It also probes the one place the paper's claims are already known to be
regime-dependent: Section 3.11 finds that disproportionality does NOT reproduce
end-to-end. If the class-conditional result does reproduce here, that is evidence it
is not an artefact of the frozen-feature setting.

Budget
------
The proposition needs k >= ceil(1/alpha) - 1 = 9 reserved slots per class for validity
and k >= 19 before the per-class quantile stops being the maximum of the reserved
scores. The paper's default of 5 is far below both, so this experiment uses 20
reserved slots on top of 20 replay slots. Replay is held fixed so that nothing but the
thresholding rule differs between the two reserved-slot schemes.

Three schemes, one trained model
--------------------------------
Every scheme is applied to the SAME trained network at the end of the stream, so any
difference is attributable to the calibration set and the thresholding rule alone.
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
import reliability as RL  # noqa: E402

ALPHA = 0.10
N_TASKS = 10
REPLAY_SLOTS = int(os.environ.get("REPLAY_SLOTS", "20"))
CAL_SLOTS = int(os.environ.get("CAL_SLOTS", "20"))
EPOCHS = int(os.environ.get("EPOCHS", "150"))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2,3,4").split(",")]
TAG = os.environ.get("TAG", "e2e_class_conditional")


def run_seed(E, M, seed):
    """One class-incremental run; evaluate three calibration schemes at the end."""
    (Xtr, ytr), (Xte, yte) = E.cifar100_tensors()
    ytr_np, yte_np = ytr.numpy(), yte.numpy()
    n_classes = int(ytr_np.max()) + 1
    tasks = M.class_incremental_split(ytr_np, N_TASKS, seed=seed)

    model = E.E2EBase(n_classes, seed=seed, epochs=EPOCHS)
    model.name = "E2E-RBCC-M"
    rng = np.random.default_rng(seed + 991)
    mem_idx, cal_idx = [], []

    for step, tc in enumerate(tasks):
        # Reserved slots are drawn BEFORE this task's training pass and withheld from
        # it, which is assumption (i) of the proposition. This arm manages its own
        # reservation rather than delegating to cl_methods.Replay, so the fix has to
        # be applied here too: an earlier version trained on the full task and then
        # reserved from the same points, which biases the reserved scores low and made
        # the per-class rule undercover (0.8664 against a nominal 0.90). Slots are
        # sampled uniformly at random, never by a selection rule that prefers
        # prototypical points — see Section 2.5.
        step_cal = []
        for c in tc:
            idx = np.where(ytr_np == c)[0]
            pick = rng.choice(idx, size=min(REPLAY_SLOTS + CAL_SLOTS, len(idx)),
                              replace=False)
            step_cal += list(pick[:CAL_SLOTS])
            mem_idx += list(pick[CAL_SLOTS:])
        cal_idx += step_cal

        tr = np.setdiff1d(np.where(np.isin(ytr_np, tc))[0],
                          np.array(step_cal, dtype=int), assume_unique=False)
        assert not (set(tr.tolist()) & set(step_cal)), "training pass sees reserved slots"
        assert not (set(mem_idx) & set(cal_idx)), "replay and calibration overlap"
        Xm = ym = None
        if mem_idx:
            Xm, ym = Xtr[np.array(mem_idx)], ytr[np.array(mem_idx)]
        model.observe(Xtr[tr], ytr[tr], tc, Xmem=Xm, ymem=ym)
        print(f"    seed={seed} step={step} done {time.strftime('%H:%M:%S')}", flush=True)

    seen = np.concatenate(tasks)
    half_by_task = {t: np.where(np.isin(yte_np, tcl))[0] for t, tcl in enumerate(tasks)}
    # Half of each task's test split calibrates the oracle; the other half evaluates
    # every scheme, so no scheme is scored on points it was calibrated on.
    cal_pool = np.concatenate([v[:len(v) // 2] for v in half_by_task.values()])
    ev = np.concatenate([v[len(v) // 2:] for v in half_by_task.values()])

    model.fit_temperature(Xte[cal_pool], yte[cal_pool])
    P = model.probs(Xte)
    Pc = model.probs(Xtr[np.array(cal_idx)])
    yc = ytr_np[np.array(cal_idx)]

    s_oracle = RL.lac_score(P[cal_pool], yte_np[cal_pool])
    s_res = RL.lac_score(Pc, yc)

    # The deployment-mixture stress test: re-weight the evaluation set toward the
    # classes learned first. The oracle is calibrated to the uniform mixture; a
    # per-class threshold has no mixture to be calibrated to.
    # coverage_under_mixture takes a PER-CLASS weight vector of length n_classes, not
    # a per-sample mask. An earlier version passed per-sample masks: inside the
    # function n_classes was then read as len(weights) (~5000), the first 100 entries
    # were taken, and because `ev` is ordered by task those entries were all 1 for
    # both mixtures — so cov_uniform and cov_old100 came out byte-identical and the
    # shift test measured nothing. Weights now follow the frozen-arm convention: the
    # oldest half of the classes carry all the mass under the shifted mixture.
    old_cls = set(np.concatenate(tasks[: len(tasks) // 2]).tolist())
    w_old = np.array([1.0 if c in old_cls else 0.0 for c in range(n_classes)], float)
    w_uni = np.full(n_classes, 0.5)
    assert len(w_old) == n_classes and w_old.sum() > 0, (len(w_old), w_old.sum())

    out = []
    schemes = {
        "pooled_oracle": ("pooled", RL.conformal_quantile(s_oracle, ALPHA)),
        "rbcc_pooled": ("pooled", RL.conformal_quantile(s_res, ALPHA)),
        "rbcc_mondrian": ("mondrian", RL.mondrian_quantiles(s_res, yc, ALPHA, n_classes)[0]),
    }
    nfb = int(np.isnan(schemes["rbcc_mondrian"][1]).sum()) if schemes["rbcc_mondrian"][1] is not None else 0
    for name, (kind, q) in schemes.items():
        S = (RL.predict_sets_mondrian(P[ev], q, "lac") if kind == "mondrian"
             else RL.predict_sets(P[ev], q))
        per, _, _ = RL.class_conditional_coverage(S, yte_np[ev], n_classes)
        fin = per[~np.isnan(per)]
        out.append(dict(
            scheme=name, seed=seed, replay_slots=REPLAY_SLOTS, k=CAL_SLOTS,
            epochs=EPOCHS, backbone="resnet18_e2e",
            marginal=float(RL.coverage(S, yte_np[ev])),
            mean_class=float(fin.mean()), worst_class=float(fin.min()),
            frac_classes_ok=float((fin >= 1 - ALPHA - 1e-9).mean()),
            size=float(S.sum(1).mean()),
            acc=float((P[ev].argmax(1) == yte_np[ev]).mean()),
            cov_uniform=float(RL.coverage_under_mixture(S, yte_np[ev], w_uni)),
            cov_old100=float(RL.coverage_under_mixture(S, yte_np[ev], w_old)),
            n_fallback_classes=nfb,
            # Keep the per-class coverages so any future deployment mixture can be
            # evaluated without re-running the stream.
            per_class=";".join(f"{v:.4f}" if not np.isnan(v) else "" for v in per)))
    return out


def main():
    import e2e_cl as E
    import cl_methods as M
    rows = []
    for seed in SEEDS:
        t0 = time.time()
        rows += run_seed(E, M, seed)
        print(f"  seed={seed} {time.time() - t0:.0f}s", flush=True)
        import pandas as pd
        pd.DataFrame(rows).to_csv(f"{TAG}.csv", index=False)
    import pandas as pd
    D = pd.DataFrame(rows)
    D.to_csv(f"{TAG}.csv", index=False)
    g = D.groupby("scheme")[["acc", "mean_class", "marginal", "size",
                             "cov_uniform", "cov_old100", "worst_class"]].mean()
    print()
    print(g.round(4).to_string())
    # The experiment is only informative if the schemes actually differ. A silent
    # tie across all three would mean the reserved slots never took effect.
    assert D.groupby("scheme")["size"].mean().nunique() > 1, "all schemes identical"
    print(f"\n{len(D)} rows -> {TAG}.csv  (replay={REPLAY_SLOTS}, k={CAL_SLOTS}, "
          f"epochs={EPOCHS}, seeds={SEEDS})")


if __name__ == "__main__":
    main()
