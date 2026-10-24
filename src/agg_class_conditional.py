"""Aggregate the replay-budget sweep, refusing to ship a run that did not vary.

A previous attempt swept the budget by string-substituting the experiment file on
the remote host. The pattern was written against a different revision, did not
match, and every setting silently ran the same configuration — the only symptom was
three byte-identical result tables. The budget is now a real parameter, and these
assertions make a repeat of that failure loud instead of plausible.
"""
import glob

import pandas as pd

fs = sorted(glob.glob("mondrian_replay*.csv"))
assert fs, "no per-budget tables were produced"
d = pd.concat([pd.read_csv(f) for f in fs])
d.to_csv("mondrian_forgetting.csv", index=False)

acc = d[d.scheme == "rbcc_mondrian"].groupby("replay_slots").acc.mean()
print(acc.round(4).to_string())
assert acc.nunique() == len(acc), f"accuracies identical across budgets: {acc.to_dict()}"
assert acc.max() - acc.min() > 0.005, f"replay budget barely moves accuracy: {acc.to_dict()}"

g = d.groupby(["replay_slots", "scheme"])[
    ["acc", "mean_class", "marginal", "size", "cov_old100", "cov_uniform"]].mean()
print()
print(g.round(4).to_string())
print()
for r in sorted(d.replay_slots.unique()):
    s = d[d.replay_slots == r]
    mo = s[s.scheme == "rbcc_mondrian"].mean_class.mean()
    po = s[s.scheme == "rbcc_pooled"].mean_class.mean()
    oc = s[s.scheme == "pooled_oracle"].mean_class.mean()
    a = s[s.scheme == "rbcc_mondrian"].acc.mean()
    print(f"replay={r:>3} acc={a:.4f} oracle={oc:.4f} rbcc-pooled={po:.4f} "
          f"rbcc-M={mo:.4f} gain={mo - po:+.4f}")
