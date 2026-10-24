"""Assemble the released run records from the individual experiment outputs.

    python assemble_records.py <work_dir>

<work_dir> must contain the outputs of the steps in README.md (section "Reproducing"):
  sweep_records_sweep.csv          sweep.py, all seven methods on CIFAR-100, CUB-200, ImageNet-R
  sweep_records_ho.csv             run_main_memory_methods.py (Replay, iCaRL; slots reserved before training)
  sweep_records_medmix26s.csv      run_main_medmix26.py (all seven methods on MedMix-26)
  ksweep_bench_ho.csv              run_reservation_size.py
  ksweep_medmix26s.csv             run_reservation_size_medmix26.py
The memory methods from sweep.py are replaced by their reserved-before-training runs, and every
MedMix-26 run comes from the class-stratified split. Writes sweep_records.csv and
reservation_size_records.csv into <work_dir>.
"""
import os, sys
import pandas as pd

d = sys.argv[1]
rd = lambda f: pd.read_csv(os.path.join(d, f))
S, HO, MS = rd("sweep_records_sweep.csv"), rd("sweep_records_ho.csv"), rd("sweep_records_medmix26s.csv")
keep = S[(S.dataset != "medmix26") & ~S.method.isin(["Replay", "iCaRL"])]
R = pd.concat([keep, HO[HO.dataset != "medmix26"], MS], ignore_index=True)
assert R.groupby(["dataset", "backbone", "method", "seed"]).ngroups == 280, "expected 280 runs"
R.to_csv(os.path.join(d, "sweep_records.csv"), index=False)
A, B = rd("ksweep_bench_ho.csv"), rd("ksweep_medmix26s.csv")
pd.concat([A[A.dataset != "medmix26"], B], ignore_index=True).to_csv(os.path.join(d, "reservation_size_records.csv"), index=False)
print(len(R), "rows ->", os.path.join(d, "sweep_records.csv"))
