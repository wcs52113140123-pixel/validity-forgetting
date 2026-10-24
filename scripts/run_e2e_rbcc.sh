#!/bin/bash
# End-to-end RBCC: ResNet-18 trained from scratch on CIFAR-100, 150 epochs per task,
# reserved slots drawn before training and withheld from it; five class orderings run in
# parallel on one GPU. Expects CIFAR-100 under src/data/cifar100 (see DATA.md).
set -e
cd "$(dirname "$0")/../src"
export MATRIX=full EPOCHS=150 METHODS_ONLY=E2E-RBCC OMP_NUM_THREADS=8 KMP_AFFINITY=disabled
for s in 0 1 2 3 4; do
  SEED_LIST=$s E2E_OUT=e2e_rbcc_seed$s.csv python e2e_cl.py > log_seed$s.txt 2>&1 &
done
wait
python - <<'PY'
import pandas as pd, glob
d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("e2e_rbcc_seed*.csv"))]); d.to_csv("e2e_rbcc_ho_150ep.csv", index=False)
assert sorted(d.seed.unique()) == [0, 1, 2, 3, 4]
PY
