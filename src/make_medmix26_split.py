"""Build the MedMix-26 benchmark split used in the paper.

The native MedMNIST train/test splits of the three source collections differ in class
proportions (organ CT is split by patient), which violates the exchangeability that split
conformal prediction assumes. We therefore pool each feature file's native train and test
sets and re-split them class-stratified at the native test fraction (seed 0).

    python make_medmix26_split.py features/medmix26_native__vit_b16_in21k.npz features/medmix26__vit_b16_in21k.npz
"""
import sys, json
import numpy as np


def resplit(path, out, seed=0):
    z = np.load(path)
    X = np.concatenate([z["Xtr"], z["Xte"]])
    y = np.concatenate([z["ytr"], z["yte"]])
    frac = len(z["yte"]) / len(y)
    rng = np.random.default_rng(seed)
    tr_i, te_i = [], []
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        nt = int(round(frac * len(idx)))
        te_i += list(idx[:nt])
        tr_i += list(idx[nt:])
    tr_i, te_i = np.sort(tr_i), np.sort(te_i)
    meta = json.loads(str(z["meta"]))
    meta["split"] = f"pooled native train+test, re-split class-stratified (test fraction {frac:.4f}, seed {seed})"
    np.savez(out, Xtr=X[tr_i], ytr=y[tr_i], Xte=X[te_i], yte=y[te_i], meta=json.dumps(meta))
    return len(tr_i), len(te_i)


if __name__ == "__main__":
    print(resplit(sys.argv[1], sys.argv[2]))
