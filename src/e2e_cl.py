"""End-to-end class-incremental learning, to answer the one structural objection.

Everything in the paper so far runs a linear head on features from a FROZEN backbone —
the standard PTM-CIL protocol. A reviewer at any venue will ask the obvious question:
is the coverage collapse a property of continual learning, or a property of linear
heads on cached features? This script answers it by letting the whole backbone train.

Design, and why each choice is the conservative one
---------------------------------------------------
* **Same benchmark, same split, same conformal machinery.** CIFAR-100, ten tasks, the
  class order from `class_incremental_split`, the reliability module from
  `reliability.py`. Only the trainable surface changes, so a difference in outcome is
  attributable to that and nothing else.
* **Same four calibration protocols.** A_current / B_pooled / C_rbcc / Cn_replayed are
  computed exactly as in `experiment.py`, so the protocol comparison transfers.
* **The frozen-backbone result is the hypothesis under test, not the expected answer.**
  If coverage holds up end-to-end, the paper's claim narrows to the PTM-CIL setting and
  we say so. That outcome is reported either way.
* **ResNet-18 from scratch, not a pre-trained ViT.** End-to-end fine-tuning of a
  pre-trained backbone on ten small tasks collapses into the frozen case for the early
  tasks (the features barely move) and would not isolate the effect. Training from
  scratch is the setting the continual-learning literature's own numbers come from, so
  the comparison to published forgetting curves stays meaningful.

Runtime
-------
Single GPU. Seven methods x ten tasks x five seeds at 40 epochs per task is the full
matrix; `MATRIX=smoke` runs one seed and 5 epochs first so a configuration error costs
minutes rather than a day.
"""
import json
import os
import time

os.environ.setdefault("KMP_AFFINITY", "disabled")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import cl_methods as M
import reliability as RL

DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_TASKS = 10
ALPHA = 0.10
MEM_PER_CLASS = 20
CAL_PER_CLASS = 5
OUT = os.environ.get("E2E_OUT", "e2e_records.csv")


# --------------------------------------------------------------------------- data

def cifar100_tensors(root="data/cifar100/cifar-100-python"):
    """Load the canonical pickled CIFAR-100 into uint8 tensors.

    Kept dependency-free so that a local copy of the official archive suffices (no
    download at run time). The pickle layout is the published one (dict with b'data' as N x 3072 and b'fine_labels').
    """
    import pickle

    def load(split):
        with open(os.path.join(root, split), "rb") as fh:
            d = pickle.load(fh, encoding="bytes")
        X = d[b"data"].reshape(-1, 3, 32, 32)
        y = np.asarray(d[b"fine_labels"], dtype=np.int64)
        return torch.from_numpy(X), torch.from_numpy(y)

    return load("train"), load("test")


MEAN = torch.tensor([0.5071, 0.4865, 0.4409]).view(1, 3, 1, 1)
STD = torch.tensor([0.2673, 0.2564, 0.2762]).view(1, 3, 1, 1)


def normalise(xb):
    return (xb.float() / 255.0 - MEAN.to(xb.device)) / STD.to(xb.device)


def augment(xb):
    """Random crop with reflection padding plus horizontal flip, done on-device."""
    n = xb.shape[0]
    xb = F.pad(xb, (4, 4, 4, 4), mode="reflect")
    i = torch.randint(0, 9, (2,))
    xb = xb[:, :, i[0]:i[0] + 32, i[1]:i[1] + 32]
    flip = torch.rand(n, device=xb.device) < 0.5
    xb[flip] = torch.flip(xb[flip], dims=[3])
    return xb


# -------------------------------------------------------------------------- model

class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.short = (nn.Sequential() if stride == 1 and cin == cout else
                      nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False),
                                    nn.BatchNorm2d(cout)))

    def forward(self, x):
        h = F.relu(self.b1(self.c1(x)))
        h = self.b2(self.c2(h))
        return F.relu(h + self.short(x))


class ResNet18(nn.Module):
    """CIFAR-style ResNet-18: 3x3 stem, no max-pool, four stages of two blocks."""

    def __init__(self, n_classes, width=64):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, width, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(width), nn.ReLU(inplace=True))
        cfg = [(width, 1), (width * 2, 2), (width * 4, 2), (width * 8, 2)]
        layers, cin = [], width
        for cout, stride in cfg:
            layers += [BasicBlock(cin, cout, stride), BasicBlock(cout, cout, 1)]
            cin = cout
        self.body = nn.Sequential(*layers)
        self.feat_dim = cin
        self.head = nn.Linear(cin, n_classes)

    def features(self, x):
        h = self.body(self.stem(x))
        return F.adaptive_avg_pool2d(h, 1).flatten(1)

    def forward(self, x):
        return self.head(self.features(x))


# ------------------------------------------------------------------------ methods

class E2EBase:
    """Class-incremental training of the whole network, with a masked softmax.

    Logits of classes not yet seen are masked to -inf during training, which is the
    conventional class-incremental treatment and keeps the comparison to the
    frozen-feature results on the same footing.
    """

    name = "E2E-Seq-FT"
    uses_memory = False

    def __init__(self, n_classes, seed=0, epochs=40, lr=0.1, bs=128, wd=5e-4):
        torch.manual_seed(seed)
        self.net = ResNet18(n_classes).to(DEV)
        self.n_classes, self.epochs, self.lr, self.bs, self.wd = n_classes, epochs, lr, bs, wd
        self.seen = []
        self.rng = np.random.default_rng(seed)
        self.T = 1.0

    def _mask(self, logits):
        m = torch.full_like(logits, float("-inf"))
        idx = torch.tensor(self.seen, device=logits.device)
        m[:, idx] = logits[:, idx]
        return m

    def _epoch_batches(self, X, y):
        order = self.rng.permutation(len(y))
        for i in range(0, len(order), self.bs):
            j = order[i:i + self.bs]
            yield X[j].to(DEV, non_blocking=True), y[j].to(DEV, non_blocking=True)

    def _extra_loss(self, xb, logits):
        return 0.0

    def observe(self, Xtr, ytr, task_classes, Xmem=None, ymem=None):
        self.seen = sorted(set(self.seen) | set(int(c) for c in task_classes))
        opt = torch.optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9,
                              weight_decay=self.wd, nesterov=True)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=self.lr, total_steps=max(1, self.epochs * (len(ytr) // self.bs + 1)))
        self.net.train()
        for _ in range(self.epochs):
            for xb, yb in self._epoch_batches(Xtr, ytr):
                xb = augment(normalise(xb))
                if Xmem is not None and len(ymem):
                    k = min(len(ymem), len(yb))
                    j = self.rng.choice(len(ymem), size=k, replace=False)
                    xm = augment(normalise(Xmem[j].to(DEV)))
                    xb = torch.cat([xb, xm])
                    yb = torch.cat([yb, ymem[j].to(DEV)])
                logits = self._mask(self.net(xb))
                loss = F.cross_entropy(logits, yb) + self._extra_loss(xb, logits)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
        self._after_task()

    def _after_task(self):
        pass

    @torch.no_grad()
    def probs(self, X, bs=512):
        self.net.eval()
        out = []
        for i in range(0, len(X), bs):
            logits = self._mask(self.net(normalise(X[i:i + bs].to(DEV))))
            out.append(F.softmax(logits / self.T, dim=1).cpu().numpy())
        return np.concatenate(out)

    def fit_temperature(self, Xcal, ycal):
        """Post-hoc temperature scaling, the same opportunity every method gets."""
        with torch.no_grad():
            self.net.eval()
            z = []
            for i in range(0, len(Xcal), 512):
                z.append(self._mask(self.net(normalise(Xcal[i:i + 512].to(DEV)))).cpu())
            z = torch.cat(z)
        logT = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=60)
        yt = ycal.clone()

        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(z / logT.exp(), yt)
            loss.backward()
            return loss

        opt.step(closure)
        self.T = float(logT.exp().item())
        return self.T


class E2ELwF(E2EBase):
    name = "E2E-LwF"

    def __init__(self, *a, lam=1.0, Tdist=2.0, **kw):
        super().__init__(*a, **kw)
        self.lam, self.Tdist, self.old = lam, Tdist, None

    def _extra_loss(self, xb, logits):
        if self.old is None:
            return 0.0
        with torch.no_grad():
            zo = self.old(xb)
        prev = [c for c in self.seen if c < logits.shape[1]]
        idx = torch.tensor(prev, device=logits.device)
        p = F.log_softmax(logits[:, idx] / self.Tdist, dim=1)
        q = F.softmax(zo[:, idx] / self.Tdist, dim=1)
        return self.lam * F.kl_div(p, q, reduction="batchmean") * self.Tdist ** 2

    def _after_task(self):
        import copy
        self.old = copy.deepcopy(self.net).eval()
        for p in self.old.parameters():
            p.requires_grad_(False)


METHODS = {"E2E-Seq-FT": (E2EBase, {}),
           "E2E-LwF": (E2ELwF, {}),
           "E2E-Replay": (E2EBase, {"uses_memory": True}),
           "E2E-RBCC": (E2EBase, {"uses_memory": True, "reserve": True})}


# ---------------------------------------------------------------------- protocols

def run_one(method_key, seed=0, epochs=40, n_tasks=N_TASKS):
    (Xtr, ytr), (Xte, yte) = cifar100_tensors()
    n_classes = int(ytr.max()) + 1
    tasks = M.class_incremental_split(ytr.numpy(), n_tasks, seed=seed)
    cls, kw = METHODS[method_key]
    use_mem = kw.get("uses_memory", False)
    reserve = kw.get("reserve", False)
    model = cls(n_classes, seed=seed, epochs=epochs)
    model.name = method_key

    rng = np.random.default_rng(seed + 991)
    mem_idx, cal_idx = [], []
    rows = []
    for step, tc in enumerate(tasks):
        # Memory update happens BEFORE the training pass so that the calibration slots
        # can be withheld from it. The guarantee requires the reserved points never to
        # have been fitted: a point the optimiser has seen carries a systematically low
        # conformal score, so a quantile over such points is too small and the
        # predictor undercovers. An earlier version trained on the full task and
        # reserved afterwards from the same points; the measured bias in the frozen arm
        # is +0.018 in class-conditional coverage at k=20 (holdout_effect.csv) and is
        # larger here, because a network trained from scratch memorises more than a
        # linear head over frozen features. Slots are sampled UNIFORMLY and are
        # disjoint from the replayed slots; a selection rule that prefers prototypical
        # points cannot yield an exchangeable calibration set.
        step_cal = []
        if use_mem:
            for c in tc:
                idx = np.where(ytr.numpy() == c)[0]
                pick = rng.choice(idx, size=min(MEM_PER_CLASS, len(idx)), replace=False)
                if reserve:
                    step_cal += list(pick[:CAL_PER_CLASS])
                    mem_idx += list(pick[CAL_PER_CLASS:])
                else:
                    mem_idx += list(pick)
            cal_idx += step_cal
            assert not (set(mem_idx) & set(cal_idx)), "replay and calibration overlap"

        tr = np.where(np.isin(ytr.numpy(), tc))[0]
        if step_cal:
            tr = np.setdiff1d(tr, np.array(step_cal, dtype=int), assume_unique=False)
            assert not (set(tr.tolist()) & set(step_cal)), "training sees reserved slots"
        Xm = ym = None
        if use_mem and mem_idx:
            Xm, ym = Xtr[np.array(mem_idx)], ytr[np.array(mem_idx)]
        model.observe(Xtr[tr], ytr[tr], tc, Xmem=Xm, ymem=ym)

        seen = np.concatenate(tasks[:step + 1])
        te_seen = np.where(np.isin(yte.numpy(), seen))[0]
        cur = np.where(np.isin(yte.numpy(), tc))[0]

        # Protocol A: calibrate on held-out data from the CURRENT task only — what a
        # practitioner actually has at deployment time.
        half = len(cur) // 2
        a_cal, a_ev = cur[:half], cur[half:]
        model.fit_temperature(Xte[a_cal], yte[a_cal])
        P = model.probs(Xte)
        q_a = RL.conformal_quantile(RL.lac_score(P[a_cal], yte[a_cal].numpy()), ALPHA)

        prot = {"A_current": q_a}
        # Protocol B: the oracle — calibrate on held-out data from every seen task.
        b_cal = np.concatenate([np.where(np.isin(yte.numpy(), t))[0][:half] for t in tasks[:step + 1]])
        prot["B_pooled"] = RL.conformal_quantile(RL.lac_score(P[b_cal], yte[b_cal].numpy()), ALPHA)
        if use_mem and reserve and cal_idx:
            Pc = model.probs(Xtr[np.array(cal_idx)])
            prot["C_rbcc"] = RL.conformal_quantile(RL.lac_score(Pc, ytr[np.array(cal_idx)].numpy()), ALPHA)
        if use_mem and mem_idx:
            Pm = model.probs(Xtr[np.array(mem_idx)])
            prot["Cn_replayed"] = RL.conformal_quantile(RL.lac_score(Pm, ytr[np.array(mem_idx)].numpy()), ALPHA)

        for pk, q in prot.items():
            for t, tcl in enumerate(tasks[:step + 1]):
                ev = np.where(np.isin(yte.numpy(), tcl))[0][half:]
                S = RL.predict_sets(P[ev], q)
                rows.append(dict(method=method_key, seed=seed, step=step, task=t,
                                 protocol=pk, dataset="cifar100", backbone="resnet18_e2e",
                                 cov=float(RL.coverage(S, yte[ev].numpy())),
                                 size=float(S.sum(1).mean()),
                                 acc=float((P[ev].argmax(1) == yte[ev].numpy()).mean()),
                                 T=model.T))
    return rows


if __name__ == "__main__":
    mode = os.environ.get("MATRIX", "smoke")
    t0 = time.time()
    print(f"device={DEV} mode={mode}", flush=True)
    if DEV == "cpu":
        raise SystemExit("refusing to run end-to-end on CPU — this needs the GPU")
    if mode == "smoke":
        rows = run_one("E2E-Seq-FT", seed=0, epochs=5, n_tasks=3)
        D = pd.DataFrame(rows)
        print(D.groupby("protocol")[["cov", "acc"]].mean().round(4).to_string(), flush=True)
    else:
        all_rows = []
        # Optional subsetting, so single (method, seed) cells can run as parallel processes.
        only = [m for m in os.environ.get("METHODS_ONLY", "").split(",") if m]
        seeds = ([int(x) for x in os.environ["SEED_LIST"].split(",")] if os.environ.get("SEED_LIST")
                 else list(range(int(os.environ.get("SEEDS", 3)))))
        for mk in (only or METHODS):
            assert mk in METHODS, mk
            for s in seeds:
                all_rows += run_one(mk, seed=s, epochs=int(os.environ.get("EPOCHS", 40)))
                print(f"  {mk} seed={s}  {time.time()-t0:.0f}s", flush=True)
                pd.DataFrame(all_rows).to_csv(OUT, index=False)
    print(f"done in {time.time()-t0:.0f}s", flush=True)
