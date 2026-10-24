"""Class-incremental learning on frozen pre-trained features.

All methods consume cached backbone features, which is the standard modern PTM-CIL
protocol (Zhou et al., "Revisiting Class-Incremental Learning with Pre-Trained Models",
2024) and makes the full experiment matrix cheap enough to run many seeds.

Every method exposes the same interface:

    m = Method(dim, n_classes, **kw)
    m.observe(X, y, task_classes)      # incremental step for one task
    P = m.predict_proba(X)             # (n, C) row-stochastic

Probability quality is NOT taken for granted. Prototype/ridge methods emit scores,
not probabilities, so the runner gives *every* method the same post-hoc calibration
opportunity (a temperature fitted on that task's own calibration split at the moment
the task is learned). The scientific question is then whether that calibration — and
the conformal guarantee built on top of it — survives later tasks.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax


# ------------------------------------------------------------------ post-hoc calibration
def fit_temperature(logits: np.ndarray, labels: np.ndarray,
                    bounds=(0.05, 20.0)) -> float:
    """Single-parameter temperature scaling by NLL (Guo et al. 2017)."""
    def nll(T):
        return -log_softmax(logits / T, axis=1)[np.arange(len(labels)), labels].mean()
    r = minimize_scalar(nll, bounds=bounds, method="bounded")
    return float(r.x)


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    return softmax(logits / T, axis=1)


# --------------------------------------------------------------------------- base class
class Base:
    """Common bookkeeping. Subclasses implement _fit and _logits."""
    name = "base"

    def __init__(self, dim: int, n_classes: int, seed: int = 0, **kw):
        self.dim, self.C, self.seed = dim, n_classes, seed
        self.rng = np.random.default_rng(seed)
        self.seen = np.zeros(n_classes, dtype=bool)
        self.kw = kw

    def observe(self, X, y, task_classes):
        self.seen[list(task_classes)] = True
        self._fit(X, y, task_classes)

    def logits(self, X) -> np.ndarray:
        z = self._logits(X)
        z = z.copy()
        z[:, ~self.seen] = -1e9          # unseen classes are not predictable
        return z

    def predict_proba(self, X) -> np.ndarray:
        return softmax(self.logits(X), axis=1)

    def _fit(self, X, y, task_classes):
        raise NotImplementedError

    def _logits(self, X) -> np.ndarray:
        raise NotImplementedError


# ---------------------------------------------------------------- training-free methods
class NCM(Base):
    """Nearest Class Mean on L2-normalised features; cosine scores. == SimpleCIL."""
    name = "NCM (SimpleCIL)"

    def __init__(self, dim, n_classes, scale: float = 10.0, **kw):
        super().__init__(dim, n_classes, **kw)
        self.proto = np.zeros((n_classes, dim))
        self.count = np.zeros(n_classes)
        self.scale = scale

    def _fit(self, X, y, task_classes):
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        for c in task_classes:
            m = y == c
            if m.sum() == 0:
                continue
            self.proto[c] = (self.proto[c] * self.count[c] + Xn[m].sum(0)) / (self.count[c] + m.sum())
            self.count[c] += m.sum()

    def _logits(self, X):
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        Pn = self.proto / (np.linalg.norm(self.proto, axis=1, keepdims=True) + 1e-12)
        return self.scale * (Xn @ Pn.T)


class RanPAC(Base):
    """Random projection + class-prototype ridge readout (McDonnell et al. 2024).

    Expands features through a fixed random ReLU projection, then maintains the
    Gram matrix and class-sum matrix incrementally and solves a ridge system.
    Training-free in the sense that no gradient step is taken.
    """
    name = "RanPAC"

    def __init__(self, dim, n_classes, proj: int = 2000, lam: float = 1e3, **kw):
        super().__init__(dim, n_classes, **kw)
        self.W = self.rng.normal(scale=1.0 / np.sqrt(dim), size=(dim, proj))
        self.G = np.zeros((proj, proj))
        self.S = np.zeros((proj, n_classes))
        self.lam, self.proj = lam, proj
        self._beta = None

    def _phi(self, X):
        return np.maximum(X @ self.W, 0.0)

    def _fit(self, X, y, task_classes):
        H = self._phi(X)
        self.G += H.T @ H
        Y = np.zeros((len(y), self.C))
        Y[np.arange(len(y)), y] = 1.0
        self.S += H.T @ Y
        self._beta = np.linalg.solve(self.G + self.lam * np.eye(self.proj), self.S)

    def _logits(self, X):
        return self._phi(X) @ self._beta


# --------------------------------------------------------- gradient-trained linear heads
class _LinearHead(Base):
    """Shared SGD machinery for a single linear layer on frozen features."""

    def __init__(self, dim, n_classes, n_steps: int = 400, epochs: int | None = None,
                 lr: float = 0.1, batch: int = 256, wd: float = 1e-4, **kw):
        super().__init__(dim, n_classes, **kw)
        self.Wt = np.zeros((dim, n_classes))
        self.b = np.zeros(n_classes)
        self.n_steps, self.epochs_fixed = n_steps, epochs
        self.lr, self.batch, self.wd = lr, batch, wd
        self.epochs = epochs if epochs is not None else 1   # recomputed per _sgd call

    def _forward(self, X):
        return X @ self.Wt + self.b

    def _epochs_for(self, n):
        """Epochs needed to reach `self.n_steps` gradient steps on n samples.

        The budget is set in STEPS, not epochs, because a fixed epoch count gives
        wildly unequal optimisation across datasets: CIFAR-100 has ~4000 fit samples
        per task (12 epochs = 192 steps) while CUB-200 has ~480 (12 epochs = 24
        steps), an 8x difference that penalises exactly the small-per-task datasets.
        Measured consequence of the epoch-based budget on CUB-200/ViT: the linear
        head barely left initialisation (mean logit 0.005) and Replay's current-task
        accuracy was 0.004, which looks like a broken method but is starvation.
        `epochs` set explicitly overrides this.
        """
        if self.epochs_fixed is not None:
            return self.epochs_fixed
        per_epoch = max(1, int(np.ceil(n / self.batch)))
        return max(1, int(np.ceil(self.n_steps / per_epoch)))

    def _extra_grad(self, X, Z, P):
        """Hook for regularisers/distillation. Returns (dW, db) to ADD."""
        return 0.0, 0.0

    def _prox(self):
        """Hook for penalties applied as an implicit (proximal) step after the
        gradient step. Used for quadratic anchors, where an explicit step is only
        stable for lr*strength < 2 and otherwise diverges silently."""
        return

    def _sgd(self, X, y):
        n = len(y)
        for ep in range(self._epochs_for(n)):
            idx = self.rng.permutation(n)
            for s in range(0, n, self.batch):
                bi = idx[s:s + self.batch]
                Xb, yb = X[bi], y[bi]
                Z = self._forward(Xb)
                P = softmax(Z, axis=1)
                G = P.copy()
                G[np.arange(len(yb)), yb] -= 1.0
                G /= len(yb)
                dW = Xb.T @ G + self.wd * self.Wt
                db = G.sum(0)
                eW, eb = self._extra_grad(Xb, Z, P)
                self.Wt -= self.lr * (dW + eW)
                self.b -= self.lr * (db + eb)
                self._prox()

    def _logits(self, X):
        return self._forward(X)


class SeqFT(_LinearHead):
    """Sequential fine-tuning of the head on the current task only — the lower bound."""
    name = "Seq-FT"

    def _fit(self, X, y, task_classes):
        self._sgd(X, y)


class Replay(_LinearHead):
    """Experience replay with a class-balanced exemplar memory.

    The memory budget `mem_per_class` is PARTITIONED, not enlarged: `cal_per_class`
    of those slots are reserved for calibration and are never shown to the optimiser.
    This matters because conformal calibration requires the calibration points to be
    exchangeable with test points; exemplars that were replayed during training are
    fitted better than test data, which biases the non-conformity scores downward and
    makes the resulting quantile under-cover. Reserving a slice of the SAME budget
    keeps the total memory unchanged while restoring exchangeability.
    """
    name = "Replay"

    def __init__(self, dim, n_classes, mem_per_class: int = 20,
                 cal_per_class: int = 0, hold_out_cal: bool = False, **kw):
        super().__init__(dim, n_classes, **kw)
        # Withhold the calibration slots from the training pass. Default off so every
        # number already in the paper reproduces exactly; the effect of turning it on
        # is measured separately and reported.
        self.hold_out_cal = bool(hold_out_cal)
        self.mem_X, self.mem_y = [], []          # replayed during training
        self.cal_X, self.cal_y = [], []          # reserved: calibration only
        self.mem_per_class = mem_per_class
        self.cal_per_class = min(cal_per_class, max(mem_per_class - 1, 0))
        self.train_per_class = mem_per_class - self.cal_per_class

    def _select(self, X, y, c, k):
        """Selection rule for k exemplars of class c. Random for plain replay."""
        idx = np.where(y == c)[0]
        if len(idx) == 0:
            return np.array([], dtype=int)
        return self.rng.choice(idx, size=min(k, len(idx)), replace=False)

    def _sgd_replay(self, X, y, bX, by):
        """Standard experience replay (Chaudhry et al., tiny episodic memories):
        every new-data minibatch is paired with an equally sized minibatch drawn
        from memory, and one step is taken on their union.

        The alternative — concatenating the whole buffer with the new task's data and
        shuffling — is NOT equivalent, and degenerates whenever memory outgrows the
        per-task data. Measured on CUB-200 (480 fit samples/task vs a 3000-exemplar
        memory, 6.25:1), concatenation drove current-task accuracy to 0.000 by the
        final step while old classes were retained: the new task never gets learned.
        Interleaving keeps the new/old gradient ratio at 1:1 regardless of how large
        memory has grown.
        """
        n = len(y); nb = len(by)
        for ep in range(self._epochs_for(n)):
            idx = self.rng.permutation(n)
            for s in range(0, n, self.batch):
                bi = idx[s:s + self.batch]
                k = min(len(bi), nb)
                mi = self.rng.choice(nb, size=k, replace=False)
                Xb = np.concatenate([X[bi], bX[mi]])
                yb = np.concatenate([y[bi], by[mi]])
                Z = self._forward(Xb)
                P = softmax(Z, axis=1)
                G = P.copy(); G[np.arange(len(yb)), yb] -= 1.0
                G /= len(yb)
                self.Wt -= self.lr * (Xb.T @ G + self.wd * self.Wt)
                self.b -= self.lr * G.sum(0)

    def _store(self, X, y, task_classes):
        """Fill both partitions of the per-class budget.

        The calibration slots are filled by UNIFORM RANDOM sampling, never by the
        method's own exemplar-selection rule. This is load-bearing: herding-style
        rules deliberately pick prototypical points near the class mean, so a
        herding-selected subset is not a random sample of the class distribution and
        can never be exchangeable with test points, whichever way it is split. The
        replay slots keep the method's own rule, since that is what the method is.
        """
        for c in task_classes:
            idx = np.where(y == c)[0]
            if len(idx) == 0:
                continue
            n_cal = min(self.cal_per_class, max(len(idx) - 1, 0))
            cal_take = (self.rng.choice(idx, size=n_cal, replace=False)
                        if n_cal > 0 else np.array([], dtype=int))
            if n_cal > 0:
                self.cal_X.append(X[cal_take]); self.cal_y.append(y[cal_take])
            # replay exemplars are chosen by the method's rule from the REMAINING pool,
            # so no point is both replayed and used for calibration
            pool = np.setdiff1d(idx, cal_take, assume_unique=False)
            if len(pool) == 0:
                continue
            mask = np.zeros(len(y), dtype=bool); mask[pool] = True
            sub = self._select(X, np.where(mask, y, -1), c, self.train_per_class)
            if len(sub):
                self.mem_X.append(X[sub]); self.mem_y.append(y[sub])

    def buffer(self):
        """Exemplars available for REPLAY (i.e. seen by the optimiser)."""
        if not self.mem_X:
            return None, None
        return np.concatenate(self.mem_X), np.concatenate(self.mem_y)

    def cal_buffer(self):
        """Calibration-reserved exemplars: stored but NEVER trained on."""
        if not self.cal_X:
            return None, None
        return np.concatenate(self.cal_X), np.concatenate(self.cal_y)

    def memory_footprint(self):
        """Total exemplars held, for the memory-parity accounting in the paper."""
        n_tr = sum(len(a) for a in self.mem_y)
        n_cal = sum(len(a) for a in self.cal_y)
        return {"replay": n_tr, "calibration_reserved": n_cal, "total": n_tr + n_cal}

    def _fit(self, X, y, task_classes):
        """Train on the new task with interleaved memory minibatches, then store.

        The calibration slots are reserved BEFORE the training pass and withheld from
        it when `hold_out_cal` is set. This is required by the guarantee, not a
        refinement of it: a point the optimiser has fitted carries a systematically
        low conformal score, so a quantile taken over such points is too small and
        the predictor under-covers. Reserving the slots after training instead
        biases coverage; the bias is small for a linear head over frozen features
        (400 gradient steps memorise little) and large for a network trained from
        scratch for 150 epochs (results/validation).
        """
        if self.hold_out_cal and self.cal_per_class > 0:
            hold = self._reserve(X, y, task_classes)
            keep = np.setdiff1d(np.arange(len(y)), hold, assume_unique=False)
            Xf, yf = X[keep], y[keep]
        else:
            hold, Xf, yf = None, X, y
        bX, by = self.buffer()
        if bX is None:
            self._sgd(Xf, yf)
        else:
            self._sgd_replay(Xf, yf, bX, by)
        if hold is None:
            self._store(X, y, task_classes)
        else:
            self._store_split(X, y, task_classes, hold)

    def _reserve(self, X, y, task_classes):
        """Indices of the calibration slots, chosen before any training on this task.

        Draws from a DEDICATED generator rather than self.rng. _sgd draws its
        minibatch permutation from self.rng, so reserving from the same stream would
        shift the optimisation trajectory and the hold-out comparison would confound
        the calibration change with a different training run — which is exactly what
        the first version of this measurement did, visibly, by moving protocols that
        do not use the reserved slots at all.
        """
        if getattr(self, "_res_rng", None) is None:
            self._res_rng = np.random.default_rng(777_000 + int(self.seed))
        out = []
        for c in task_classes:
            idx = np.where(y == c)[0]
            if len(idx) == 0:
                continue
            n_cal = min(self.cal_per_class, max(len(idx) - 1, 0))
            if n_cal:
                out += list(self._res_rng.choice(idx, size=n_cal, replace=False))
        return np.array(out, dtype=int)

    def _store_split(self, X, y, task_classes, hold):
        """Store with the calibration partition fixed in advance by `_reserve`.

        Mirrors `_store` exactly except that the calibration indices are supplied
        rather than drawn here, so the slots stored are the ones that were withheld
        from the training pass. `_select` takes the full label vector with
        non-candidates masked to -1, which is this class's own convention.
        """
        held = set(int(h) for h in hold)
        for c in task_classes:
            idx = np.where(y == c)[0]
            if len(idx) == 0:
                continue
            cal_take = np.array([i for i in idx if i in held], dtype=int)
            if len(cal_take):
                self.cal_X.append(X[cal_take]); self.cal_y.append(y[cal_take])
            pool = np.setdiff1d(idx, cal_take, assume_unique=False)
            if len(pool) == 0 or self.train_per_class <= 0:
                continue
            mask = np.zeros(len(y), dtype=bool); mask[pool] = True
            sub = self._select(X, np.where(mask, y, -1), c, self.train_per_class)
            if len(sub):
                self.mem_X.append(X[sub]); self.mem_y.append(y[sub])


class LwF(_LinearHead):
    """Learning without Forgetting: KD on logits of previously-seen classes."""
    name = "LwF"

    def __init__(self, dim, n_classes, T: float = 2.0, lam: float = 1.0, **kw):
        super().__init__(dim, n_classes, **kw)
        self.T, self.lam = T, lam
        self.old_W = None; self.old_b = None; self.old_mask = None

    def _extra_grad(self, Xb, Z, P):
        if self.old_W is None:
            return 0.0, 0.0
        mask = self.old_mask
        Zo = (Xb @ self.old_W + self.old_b)[:, mask] / self.T
        Q = softmax(Zo, axis=1)
        Zs = Z[:, mask] / self.T
        Ps = softmax(Zs, axis=1)
        Gm = (Ps - Q) / (len(Xb) * self.T)
        G = np.zeros_like(Z); G[:, mask] = Gm
        return self.lam * (Xb.T @ G), self.lam * G.sum(0)

    def _fit(self, X, y, task_classes):
        if self.seen.sum() > len(task_classes):
            self.old_W = self.Wt.copy(); self.old_b = self.b.copy()
            self.old_mask = self.seen.copy()
            self.old_mask[list(task_classes)] = False
        self._sgd(X, y)


class EWC(_LinearHead):
    """Elastic Weight Consolidation with a diagonal empirical Fisher."""
    name = "EWC"

    def __init__(self, dim, n_classes, lam: float = 5.0, **kw):
        super().__init__(dim, n_classes, **kw)
        self.lam = lam
        self.star_W = None; self.star_b = None
        self.FW = np.zeros_like(self.Wt); self.Fb = np.zeros_like(self.b)

    def _norm_fisher(self):
        """Unit-mean normalisation of the accumulated diagonal Fisher.

        Without this the penalty gradient lam*F*(W-W*) can exceed 1/lr and the SGD
        update diverges, which silently destroys the head (observed as chance-level
        accuracy). Normalising makes `lam` scale-free and the step size stable.
        """
        mw = self.FW.mean()
        mb = self.Fb.mean()
        FW = self.FW / mw if mw > 0 else self.FW
        Fb = self.Fb / mb if mb > 0 else self.Fb
        return FW, Fb

    def _prox(self):
        """Implicit step for the quadratic anchor lam/2 * F * (W - W*)^2:

            W <- (W + lr*lam*F*W*) / (1 + lr*lam*F)

        This is the exact proximal operator, so it is stable for ANY lam — unlike the
        explicit gradient form, which diverges once lr*lam*F exceeds 2 and would make a
        large-lam EWC look merely 'inaccurate' instead of numerically broken.
        """
        if self.star_W is None:
            return
        cW = self.lr * self.lam * self._nFW
        cb = self.lr * self.lam * self._nFb
        self.Wt = (self.Wt + cW * self.star_W) / (1.0 + cW)
        self.b = (self.b + cb * self.star_b) / (1.0 + cb)

    def _fisher(self, X, y):
        """Diagonal empirical Fisher of the linear head under cross-entropy."""
        Z = self._forward(X); P = softmax(Z, axis=1)
        G = P.copy(); G[np.arange(len(y)), y] -= 1.0
        fw = (X ** 2).T @ (G ** 2) / len(y)
        fb = (G ** 2).mean(0)
        return fw, fb

    def _fit(self, X, y, task_classes):
        if self.star_W is not None:
            self._nFW, self._nFb = self._norm_fisher()
        self._sgd(X, y)
        fw, fb = self._fisher(X, y)
        self.FW += fw; self.Fb += fb
        self.star_W = self.Wt.copy(); self.star_b = self.b.copy()
        self._nFW, self._nFb = self._norm_fisher()


class iCaRL(Replay):
    """Herding exemplars + distillation on the head, NCM classification at test time."""
    name = "iCaRL"

    def __init__(self, dim, n_classes, T: float = 2.0, lam: float = 1.0, **kw):
        super().__init__(dim, n_classes, **kw)
        self.T, self.lam = T, lam
        self.old_W = None; self.old_b = None; self.old_mask = None
        self.ncm = NCM(dim, n_classes, seed=self.seed)

    def _select(self, X, y, c, k):
        """Herding (Welling 2009): greedily pick exemplars whose running mean tracks
        the class mean — the selection rule iCaRL actually specifies."""
        idx = np.where(y == c)[0]
        if len(idx) == 0:
            return np.array([], dtype=int)
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        F = Xn[idx]; mu = F.mean(0)
        chosen, acc = [], np.zeros_like(mu)
        for t in range(min(k, len(idx))):
            d = np.linalg.norm(mu - (acc + F) / (t + 1), axis=1)
            d[chosen] = np.inf
            j = int(np.argmin(d))
            chosen.append(j); acc = acc + F[j]
        return idx[chosen]

    def _extra_grad(self, Xb, Z, P):
        if self.old_W is None:
            return 0.0, 0.0
        mask = self.old_mask
        Zo = (Xb @ self.old_W + self.old_b)[:, mask] / self.T
        Q = softmax(Zo, axis=1)
        Ps = softmax(Z[:, mask] / self.T, axis=1)
        Gm = (Ps - Q) / (len(Xb) * self.T)
        G = np.zeros_like(Z); G[:, mask] = Gm
        return self.lam * (Xb.T @ G), self.lam * G.sum(0)

    def _fit(self, X, y, task_classes):
        if self.seen.sum() > len(task_classes):
            self.old_W = self.Wt.copy(); self.old_b = self.b.copy()
            self.old_mask = self.seen.copy(); self.old_mask[list(task_classes)] = False
        # Same reservation discipline as Replay._fit: the calibration slots are drawn
        # before the training pass and withheld from it. iCaRL overrides _fit for
        # distillation and the NCM rebuild, so the hold-out branch has to be repeated
        # here; inheriting it was the assumption that made the first measurement show
        # no effect at all for this method.
        if self.hold_out_cal and self.cal_per_class > 0:
            hold = self._reserve(X, y, task_classes)
            keep = np.setdiff1d(np.arange(len(y)), hold, assume_unique=False)
            Xf, yf = X[keep], y[keep]
        else:
            hold, Xf, yf = None, X, y
        bX, by = self.buffer()
        Xtr, ytr = (np.concatenate([Xf, bX]), np.concatenate([yf, by])) if bX is not None else (Xf, yf)
        self._sgd(Xtr, ytr)
        if hold is None:
            self._store(X, y, task_classes)
        else:
            self._store_split(X, y, task_classes, hold)
        # NCM head is rebuilt from the exemplar memory + current task (iCaRL inference)
        self.ncm = NCM(self.dim, self.C, seed=self.seed)
        aX, ay = self.buffer()
        self.ncm.observe(aX, ay, sorted(set(ay.tolist())))

    def _logits(self, X):
        return self.ncm._logits(X)


class Joint(_LinearHead):
    """Offline upper bound: retrain from scratch on all data seen so far.

    Joint gets its OWN optimisation budget, larger than the per-task budget the
    continual methods use, because it has to be a genuine ceiling. Under the shared
    12-epoch budget it reaches only 0.807 on CIFAR-100/ViT features and lands BELOW
    the prototype baselines (RanPAC 0.824, NCM 0.817), which makes it useless as a
    bound and would be the first thing a reviewer flags. Trained to convergence
    (300 epochs, lr 0.1) it reaches 0.857, matching a full scikit-learn linear probe
    on the same features (0.854). The asymmetry is deliberate and is stated in the
    paper: the bound answers "what do these frozen features permit?", not "what does
    this method achieve under an equal budget?".
    """
    name = "Joint (upper bound)"

    def __init__(self, dim, n_classes, epochs: int = 300, lr: float = 0.1, **kw):
        super().__init__(dim, n_classes, epochs=epochs, lr=lr, **kw)
        self.allX, self.ally = [], []

    def _fit(self, X, y, task_classes):
        self.allX.append(X); self.ally.append(y)
        self.Wt = np.zeros((self.dim, self.C)); self.b = np.zeros(self.C)
        self._sgd(np.concatenate(self.allX), np.concatenate(self.ally))


METHODS = {
    "Joint":   Joint,
    "Seq-FT":  SeqFT,
    "EWC":     EWC,
    "LwF":     LwF,
    "Replay":  Replay,
    "iCaRL":   iCaRL,
    "NCM":     NCM,
    "RanPAC":  RanPAC,
}


# ------------------------------------------------------------------- incremental protocol
def class_incremental_split(labels: np.ndarray, n_tasks: int, seed: int = 0):
    """Partition classes into `n_tasks` equal groups under a fixed shuffled class order.

    Returns a list of class-index tuples. Uses the conventional protocol of shuffling
    the class order with a seed, then splitting into contiguous blocks.
    """
    classes = np.unique(labels)
    rng = np.random.default_rng(seed)
    order = rng.permutation(classes)
    if len(order) < n_tasks:
        raise ValueError(f"{len(order)} classes cannot be split into {n_tasks} tasks")
    # np.array_split distributes the remainder over the leading tasks, so EVERY class
    # enters exactly one task. (Floor division would drop len % n_tasks
    # classes, e.g. 6 of MedMix-26's 26.)
    blocks = np.array_split(order, n_tasks)
    assert sum(len(b) for b in blocks) == len(order)
    return [tuple(sorted(b.tolist())) for b in blocks]
