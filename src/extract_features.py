"""Phase 1 step 2: cache frozen-backbone features for every benchmark.

Design notes
------------
* The backbone is FROZEN and used only as a frozen feature map, which is the standard
  modern PTM-CIL protocol. Extraction happens once; every continual-learning method
  then runs on the cached matrices, so the full experiment matrix is cheap.
* Features are the final-layer [CLS] token (both ViT-B/16-in21k and DINOv2 expose it
  as `last_hidden_state[:, 0]`). We deliberately do NOT use the pooler, which is
  task-head specific and absent/randomly initialised in some checkpoints.
* ImageNet-R ships as a single split; PTM-CIL work splits it itself. We take a
  class-stratified split with a fixed seed and record it, so it is reproducible.
* Resumable: an existing output .npz is left alone.
"""
from __future__ import annotations
import os, sys, json, glob, time
os.environ.setdefault("KMP_AFFINITY", "disabled")
os.environ.setdefault("OMP_NUM_THREADS", "32")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
import torch
torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
from transformers import AutoModel, AutoImageProcessor

OUT = "features"; os.makedirs(OUT, exist_ok=True)
BACKBONES = {"vit_b16_in21k": "models/vit_b16_in21k", "dinov2_base": "models/dinov2_base"}
SPLIT_SEED = 0
BATCH = int(os.environ.get("EXTRACT_BATCH", "128"))


# ------------------------------------------------------------------------ data loading
def _hf_parquet(files, image_key=None, label_key=None):
    from datasets import load_dataset
    ds = load_dataset("parquet", data_files=files, split="train")
    cols = ds.column_names
    ik = image_key or next(c for c in ("img", "image", "jpg", "picture") if c in cols)
    lk = label_key or next(c for c in ("fine_label", "label", "labels", "coarse_label") if c in cols)
    return ds, ik, lk


def load_cifar100():
    tr, ik, lk = _hf_parquet(sorted(glob.glob("data/cifar100/cifar100/train-*.parquet")))
    te, _, _ = _hf_parquet(sorted(glob.glob("data/cifar100/cifar100/test-*.parquet")))
    return (tr, ik, lk), (te, ik, lk), None


def load_cub200():
    tr, ik, lk = _hf_parquet(sorted(glob.glob("data/cub200/data/train-*.parquet")))
    te, _, _ = _hf_parquet(sorted(glob.glob("data/cub200/data/test-*.parquet")))
    return (tr, ik, lk), (te, ik, lk), None


def load_imagenet_r():
    """Single-split dataset -> class-stratified 80/20 split with a recorded seed."""
    ds, ik, lk = _hf_parquet(sorted(glob.glob("data/imagenet_r/test/test-*.parquet")))
    y = np.asarray(ds[lk])
    rng = np.random.default_rng(SPLIT_SEED)
    tr_idx, te_idx = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        k = max(1, int(round(0.8 * len(idx))))
        tr_idx += idx[:k].tolist(); te_idx += idx[k:].tolist()
    meta = {"split": "self-made class-stratified 80/20", "seed": SPLIT_SEED,
            "n_train": len(tr_idx), "n_test": len(te_idx)}
    return (ds.select(sorted(tr_idx)), ik, lk), (ds.select(sorted(te_idx)), ik, lk), meta


def _medmnist_one(subset_path):
    """Read one MedMNIST .npz, tolerating both layouts found in the HF mirrors.

    Layout 1 (MedMNIST v2): train_images/train_labels/test_images/test_labels.
    Layout 2 (MedMNIST+ extras): a single images/labels pair with no split, in which
    case we make a class-stratified 80/20 split with a recorded seed.
    Grayscale arrays are promoted to 3 channels so one processor handles everything.
    """
    z = np.load(subset_path, mmap_mode="r")
    k = set(z.files)
    if {"train_images", "train_labels", "test_images", "test_labels"} <= k:
        return (np.asarray(z["train_images"]), np.asarray(z["train_labels"]).reshape(-1).astype(int),
                np.asarray(z["test_images"]), np.asarray(z["test_labels"]).reshape(-1).astype(int),
                {"split": "native"})
    if {"images", "labels"} <= k:
        X = np.asarray(z["images"]); y = np.asarray(z["labels"]).reshape(-1).astype(int)
        rng = np.random.default_rng(SPLIT_SEED)
        tr, te = [], []
        for c in np.unique(y):
            idx = np.where(y == c)[0]; rng.shuffle(idx)
            n = max(1, int(round(0.8 * len(idx))))
            tr += idx[:n].tolist(); te += idx[n:].tolist()
        tr, te = np.array(sorted(tr)), np.array(sorted(te))
        return (X[tr], y[tr], X[te], y[te],
                {"split": f"self-made stratified 80/20, seed={SPLIT_SEED}"})
    raise KeyError(f"{subset_path}: unrecognised keys {sorted(k)}")


def _to_rgb(X):
    if X.ndim == 3:                       # (n, H, W) grayscale
        return np.repeat(X[..., None], 3, axis=-1)
    if X.ndim == 4 and X.shape[-1] == 1:
        return np.repeat(X, 3, axis=-1)
    return X


def load_medmnist(subset_path):
    Xtr, ytr, Xte, yte, meta = _medmnist_one(subset_path)
    meta["source"] = os.path.basename(subset_path)
    return (_to_rgb(Xtr), ytr), (_to_rgb(Xte), yte), meta


def load_medmix(subsets):
    """Cross-modality medical class-incremental benchmark.

    Concatenates several MedMNIST subsets into ONE label space with disjoint class
    ranges, so an incremental task sequence walks across imaging modalities rather
    than staying inside one. This mirrors how a deployed clinical system actually
    grows — new departments and new modalities arrive over time — and it is a harder,
    more informative setting than splitting a single modality, which is why we prefer
    it for the medical arm.
    """
    Xtr_all, ytr_all, Xte_all, yte_all, offs, meta = [], [], [], [], 0, []
    for p in subsets:
        a, b, m = load_medmnist(p)
        Xtr, ytr = a; Xte, yte = b
        Xtr_all.append(Xtr); Xte_all.append(Xte)
        ytr_all.append(ytr + offs); yte_all.append(yte + offs)
        ncl = int(max(ytr.max(), yte.max())) + 1
        meta.append({"subset": m["source"], "classes": ncl,
                     "label_range": [offs, offs + ncl - 1], "split": m["split"],
                     "n_train": int(len(ytr)), "n_test": int(len(yte))})
        offs += ncl
    return (np.concatenate(Xtr_all), np.concatenate(ytr_all)), \
           (np.concatenate(Xte_all), np.concatenate(yte_all)), \
           {"modalities": meta, "total_classes": offs}


# ------------------------------------------------------------------------- extraction
@torch.no_grad()
def embed_hf(ds, ik, lk, model, proc, batch=64, tag=""):
    n = len(ds); feats = []; labels = np.asarray(ds[lk]).astype(int)
    t0 = time.time()
    for s in range(0, n, batch):
        imgs = [im.convert("RGB") for im in ds[s:s + batch][ik]]
        px = proc(images=imgs, return_tensors="pt")["pixel_values"]
        h = model(pixel_values=px).last_hidden_state[:, 0]
        feats.append(h.numpy().astype(np.float32))
        if s % (batch * 40) == 0:
            done = min(s + batch, n)
            print(f"    {tag} {done}/{n} ({done/n:.0%}) {time.time()-t0:.0f}s", flush=True)
    return np.concatenate(feats), labels


@torch.no_grad()
def embed_arr(images, labels, model, proc, batch=64, tag=""):
    from PIL import Image
    n = len(images); feats = []; t0 = time.time()
    for s in range(0, n, batch):
        chunk = images[s:s + batch]
        imgs = [Image.fromarray(a).convert("RGB") for a in chunk]
        px = proc(images=imgs, return_tensors="pt")["pixel_values"]
        h = model(pixel_values=px).last_hidden_state[:, 0]
        feats.append(h.numpy().astype(np.float32))
        if s % (batch * 40) == 0:
            done = min(s + batch, n)
            print(f"    {tag} {done}/{n} ({done/n:.0%}) {time.time()-t0:.0f}s", flush=True)
    return np.concatenate(feats), np.asarray(labels).astype(int)


def run(name, loader, bb_key):
    out = os.path.join(OUT, f"{name}__{bb_key}.npz")
    if os.path.exists(out):
        z = np.load(out)
        print(f"  skip {out} (exists: train {z['Xtr'].shape}, test {z['Xte'].shape})", flush=True)
        return out
    path = BACKBONES[bb_key]
    model = AutoModel.from_pretrained(path).eval()
    proc = AutoImageProcessor.from_pretrained(path)
    tr, te, meta = loader()
    t0 = time.time()
    if isinstance(tr, tuple) and len(tr) == 3:
        Xtr, ytr = embed_hf(*tr, model, proc, batch=BATCH, tag=f"{name}/{bb_key}/train")
        Xte, yte = embed_hf(*te, model, proc, batch=BATCH, tag=f"{name}/{bb_key}/test")
    else:
        Xtr, ytr = embed_arr(*tr, model, proc, batch=BATCH, tag=f"{name}/{bb_key}/train")
        Xte, yte = embed_arr(*te, model, proc, batch=BATCH, tag=f"{name}/{bb_key}/test")
    np.savez_compressed(out, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte,
                        meta=json.dumps({"dataset": name, "backbone": bb_key,
                                         "dim": int(Xtr.shape[1]),
                                         "n_classes": int(len(np.unique(ytr))),
                                         "extra": meta, "sec": round(time.time() - t0, 1)}))
    print(f"  wrote {out}  train{Xtr.shape} test{Xte.shape} "
          f"classes={len(np.unique(ytr))} in {time.time()-t0:.0f}s", flush=True)
    return out


if __name__ == "__main__":
    only = sys.argv[1:] or None
    jobs = [("cifar100", load_cifar100), ("imagenet_r", load_imagenet_r), ("cub200", load_cub200)]
    # medical crossover arm: one cross-modality benchmark (preferred) ...
    MIX = ["data/medmnist/bloodmnist_224.npz",       # 8 classes, blood-cell microscopy
           "data/medmnist/dermamnist_224.npz",       # 7 classes, dermatoscopy
           "data/medmnist/organcmnist_224.npz"]      # 11 classes, abdominal CT (coronal)
    if all(os.path.exists(p) for p in MIX):
        jobs.append(("medmix26", lambda: load_medmix(MIX)))
    # ... plus each single-modality subset, for the within-modality comparison
    for p in sorted(glob.glob("data/medmnist/*.npz")):
        stem = os.path.basename(p).replace("_224.npz", "").replace(".npz", "")
        jobs.append((f"med_{stem}", (lambda pp: (lambda: load_medmnist(pp)))(p)))
    for bb in BACKBONES:
        for name, loader in jobs:
            if only and not any(o in name or o == bb for o in only):
                continue
            print(f"[{bb}] {name}", flush=True)
            try:
                run(name, loader, bb)
            except Exception as e:
                import traceback
                print(f"  FAIL {name}/{bb}: {type(e).__name__}: {str(e)[:200]}", flush=True)
                traceback.print_exc()
