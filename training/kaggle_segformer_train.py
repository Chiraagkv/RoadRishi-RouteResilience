"""
RoadRishi — SegFormer Fine-Tuning Script (Kaggle / local, crash-proof, exact mid-epoch resume)
=============================================================================================
Dataset: https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset

Kaggle:
  1. Settings → Accelerator: GPU T4 x2.  Attach the DeepGlobe dataset.
  2. Prefer "Save Version → Save & Run All (Commit)" over running cells interactively.
     A committed run is headless: it keeps going when the browser tab sleeps or
     disconnects (the usual reason interactive sessions die), for up to 12 h.
  3. If a committed run dies anyway, open its Output, "Add as input" to a new version of
     this notebook and Run All again — the script finds latest.pth under /kaggle/input,
     copies it to /kaggle/working and resumes from the exact batch it stopped at.

Local (e.g. a 6 GB laptop GPU):
  ROADRISHI_DATA_ROOT=data/deepglobe/train ROADRISHI_BATCH=4 ROADRISHI_ACCUM=4 \
      python training/kaggle_segformer_train.py
  Ctrl-C (or SIGTERM) saves a checkpoint at the next optimizer step; re-run to resume.

Crash defences:
  ✅ Mid-epoch checkpoint  → latest.pth every `ckpt_every_batches` batches (full state,
                             incl. partial epoch metrics) — a crash loses minutes, not epochs
  ✅ Exact resume          → seeded per-epoch permutation; resume skips exactly the batches done
  ✅ Atomic saves          → write to .tmp then rename; a crash mid-save never corrupts a ckpt
  ✅ Cross-session resume  → picks up latest.pth from /kaggle/input (a previous run's output)
  ✅ Graceful stop         → SIGINT/SIGTERM or the wall-clock budget → checkpoint, then exit
  ✅ CUDA OOM mid-batch    → skip batch, empty_cache, continue
  ✅ NaN loss              → skip step; auto-disable AMP after 10 NaNs
  ✅ Quiet progress bars   → tqdm refreshes every 30 s when not on a TTY (huge notebook
                             output is a known cause of browser/session lock-ups)

Every setting in CFG can be overridden with an environment variable ROADRISHI_<NAME>
(see _env() below), which is also how tests/test_training_smoke.py drives a tiny CPU run.

Result files written to CFG["output_dir"] (download ALL of them from the Kaggle Output tab):
  roadrishi_finetuned.pth   best checkpoint (by val mIoU) — the one the backend loads
  latest.pth                most recent resumable state (may be mid-epoch)
  metrics.json              held-out evaluation of the best checkpoint (the numbers to cite)
  history.json              per-epoch train/val loss, mIoU, road IoU, P/R/F1, lr, time
  run_meta.json             config, seed, git/script hash, versions, GPUs, augmentations
  split.json                exact train/val file lists (reused on resume)
  per_image_val.json        per-image road IoU, TP/FP/FN
  curves.png                loss / IoU / P-R-F1 curves
  predictions_grid.png      worst / median / best validation predictions
"""

import os, sys, json, math, time, random, signal, shutil, subprocess, warnings, importlib

IS_KAGGLE = os.path.exists("/kaggle/input")

# ── Albumentations upgrade (Kaggle image ships an old one) ─────────────────────
if IS_KAGGLE:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                    "albumentations>=1.4.15"], check=True)

# ── Dependency check ───────────────────────────────────────────────────────────
for pkg, (maj, mn) in {"transformers": (4, 37), "albumentations": (1, 4)}.items():
    try:
        mod = importlib.import_module(pkg)
        ver = getattr(mod, "__version__", "0.0")
        parts = [int(x) for x in ver.split(".")[:2] if x.isdigit()]
        ok = tuple(parts) >= (maj, mn)
        print(f"{'✅' if ok else '⚠️ TOO OLD'}  {pkg} {ver}")
    except ImportError:
        print(f"❌  {pkg} not found"); sys.exit(1)

# ── Imports ────────────────────────────────────────────────────────────────────
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
try:
    from torch.amp import autocast, GradScaler          # torch >= 2.0
    _AMP_DEVICE = "cuda"
except ImportError:
    from torch.cuda.amp import autocast, GradScaler     # torch < 2.0 fallback
    _AMP_DEVICE = None
from torch.utils.data import Dataset, DataLoader, Sampler
from transformers import SegformerForSemanticSegmentation, SegformerConfig
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

print(f"PyTorch {torch.__version__} | CUDA {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {p.name}  {p.total_memory//1024**2} MB")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Directory of this script (a Kaggle notebook cell has no __file__)
_HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

# ── Auto-detect dataset / pretrained weights on Kaggle ─────────────────────────
def _find_deepglobe_root() -> str:
    import glob as _g
    masks = _g.glob("/kaggle/input/**/*_mask.png", recursive=True)
    if masks:
        dirs = list({str(Path(p).parent) for p in masks})
        chosen = next((d for d in dirs if Path(d).name == "train"), dirs[0])
        print(f"[AUTO-DETECT] {chosen}  ({len([m for m in masks if str(Path(m).parent)==chosen])} masks)")
        return chosen
    fallback = "/kaggle/input/deepglobe-road-extraction-dataset/train"
    print(f"[WARN] No masks found. Falling back to {fallback}")
    return fallback

def _find_local_segformer(default: str) -> str:
    """Use a SegFormer attached as a Kaggle Model (no HF download / rate limits) if present."""
    import glob as _g
    for cfg_path in _g.glob("/kaggle/input/models/**/config.json", recursive=True):
        try:
            if "segformer" in json.load(open(cfg_path)).get("model_type", ""):
                print(f"[AUTO-DETECT] pretrained weights: {Path(cfg_path).parent}")
                return str(Path(cfg_path).parent)
        except Exception:
            pass
    return default

def _env(name, default, cast=str):
    """ROADRISHI_<NAME> environment override."""
    v = os.environ.get(f"ROADRISHI_{name}")
    if v is None or v == "":
        return default
    if cast is bool:
        return v.strip().lower() in ("1", "true", "yes", "on")
    return cast(v)

# ── Config ─────────────────────────────────────────────────────────────────────
_HF_B3 = "nvidia/segformer-b3-finetuned-ade-512-512"
CFG = {
    "data_root":    _env("DATA_ROOT", None) or (_find_deepglobe_root() if IS_KAGGLE
                                                else str(_HERE.parent / "data" / "deepglobe" / "train")),
    "output_dir":   _env("OUTPUT_DIR", "/kaggle/working/checkpoints" if IS_KAGGLE
                                       else str(_HERE / "checkpoints")),
    "model_name":   _env("MODEL", None) or (_find_local_segformer(_HF_B3) if IS_KAGGLE else _HF_B3),
    "tiny_model":   _env("TINY_MODEL", False, bool),   # random-init mini SegFormer (tests only)
    "num_labels":   2,
    "img_size":     _env("IMG", 384, int),
    "batch_size":   _env("BATCH", 8, int),     # 8 fits a T4 at 384 px with AMP; use 4 on 6 GB
    "grad_accum":   _env("ACCUM", 2, int),
    "num_epochs":   _env("EPOCHS", 8, int),
    "lr":           _env("LR", 5e-5, float),
    "weight_decay": 1e-4,
    "val_split":    0.1,
    "max_train":    _env("MAX_TRAIN", 0, int), # >0: subsample (smoke tests)
    "max_val":      _env("MAX_VAL", 0, int),
    "num_workers":  _env("WORKERS", 2, int),
    "seed":         _env("SEED", 42, int),
    "bce_weight":   1.0,
    "dice_weight":  1.0,
    "use_amp":      _env("AMP", True, bool) and torch.cuda.is_available(),
    "ckpt_every_batches": _env("CKPT_EVERY", 150, int),   # ~2.5 min on T4 x2
    "max_hours":    _env("MAX_HOURS", 11.5 if IS_KAGGLE else 0.0, float),  # 0 = no limit
    "early_stopping_patience": _env("PATIENCE", 3, int),
    "resume_from":  _env("RESUME_FROM", ""),   # explicit checkpoint path (optional)
}
# Mid-epoch checkpoints are only taken right after an optimizer step (no partial grads).
CFG["ckpt_every_batches"] = max(CFG["grad_accum"],
                                math.ceil(CFG["ckpt_every_batches"] / CFG["grad_accum"]) * CFG["grad_accum"])
# Test hook: hard-kill the process after N training batches (simulates a Kaggle crash).
_CRASH_AFTER = _env("CRASH_AFTER_BATCHES", 0, int)
_TQDM_INTERVAL = _env("TQDM_INTERVAL", 0.5 if sys.stderr.isatty() else 30.0, float)

os.makedirs(CFG["output_dir"], exist_ok=True)
random.seed(CFG["seed"]); np.random.seed(CFG["seed"])
torch.manual_seed(CFG["seed"])
if torch.cuda.is_available(): torch.cuda.manual_seed_all(CFG["seed"])

print(f"\nimg={CFG['img_size']} bs={CFG['batch_size']} accum={CFG['grad_accum']} "
      f"eff_bs={CFG['batch_size']*CFG['grad_accum']} ep={CFG['num_epochs']} "
      f"AMP={CFG['use_amp']} workers={CFG['num_workers']} "
      f"ckpt_every={CFG['ckpt_every_batches']} batches\n")

# ── VRAM check ─────────────────────────────────────────────────────────────────
def check_vram():
    if not torch.cuda.is_available(): return
    for i in range(torch.cuda.device_count()):
        tot  = torch.cuda.get_device_properties(i).total_memory / 1024**3
        used = torch.cuda.memory_reserved(i) / 1024**3
        free = tot - used
        flag = "⚠️" if free < 3 else "✅"
        print(f"  {flag} GPU {i}: {free:.1f}/{tot:.1f} GB free")

# ── Augmentations ──────────────────────────────────────────────────────────────
_SZ = CFG["img_size"]
train_transform = A.Compose([
    A.RandomResizedCrop(size=(_SZ, _SZ), scale=(0.5, 1.0)),
    A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.3), A.RandomRotate90(p=0.3),
    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1, p=0.5),
    A.GaussNoise(std_range=(0.04, 0.22), p=0.3),
    A.GaussianBlur(blur_limit=3, p=0.2), A.ElasticTransform(p=0.2),
    # Warps road geometry (bends/kinks) without breaking connectivity — mask gets the same warp
    A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.25),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)), ToTensorV2(),
])
val_transform = A.Compose([
    A.Resize(height=_SZ, width=_SZ),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)), ToTensorV2(),
])

# ── Dataset ────────────────────────────────────────────────────────────────────
class DeepGlobeDataset(Dataset):
    def __init__(self, imgs, masks, tfm=None, seed=None):
        self.imgs, self.masks, self.tfm = imgs, masks, tfm
        self.seed, self.epoch = seed, 1
    def __len__(self): return len(self.imgs)
    def __getitem__(self, i):
        img  = np.array(Image.open(self.imgs[i]).convert("RGB"))
        mask = np.array(Image.open(self.masks[i]).convert("L"))
        mask = (mask > 127).astype(np.uint8)
        if self.tfm:
            if self.seed is not None and hasattr(self.tfm, "set_random_seed"):
                # Augmentation depends only on (seed, epoch, sample) — independent of worker
                # count and of where a resumed run starts, so resume is bit-for-bit exact.
                self.tfm.set_random_seed((self.seed * 1_000_003 + self.epoch * 100_003 + i) % 2**32)
            aug = self.tfm(image=img, mask=mask)
            img, mask = aug["image"], aug["mask"]
        return {"pixel_values": img.float(), "labels": mask.long()}

def make_split(pairs, val_split, seed, split_path):
    """
    Deterministic train/val split, persisted to split.json and re-used on resume so a
    restarted run can never leak validation images into training.
    NOTE: DeepGlobe only ships labels for its 'train' set (6,226 tiles), so the held-out
    set is a seeded random 10% of those; the public valid/test folders have no masks.
    """
    by_name = {s.name: (s, m) for s, m in pairs}
    if os.path.exists(split_path):
        sp = json.load(open(split_path))
        if all(n in by_name for n in sp["train"] + sp["val"]):
            print(f"[SPLIT] re-using {split_path}  (train {len(sp['train'])} / val {len(sp['val'])})")
            return [by_name[n] for n in sp["train"]], [by_name[n] for n in sp["val"]]
        print("[WARN] split.json does not match the dataset on disk — regenerating")
    names = sorted(by_name)
    random.Random(seed).shuffle(names)
    n = max(1, int(len(names) * val_split))
    val_names, train_names = names[:n], names[n:]
    with open(split_path, "w") as f:
        json.dump({"seed": seed, "val_split": val_split,
                   "train": train_names, "val": val_names}, f, indent=1)
    return [by_name[x] for x in train_names], [by_name[x] for x in val_names]

def build_dataloaders(cfg):
    root = Path(cfg["data_root"])
    sats  = sorted(root.glob("*_sat.jpg"))
    masks = [root / s.name.replace("_sat.jpg","_mask.png") for s in sats]
    pairs = [(s,m) for s,m in zip(sats,masks) if m.exists()]
    if not pairs:
        raise FileNotFoundError(f"No image-mask pairs in {root}")
    print(f"Pairs: {len(pairs)}")
    tr, va = make_split(pairs, cfg["val_split"], cfg["seed"],
                        os.path.join(cfg["output_dir"], "split.json"))
    if cfg.get("max_train"): tr = tr[:cfg["max_train"]]
    if cfg.get("max_val"):   va = va[:cfg["max_val"]]
    nw = cfg["num_workers"]
    _pin = torch.cuda.is_available()   # pin_memory requires CUDA; crashes silently on CPU
    train_ds = DeepGlobeDataset([p[0] for p in tr],[p[1] for p in tr],train_transform,
                                seed=cfg["seed"])
    sampler  = ResumableEpochSampler(len(train_ds), cfg["seed"], dataset=train_ds)
    # Own generator: creating the iterator must not consume the global RNG (used by dropout),
    # otherwise a resumed run's dropout masks would drift from the uninterrupted run.
    tl = DataLoader(train_ds, batch_size=cfg["batch_size"], sampler=sampler, num_workers=nw,
                    pin_memory=_pin, persistent_workers=False,
                    generator=torch.Generator().manual_seed(cfg["seed"]))
    vl = DataLoader(DeepGlobeDataset([p[0] for p in va],[p[1] for p in va],val_transform),
                    batch_size=cfg["batch_size"]*2, shuffle=False, num_workers=nw,
                    pin_memory=_pin, persistent_workers=False)
    print(f"Train {len(tr)} | Val {len(va)}")
    return tl, vl, sampler

class ResumableEpochSampler(Sampler):
    """
    Shuffles with a permutation that depends only on (seed, epoch), so after a crash the
    resumed run sees exactly the same order and can skip the first `start` samples instead
    of re-training on (or never seeing) part of the epoch.
    """
    def __init__(self, n, seed, dataset=None):
        self.n, self.seed, self.epoch, self.start, self.dataset = n, seed, 1, 0, dataset
    def set_epoch(self, epoch, start=0):
        self.epoch, self.start = epoch, start
        if self.dataset is not None:
            self.dataset.epoch = epoch     # copied into workers when the iterator starts
    def order(self):
        g = torch.Generator(); g.manual_seed(self.seed * 10_000 + self.epoch)
        return torch.randperm(self.n, generator=g).tolist()
    def __iter__(self):
        return iter(self.order()[self.start:])
    def __len__(self):
        return max(0, self.n - self.start)

# ── Loss ───────────────────────────────────────────────────────────────────────
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0): super().__init__(); self.s = smooth
    def forward(self, logits, targets):
        p = torch.softmax(logits,1)[:,1]; t = targets.float()
        i = (p*t).sum((1,2)); u = p.sum((1,2))+t.sum((1,2))
        return 1 - ((2*i+self.s)/(u+self.s)).mean()

class CombinedLoss(nn.Module):
    def __init__(self, bw, dw):
        super().__init__()
        weights = torch.tensor([1.0, 4.0], device=DEVICE)
        self.bce = nn.CrossEntropyLoss(weight=weights)
        self.dice = DiceLoss()
        self.bw = bw
        self.dw = dw
    def forward(self, logits, targets):
        return self.bw*self.bce(logits,targets) + self.dw*self.dice(logits,targets)

# ── Metrics ────────────────────────────────────────────────────────────────────
class ConfusionMeter:
    """
    Dataset-level pixel confusion counts for the binary road task (road = class 1).

    Scores are computed ONCE from the summed counts. They are NOT a mean of per-batch
    scores, which is biased (small/empty batches weigh the same as full ones).
    mIoU here = mean(road IoU, background IoU); road IoU is the stricter number and is
    reported separately because background IoU is ~0.95+ on DeepGlobe and inflates mIoU.
    """
    def __init__(self):
        self.tp = self.fp = self.fn = self.tn = 0

    def update(self, preds, targets):
        p = preds.reshape(-1) == 1
        t = targets.reshape(-1) == 1
        c = torch.stack([(p & t).sum(), (p & ~t).sum(), (~p & t).sum(), (~p & ~t).sum()]).tolist()
        self.tp += int(c[0]); self.fp += int(c[1]); self.fn += int(c[2]); self.tn += int(c[3])

    def summary(self):
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        d = lambda a, b: (a / b) if b > 0 else 0.0
        iou_road = d(tp, tp + fp + fn)
        iou_bg   = d(tn, tn + fp + fn)
        prec, rec = d(tp, tp + fp), d(tp, tp + fn)
        total = tp + fp + fn + tn
        return {
            "iou_road":        iou_road,
            "iou_background":  iou_bg,
            "miou":            (iou_road + iou_bg) / 2,
            "precision":       prec,
            "recall":          rec,
            "f1":              d(2 * prec * rec, prec + rec),
            "dice":            d(2 * tp, 2 * tp + fp + fn),
            "road_pixel_fraction": d(tp + fn, total),
            "pixels":          total,
        }

# ── Safe logits extractor ──────────────────────────────────────────────────────
def get_logits(out):
    if hasattr(out,"logits"): return out.logits
    if isinstance(out,dict) and "logits" in out: return out["logits"]
    if isinstance(out,(list,tuple)): return out[0]
    raise ValueError(f"Cannot extract logits from {type(out)}")

# ── Model ──────────────────────────────────────────────────────────────────────
def build_model(cfg):
    labels = dict(id2label={0: "background", 1: "road"}, label2id={"background": 0, "road": 1})
    if cfg.get("tiny_model"):
        # Tiny random-init SegFormer — exercises the full pipeline on CPU in seconds.
        print("Building TINY random-init SegFormer (test mode)")
        m = SegformerForSemanticSegmentation(SegformerConfig(
            num_labels=cfg["num_labels"], depths=[1, 1, 1, 1], hidden_sizes=[16, 32, 64, 128],
            num_attention_heads=[1, 2, 4, 8], decoder_hidden_size=64, **labels))
    else:
        print(f"Loading {cfg['model_name']}")
        m = SegformerForSemanticSegmentation.from_pretrained(
            cfg["model_name"], num_labels=cfg["num_labels"],
            ignore_mismatched_sizes=True, **labels)
    if torch.cuda.device_count() > 1:
        print(f"DataParallel x{torch.cuda.device_count()}")
        m = nn.DataParallel(m)
    m = m.to(DEVICE)
    print(f"Params: {sum(p.numel() for p in m.parameters()):,}")
    return m

# ── Checkpoint helpers ─────────────────────────────────────────────────────────
LATEST = "latest.pth"
BEST   = "roadrishi_finetuned.pth"
_RESUME_SIDECARS = ("split.json", "history.json", "run_meta.json", BEST)

def _atomic_save(obj, path):
    """torch.save to a temp file, then rename — a crash mid-write never leaves a torn file."""
    tmp = f"{path}.tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)

def _rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}

def _set_rng_state(s):
    if not s: return
    try:
        random.setstate(s["python"]); np.random.set_state(s["numpy"])
        torch.set_rng_state(s["torch"])
        if s.get("cuda") and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(s["cuda"])
    except Exception as e:                       # different GPU count etc. — not fatal
        print(f"  ⚠️  could not restore RNG state: {e}")

def _save_full(path, model, optimizer, scheduler, scaler, *, epoch, batches_done,
               epoch_complete, best_miou, val_loss, log, epochs_without_improvement,
               partial=None):
    """
    Save everything needed to resume training.
      epoch_complete=True  → `epoch` finished; resume starts at epoch+1.
      epoch_complete=False → `batches_done` batches of `epoch` finished; `partial` holds the
                             running loss / confusion counts so epoch metrics stay exact.
    """
    raw = model.module if isinstance(model, nn.DataParallel) else model
    _atomic_save({
        "epoch":                epoch,
        "batches_done":         batches_done,
        "epoch_complete":       epoch_complete,
        "model_state_dict":     raw.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict":    scaler.state_dict(),
        "best_miou":            best_miou,
        "val_loss":             val_loss,
        "log":                  log,
        "epochs_without_improvement": epochs_without_improvement,
        "partial":              partial,
        "rng":                  _rng_state(),
        "config":               CFG,
    }, path)

def _import_previous_run(output_dir):
    """
    Kaggle: a crashed session's /kaggle/working is gone, but its saved Output can be attached
    as an input. Copy the newest resumable checkpoint (+ split/history/meta) into output_dir.
    """
    if not IS_KAGGLE:
        return None
    import glob as _g
    found = [p for pat in (f"/kaggle/input/**/{LATEST}", f"/kaggle/input/**/{BEST}",
                           "/kaggle/input/**/checkpoint_epoch*.pth")
             for p in _g.glob(pat, recursive=True)]
    if not found:
        return None
    src = max(found, key=os.path.getmtime)
    src_dir = Path(src).parent
    print(f"[RESUME] importing previous run from {src_dir}")
    dst = os.path.join(output_dir, LATEST if Path(src).name != BEST else BEST)
    shutil.copy2(src, dst)
    for side in _RESUME_SIDECARS:
        if (src_dir / side).exists() and not os.path.exists(os.path.join(output_dir, side)):
            shutil.copy2(src_dir / side, os.path.join(output_dir, side))
    return dst

def find_resume_checkpoint(output_dir):
    """Returns path of the most advanced resumable checkpoint, or None."""
    if CFG.get("resume_from"):
        return CFG["resume_from"]
    out = Path(output_dir)
    if (out / LATEST).exists():
        return str(out / LATEST)
    periodics = sorted(out.glob("checkpoint_epoch*.pth"))     # older script versions
    if periodics:
        return str(periodics[-1])
    best = out / BEST
    if best.exists():
        try:
            if "optimizer_state_dict" in torch.load(best, map_location="cpu", weights_only=False):
                return str(best)
        except Exception:
            pass
    return _import_previous_run(output_dir)

def load_checkpoint(path, model, optimizer, scheduler, scaler):
    """Load all training state. Returns a dict describing where to resume."""
    print(f"\n  🔄 Resuming from: {path}")
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    raw  = model.module if isinstance(model, nn.DataParallel) else model
    raw.load_state_dict(ckpt["model_state_dict"])
    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if ckpt.get("scaler_state_dict"):
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    _set_rng_state(ckpt.get("rng"))
    epoch = ckpt.get("epoch", 0)
    # Checkpoints from older script versions are always end-of-epoch.
    complete = ckpt.get("epoch_complete", True)
    state = {
        "start_epoch":  epoch + 1 if complete else epoch,
        "start_batch":  0 if complete else ckpt.get("batches_done", 0),
        "partial":      None if complete else ckpt.get("partial"),
        "best_miou":    ckpt.get("best_miou", ckpt.get("val_miou", 0.0)),
        "log":          ckpt.get("log", []),
        "epochs_without_improvement": ckpt.get("epochs_without_improvement", 0),
    }
    where = (f"epoch {state['start_epoch']}" if complete
             else f"epoch {state['start_epoch']}, batch {state['start_batch']}")
    print(f"  ✅ Resuming at {where}  (best mIoU so far: {state['best_miou']:.4f})\n")
    return state

# ── Graceful stop (Ctrl-C / SIGTERM / time budget) ─────────────────────────────
class StopFlag:
    def __init__(self):
        self.reason = None
        self.t0 = time.time()
    def request(self, reason):
        if self.reason is None:
            print(f"\n  ⏸️  {reason} — will checkpoint after the current optimizer step")
        self.reason = reason
    def should_stop(self):
        if self.reason is None and CFG["max_hours"] > 0 and \
                time.time() - self.t0 > CFG["max_hours"] * 3600:
            self.request(f"time budget of {CFG['max_hours']} h reached")
        return self.reason is not None

STOP = StopFlag()

def _install_signal_handlers():
    old = {}
    def handler(signum, _frame):
        STOP.request(f"received {signal.Signals(signum).name}")
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            old[sig] = signal.signal(sig, handler)
        except (ValueError, OSError):          # not on the main thread
            pass
    return old

# ── Safe float format (handles nan/inf without crashing) ─────────────────────
def _fmt(v, fmt=".4f"):
    return f"{v:{fmt}}" if isinstance(v, float) and math.isfinite(v) else str(v)

# ── Training loop (OOM-safe, NaN-safe, checkpoint-able mid-epoch) ─────────────
def _empty_summary():
    nan = float("nan")
    return {k: nan for k in ("iou_road", "iou_background", "miou", "precision",
                             "recall", "f1", "dice", "road_pixel_fraction", "pixels")}

def train_one_epoch(model, loader, optimizer, criterion, scaler, epoch,
                    start_batch=0, partial=None, save_partial=None):
    """
    Trains on loader (whose sampler already skips the first `start_batch` batches).
    `partial` restores the running totals of an interrupted epoch.
    `save_partial(batches_done, partial_dict)` is called every ckpt_every_batches.
    Returns (mean_loss, summary_dict, completed: bool).
    """
    model.train()
    p = partial or {}
    valid, skipped_oom, skipped_nan = p.get("valid", 0), p.get("oom", 0), p.get("nan", 0)
    total_loss = p.get("loss_sum", 0.0)
    meter      = ConfusionMeter()
    meter.tp, meter.fp, meter.fn, meter.tn = p.get("confusion", (0, 0, 0, 0))
    use_amp    = CFG.get("use_amp", True)
    grad_accum = CFG.get("grad_accum", 1)
    every      = CFG["ckpt_every_batches"]
    total_batches = start_batch + len(loader)
    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=True,
                initial=start_batch, total=total_batches, mininterval=_TQDM_INTERVAL)
    optimizer.zero_grad()

    def _partial():
        return {"valid": valid, "oom": skipped_oom, "nan": skipped_nan, "loss_sum": total_loss,
                "confusion": (meter.tp, meter.fp, meter.fn, meter.tn)}

    for i, batch in enumerate(pbar):
        b = start_batch + i + 1                 # batches of this epoch completed after this one
        try:
            pv = batch["pixel_values"].to(DEVICE, non_blocking=True)
            lb = batch["labels"].to(DEVICE, non_blocking=True)

            _dev = _AMP_DEVICE if _AMP_DEVICE else "cuda"
            with autocast(device_type=_dev, enabled=use_amp):
                out    = model(pixel_values=pv)
                logits = get_logits(out)
                up     = nn.functional.interpolate(logits, size=lb.shape[-2:],
                                                   mode="bilinear", align_corners=False)
                loss   = criterion(up, lb) / grad_accum

            # NaN guard: check BEFORE backward to avoid graph corruption
            raw_loss = loss.item() * grad_accum
            if not math.isfinite(raw_loss):
                skipped_nan += 1
                optimizer.zero_grad()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                if skipped_nan > 10 and use_amp:
                    print("\n  ⚠️  >10 NaN steps — disabling AMP and continuing in fp32")
                    CFG["use_amp"] = False
                    use_amp = False
                continue

            scaler.scale(loss).backward()

            stepped = False
            if b % grad_accum == 0 or b == total_batches:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() < old_scale:   # scaler saw inf/NaN grads
                    skipped_nan += 1
                optimizer.zero_grad()
                stepped = True

            with torch.no_grad():
                meter.update(up.argmax(1), lb)
            total_loss += raw_loss
            valid += 1
            pbar.set_postfix(loss=f"{raw_loss:.4f}", oom=skipped_oom, nan=skipped_nan,
                             refresh=False)

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                skipped_oom += 1
                torch.cuda.empty_cache(); optimizer.zero_grad()
                if skipped_oom > 20:
                    raise RuntimeError("Too many OOM errors. Reduce batch_size.") from e
                stepped = (b % grad_accum == 0)
            else:
                raise

        if _CRASH_AFTER and b >= _CRASH_AFTER and epoch == 1:
            print(f"\n  💥 [test hook] simulated crash after {b} batches", flush=True)
            os._exit(137)

        # Mid-epoch checkpoint / graceful stop — only on optimizer-step boundaries
        if stepped and b < total_batches and save_partial is not None:
            if STOP.should_stop():
                save_partial(b, _partial())
                pbar.close()
                return float("nan"), _empty_summary(), False
            if b % every == 0:
                save_partial(b, _partial())

    pbar.close()
    if valid == 0:
        return float("nan"), _empty_summary(), True
    if skipped_oom + skipped_nan:
        print(f"  ℹ️  Skipped {skipped_oom} OOM + {skipped_nan} NaN out of {total_batches}")
    return total_loss / valid, meter.summary(), True

# ── Eval loop ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, criterion, epoch, use_amp=None, per_image=False):
    """
    Returns (mean_loss, summary_dict, per_image_rows).
    per_image_rows (only if per_image=True) = [{name, tp, fp, fn, iou_road}] in dataset order.
    The loader MUST have shuffle=False for names to line up.
    """
    model.eval()
    valid  = 0
    total_loss = 0
    meter  = ConfusionMeter()
    rows   = []
    idx    = 0
    use_amp = CFG.get("use_amp", True) if use_amp is None else use_amp
    for batch in tqdm(loader, desc=f"Epoch {epoch} [val]  ", leave=False, mininterval=_TQDM_INTERVAL):
        start = idx
        idx  += batch["labels"].shape[0]
        try:
            pv = batch["pixel_values"].to(DEVICE, non_blocking=True)
            lb = batch["labels"].to(DEVICE, non_blocking=True)
            _dev = _AMP_DEVICE if _AMP_DEVICE else "cuda"
            with autocast(device_type=_dev, enabled=use_amp):
                out    = model(pixel_values=pv)
                logits = get_logits(out)
                up     = nn.functional.interpolate(logits, size=lb.shape[-2:],
                                                   mode="bilinear", align_corners=False)
                loss   = criterion(up, lb)
            if not math.isfinite(loss.item()): continue
            total_loss += loss.item()
            pred = up.argmax(1)
            meter.update(pred, lb)
            if per_image:
                p, t = pred == 1, lb == 1
                tp = (p & t).sum((1, 2)).tolist()
                fp = (p & ~t).sum((1, 2)).tolist()
                fn = (~p & t).sum((1, 2)).tolist()
                for j in range(len(tp)):
                    den = tp[j] + fp[j] + fn[j]
                    rows.append({
                        "name": Path(loader.dataset.imgs[start + j]).name,
                        "tp": int(tp[j]), "fp": int(fp[j]), "fn": int(fn[j]),
                        "iou_road": (tp[j] / den) if den > 0 else 1.0,
                    })
            valid += 1
        except RuntimeError as e:
            if "out of memory" in str(e).lower(): torch.cuda.empty_cache()
            else: raise
    if valid == 0:
        return float("nan"), _empty_summary(), rows
    return total_loss / valid, meter.summary(), rows

# ── Run metadata & result artefacts ────────────────────────────────────────────
def _r(v, nd=4):
    """Round finite floats; map nan/inf to None so the JSON stays valid."""
    return round(v, nd) if isinstance(v, float) and math.isfinite(v) else None

def _git_commit():
    try:
        here = Path(__file__).resolve().parent
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=here,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None

def _script_sha256():
    try:
        import hashlib
        return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except Exception:
        return None

def _jdump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def write_run_meta(n_train, n_val):
    """run_meta.json: everything needed to reproduce / audit the run."""
    import datetime, platform
    path = os.path.join(CFG["output_dir"], "run_meta.json")
    now  = datetime.datetime.now(datetime.timezone.utc).isoformat()
    if os.path.exists(path):                       # resumed run: just note it
        meta = json.load(open(path))
        meta.setdefault("resumed_at", []).append(now)
        _jdump(meta, path)
        return
    import transformers
    meta = {
        "started_at":     now,
        "git_commit":     _git_commit(),           # None on Kaggle (no repo) — see script_sha256
        "script_sha256":  _script_sha256(),
        "seed":           CFG["seed"],
        "config":         CFG,
        "augmentations":  [type(t).__name__ for t in train_transform.transforms],
        "dataset":        {"name": "DeepGlobe Road Extraction (labelled train split)",
                           "root": CFG["data_root"], "n_train": n_train, "n_val": n_val,
                           "native_size_px": 2448, "native_gsd_m": 0.5},
        # Honest resolution note: images are resized 2448 -> img_size before the model,
        # so the reported scores are at a much coarser ground-sample distance than 0.5 m.
        "effective_gsd_m": round(0.5 * 2448 / CFG["img_size"], 2),
        "versions":       {"python": platform.python_version(), "torch": torch.__version__,
                           "transformers": transformers.__version__,
                           "albumentations": A.__version__},
        "gpus":           [torch.cuda.get_device_properties(i).name
                           for i in range(torch.cuda.device_count())],
    }
    _jdump(meta, path)

def save_curves(history, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nan = float("nan")
    col = lambda k: [(r.get(k) if r.get(k) is not None else nan) for r in history]
    ep  = [r["epoch"] for r in history]
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    ax[0].plot(ep, col("train_loss"), "o-", label="train"); ax[0].plot(ep, col("val_loss"), "o-", label="val")
    ax[0].set_title("Loss (CE + Dice)")
    ax[1].plot(ep, col("train_miou"), "o--", label="train mIoU")
    ax[1].plot(ep, col("val_miou"), "o-", label="val mIoU")
    ax[1].plot(ep, col("val_iou_road"), "s-", label="val road IoU")
    ax[1].set_title("IoU")
    ax[2].plot(ep, col("val_precision"), "o-", label="precision")
    ax[2].plot(ep, col("val_recall"), "o-", label="recall")
    ax[2].plot(ep, col("val_f1"), "o-", label="F1")
    ax[2].set_title("Validation road-class P / R / F1")
    for a in ax:
        a.set_xlabel("epoch"); a.grid(alpha=.3); a.legend()
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)

@torch.no_grad()
def save_prediction_grid(model, dataset, per_img, path, k=2):
    """Rows: worst-k / median-k / best-k val images by road IoU. Cols: image | GT | error map."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ranked = sorted(per_img, key=lambda r: r["iou_road"])
    n, mid = len(ranked), len(ranked) // 2
    picks  = ([("worst", r) for r in ranked[:k]] +
              [("median", r) for r in ranked[max(0, mid - k // 2): max(0, mid - k // 2) + k]] +
              [("best", r) for r in ranked[-k:]])
    index  = {Path(p).name: i for i, p in enumerate(dataset.imgs)}
    mean   = np.array([0.485, 0.456, 0.406]); std = np.array([0.229, 0.224, 0.225])
    model.eval()
    fig, axes = plt.subplots(len(picks), 3, figsize=(10.5, 3.4 * len(picks)))
    for row, (tag, r) in enumerate(picks):
        s  = dataset[index[r["name"]]]
        pv = s["pixel_values"].unsqueeze(0).to(DEVICE)
        lg = nn.functional.interpolate(get_logits(model(pixel_values=pv)), size=pv.shape[-2:],
                                       mode="bilinear", align_corners=False)
        pred = lg.argmax(1)[0].cpu().numpy() == 1
        gt   = s["labels"].numpy() == 1
        img  = np.clip(s["pixel_values"].permute(1, 2, 0).numpy() * std + mean, 0, 1)
        err  = np.zeros((*gt.shape, 3))
        err[pred & gt] = (0.1, 0.9, 0.3); err[pred & ~gt] = (0.95, 0.2, 0.2); err[~pred & gt] = (0.2, 0.4, 1.0)
        for c, (im, title) in enumerate([(img, f"{tag}: {r['name']}  IoU={r['iou_road']:.3f}"),
                                         (gt, "ground truth"),
                                         (err, "green=TP  red=FP  blue=FN")]):
            axes[row, c].imshow(im, cmap="gray" if im.ndim == 2 else None)
            axes[row, c].set_title(title, fontsize=8); axes[row, c].axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)

def finalize(model, vl, criterion, history, best_miou):
    """
    Re-evaluates the BEST checkpoint on the held-out split (fp32) and writes the
    result files. Every artefact is independent: one failing never loses the others.
    """
    out = CFG["output_dir"]
    _jdump(history, os.path.join(out, "history.json"))
    best_path = os.path.join(out, "roadrishi_finetuned.pth")
    if not os.path.exists(best_path):
        print("[finalize] no best checkpoint — skipping held-out evaluation"); return
    ckpt = torch.load(best_path, map_location=DEVICE, weights_only=False)
    raw  = model.module if isinstance(model, nn.DataParallel) else model
    raw.load_state_dict(ckpt["model_state_dict"])

    loss, summ, per_img = evaluate(model, vl, criterion, epoch="final", use_amp=False, per_image=True)
    ious = np.array([r["iou_road"] for r in per_img]) if per_img else np.array([float("nan")])
    metrics = {
        "checkpoint":        "roadrishi_finetuned.pth",
        "checkpoint_epoch":  ckpt.get("epoch"),
        "protocol": {
            "split":         "held-out validation images (see split.json)",
            "n_val_images":  len(per_img),
            "input_size_px": CFG["img_size"],
            "effective_gsd_m": round(0.5 * 2448 / CFG["img_size"], 2),
            "precision":     "fp32, argmax over 2 classes, metrics from summed pixel counts",
            "seed":          CFG["seed"],
        },
        "val": {"loss": _r(loss), **{k: (_r(v) if k != "pixels" else v) for k, v in summ.items()}},
        "per_image_road_iou": {
            "mean": _r(float(np.mean(ious))), "median": _r(float(np.median(ious))),
            "p10":  _r(float(np.percentile(ious, 10))),
            "min":  _r(float(np.min(ious))), "max": _r(float(np.max(ious))),
        },
        "best_epoch_val_miou_during_training": _r(best_miou),
    }
    _jdump(metrics, os.path.join(out, "metrics.json"))
    _jdump(per_img, os.path.join(out, "per_image_val.json"))
    print("\n  📊 HELD-OUT RESULTS (best checkpoint)")
    print(f"     mIoU={summ['miou']:.4f}  road IoU={summ['iou_road']:.4f}  "
          f"P={summ['precision']:.4f}  R={summ['recall']:.4f}  F1={summ['f1']:.4f}  "
          f"(n={len(per_img)})")

    for label, fn in (("curves.png",           lambda p: save_curves(history, p)),
                      ("predictions_grid.png", lambda p: save_prediction_grid(model, vl.dataset, per_img, p))):
        try:
            fn(os.path.join(out, label)); print(f"  🖼️  wrote {label}")
        except Exception as e:             # plotting must never cost us the metrics
            print(f"  ⚠️  could not write {label}: {e}")

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    """Returns "done" | "early_stopped" | "interrupted"."""
    old_handlers = _install_signal_handlers()
    try:
        return _main()
    finally:
        for sig, h in old_handlers.items():
            signal.signal(sig, h)

def _main():
    STOP.__init__()                              # fresh flag/clock (notebook re-runs)
    # Locate (and on Kaggle, import) a resumable run BEFORE building the split,
    # so an imported split.json is the one that gets used.
    resume = find_resume_checkpoint(CFG["output_dir"])
    tl, vl, sampler = build_dataloaders(CFG)
    write_run_meta(len(tl.dataset), len(vl.dataset))
    model     = build_model(CFG)
    criterion = CombinedLoss(CFG["bce_weight"], CFG["dice_weight"])
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=2, T_mult=2, eta_min=1e-6)
    scaler    = GradScaler("cuda", enabled=CFG.get("use_amp", True))

    state = {"start_epoch": 1, "start_batch": 0, "partial": None, "best_miou": 0.0,
             "log": [], "epochs_without_improvement": 0}
    if resume:
        state = load_checkpoint(resume, model, optimizer, scheduler, scaler)
    else:
        print("ℹ️  No checkpoint found — starting fresh.")
    best_miou = state["best_miou"]
    log       = state["log"]
    patience_count = state["epochs_without_improvement"]
    latest_path = os.path.join(CFG["output_dir"], LATEST)
    best_path   = os.path.join(CFG["output_dir"], BEST)

    if state["start_epoch"] > CFG["num_epochs"] or patience_count >= CFG["early_stopping_patience"]:
        print("✅ Training already complete.")
        finalize(model, vl, criterion, log, best_miou)
        return "done"

    print("\n" + "="*60)
    print(f"  RoadRishi SegFormer  |  epochs {state['start_epoch']}–{CFG['num_epochs']}")
    print(f"  img={CFG['img_size']}  bs={CFG['batch_size']}  "
          f"eff_bs={CFG['batch_size']*CFG['grad_accum']}  AMP={CFG['use_amp']}")
    print("="*60)
    check_vram()

    outcome = "done"
    for epoch in range(state["start_epoch"], CFG["num_epochs"]+1):
        t0 = time.time()
        start_batch = state["start_batch"] if epoch == state["start_epoch"] else 0
        partial     = state["partial"]     if epoch == state["start_epoch"] else None
        sampler.set_epoch(epoch, start_batch * CFG["batch_size"])

        def save_partial(batches_done, part, _epoch=epoch):
            _save_full(latest_path, model, optimizer, scheduler, scaler,
                       epoch=_epoch, batches_done=batches_done, epoch_complete=False,
                       best_miou=best_miou, val_loss=None, log=log,
                       epochs_without_improvement=patience_count, partial=part)

        train_loss, tr, completed = train_one_epoch(
            model, tl, optimizer, criterion, scaler, epoch,
            start_batch=start_batch, partial=partial, save_partial=save_partial)
        if not completed:
            outcome = "interrupted"
            break

        val_loss, va, _ = evaluate(model, vl, criterion, epoch)
        scheduler.step()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        val_miou = va["miou"]

        elapsed = time.time()-t0
        eta     = elapsed * (CFG["num_epochs"]-epoch)
        print(
            f"Epoch {epoch:2d}/{CFG['num_epochs']} | "
            f"Train Loss={_fmt(train_loss)} mIoU={_fmt(tr['miou'])} | "
            f"Val Loss={_fmt(val_loss)} mIoU={_fmt(va['miou'])} roadIoU={_fmt(va['iou_road'])} "
            f"P={_fmt(va['precision'])} R={_fmt(va['recall'])} F1={_fmt(va['f1'])} | "
            f"{elapsed/60:.1f} min | ETA {eta/3600:.1f} h"
        )

        row = {
            "epoch":        epoch,
            "train_loss":   _r(train_loss),   "val_loss":      _r(val_loss),
            "train_miou":   _r(tr["miou"]),   "val_miou":      _r(va["miou"]),
            "train_iou_road": _r(tr["iou_road"]), "val_iou_road": _r(va["iou_road"]),
            "train_dice":   _r(tr["dice"]),   "val_dice":      _r(va["dice"]),
            "train_f1":     _r(tr["f1"]),     "val_f1":        _r(va["f1"]),
            "val_precision": _r(va["precision"]), "val_recall": _r(va["recall"]),
            "train_pixels": tr["pixels"],     # sanity: identical for crashed+resumed epochs
            "lr":           round(scheduler.get_last_lr()[0], 8),
            "secs":         round(elapsed, 1),
            "resumed_mid_epoch": bool(start_batch),
        }
        log = [r for r in log if r["epoch"] != epoch] + [row]

        # Flush history every epoch — BEFORE any break, so early stopping loses nothing
        _jdump(log, os.path.join(CFG["output_dir"], "history.json"))

        if math.isfinite(val_miou) and val_miou > best_miou + 1e-4:
            best_miou = val_miou
            patience_count = 0
            _save_full(best_path, model, optimizer, scheduler, scaler,
                       epoch=epoch, batches_done=0, epoch_complete=True, best_miou=best_miou,
                       val_loss=val_loss, log=log, epochs_without_improvement=patience_count)
            print(f"  ✅ Best saved → val_mIoU={val_miou:.4f}")
        else:
            patience_count += 1

        # End-of-epoch resumable state (replaces the old every-2-epochs periodic files)
        _save_full(latest_path, model, optimizer, scheduler, scaler,
                   epoch=epoch, batches_done=0, epoch_complete=True, best_miou=best_miou,
                   val_loss=val_loss, log=log, epochs_without_improvement=patience_count)

        if patience_count >= CFG["early_stopping_patience"]:
            print(f"\n🛑 Early stopping triggered ({patience_count} epochs without improvement)")
            outcome = "early_stopped"
            break
        if STOP.should_stop():
            outcome = "interrupted"
            break

    if outcome == "interrupted":
        print("\n" + "="*60)
        print(f"  ⏸️  Stopped ({STOP.reason}). State saved to {latest_path}")
        print("     Run the script again to resume exactly where it stopped.")
        print("="*60)
        return outcome

    print("\n" + "="*60)
    print(f"  Training done. Best val mIoU during training: {best_miou:.4f}")
    finalize(model, vl, criterion, log, best_miou)
    print(f"  Outputs in: {CFG['output_dir']}")
    print("    roadrishi_finetuned.pth  metrics.json  history.json  run_meta.json")
    print("    split.json  per_image_val.json  curves.png  predictions_grid.png")
    print("="*60)
    return outcome


if __name__ == "__main__":
    _outcome = main()
    if "__file__" in globals():                 # CLI: non-zero exit if interrupted
        sys.exit(0 if _outcome in ("done", "early_stopped") else 3)
