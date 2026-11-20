"""
End-to-end smoke tests for training/kaggle_segformer_train.py.

Runs the REAL script (subprocess, CPU, tiny random-init SegFormer) on a synthetic
DeepGlobe-style dataset and checks:
  * a full run completes and writes every artefact
  * GridDistortion is part of the augmentation pipeline
  * a hard crash mid-epoch (os._exit) resumes from the mid-epoch checkpoint and ends with
    weights bit-identical to an uninterrupted run
  * a graceful stop (time budget → same path as SIGINT/SIGTERM) does the same

Run:  venv/bin/python -m pytest tests/test_training_smoke.py -v
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

ROOT   = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "training" / "kaggle_segformer_train.py"
N_IMAGES, IMG = 20, 64        # 18 train / 2 val → 9 batches/epoch at batch 2


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    d = tmp_path_factory.mktemp("deepglobe")
    rng = np.random.default_rng(0)
    for k in range(N_IMAGES):
        sat  = Image.new("RGB", (96, 96), tuple(int(c) for c in rng.integers(40, 120, 3)))
        mask = Image.new("L", (96, 96), 0)
        ds, dm = ImageDraw.Draw(sat), ImageDraw.Draw(mask)
        for _ in range(2):
            p = [tuple(int(v) for v in rng.integers(0, 96, 2)) for _ in range(2)]
            ds.line(p, fill=(200, 200, 190), width=6)
            dm.line(p, fill=255, width=6)
        sat.save(d / f"{k}_sat.jpg")
        mask.save(d / f"{k}_mask.png")
    return d


def run(dataset, out, **env_extra):
    env = {**os.environ,
           "ROADRISHI_DATA_ROOT": str(dataset), "ROADRISHI_OUTPUT_DIR": str(out),
           "ROADRISHI_TINY_MODEL": "1", "ROADRISHI_IMG": str(IMG), "ROADRISHI_BATCH": "2",
           "ROADRISHI_ACCUM": "2", "ROADRISHI_EPOCHS": "2", "ROADRISHI_WORKERS": "0",
           "ROADRISHI_CKPT_EVERY": "2", "ROADRISHI_MAX_HOURS": "0", "ROADRISHI_PATIENCE": "5",
           "ROADRISHI_TQDM_INTERVAL": "5", "CUDA_VISIBLE_DEVICES": ""}
    env.update({f"ROADRISHI_{k}": str(v) for k, v in env_extra.items()})
    p = subprocess.run([sys.executable, str(SCRIPT)], env=env, capture_output=True,
                       text=True, timeout=600)
    return p.returncode, p.stdout + p.stderr


def weights(out):
    return torch.load(out / "latest.pth", map_location="cpu", weights_only=False)["model_state_dict"]


def assert_same_weights(a, b):
    assert a.keys() == b.keys()
    diff = [k for k in a if not torch.equal(a[k], b[k])]
    assert not diff, f"{len(diff)} tensors differ, e.g. {diff[:3]}"


@pytest.fixture(scope="module")
def reference(dataset, tmp_path_factory):
    out = tmp_path_factory.mktemp("ref")
    code, log = run(dataset, out)
    assert code == 0, log[-3000:]
    return out, log


def test_full_run_writes_all_artifacts(reference):
    out, log = reference
    for f in ("roadrishi_finetuned.pth", "latest.pth", "metrics.json", "history.json",
              "run_meta.json", "split.json", "per_image_val.json", "curves.png",
              "predictions_grid.png"):
        assert (out / f).exists(), f"missing {f}"
    hist = json.load(open(out / "history.json"))
    assert [r["epoch"] for r in hist] == [1, 2]
    n_train = len(json.load(open(out / "split.json"))["train"])
    assert all(r["train_pixels"] == n_train * IMG * IMG for r in hist)
    m = json.load(open(out / "metrics.json"))
    assert 0.0 <= m["val"]["iou_road"] <= 1.0 and 0.0 <= m["val"]["miou"] <= 1.0


def test_griddistortion_in_pipeline(reference):
    out, _ = reference
    augs = json.load(open(out / "run_meta.json"))["augmentations"]
    assert "GridDistortion" in augs and "ElasticTransform" in augs


def test_hard_crash_resumes_bit_exact(dataset, reference, tmp_path):
    ref_out, _ = reference
    code, log = run(dataset, tmp_path, CRASH_AFTER_BATCHES=5)
    assert code == 137, log[-2000:]
    ck = torch.load(tmp_path / "latest.pth", map_location="cpu", weights_only=False)
    assert ck["epoch"] == 1 and ck["epoch_complete"] is False and ck["batches_done"] == 4

    code, log = run(dataset, tmp_path)
    assert code == 0, log[-3000:]
    assert "Resuming at epoch 1, batch 4" in log
    hist = json.load(open(tmp_path / "history.json"))
    ref_hist = json.load(open(ref_out / "history.json"))
    assert [r["epoch"] for r in hist] == [1, 2]
    assert hist[0]["resumed_mid_epoch"] is True
    # Every training pixel counted exactly once despite the crash
    assert [r["train_pixels"] for r in hist] == [r["train_pixels"] for r in ref_hist]
    assert_same_weights(weights(tmp_path), weights(ref_out))
    assert hist[-1]["val_miou"] == ref_hist[-1]["val_miou"]


def test_graceful_stop_resumes_bit_exact(dataset, reference, tmp_path):
    ref_out, _ = reference
    code, log = run(dataset, tmp_path, MAX_HOURS="1e-9")   # budget expires immediately
    assert code == 3, log[-2000:]
    assert "time budget" in log and "State saved" in log
    ck = torch.load(tmp_path / "latest.pth", map_location="cpu", weights_only=False)
    assert ck["epoch_complete"] is False and ck["batches_done"] == 2

    code, log = run(dataset, tmp_path)
    assert code == 0, log[-3000:]
    assert_same_weights(weights(tmp_path), weights(ref_out))


def test_completed_run_is_not_retrained(dataset, reference, tmp_path):
    ref_out, _ = reference
    import shutil
    for f in ref_out.iterdir():
        shutil.copy2(f, tmp_path / f.name)
    code, log = run(dataset, tmp_path)
    assert code == 0 and "Training already complete" in log
