import cv2
import numpy as np
import random
from pathlib import Path
import shutil

# ─── CONFIG ───────────────────────────────────────────────────────────
# Set seed for reproducibility
random.seed(42)
# Proportion of data to use for training
TRAIN_SPLIT = 0.7

# ─── PATHS ────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent  # script folder
HQ_DIR     = BASE_DIR / "data" / "HQ_DWI"       #  HQ PNGs
LQ_ROOT    = BASE_DIR / "data" / "LQ_DWI"       #  10 LQs per each HQ PNGs
TRAIN_DIR  = BASE_DIR / "pytorch-CycleGAN-and-pix2pix" / "data" / "train"
TEST_DIR   = BASE_DIR / "pytorch-CycleGAN-and-pix2pix" / "data" / "test"

# ─── PREPARE OUTPUT FOLDERS ───────────────────────────────────────────
shutil.rmtree(TRAIN_DIR, ignore_errors=True)
shutil.rmtree(TEST_DIR, ignore_errors=True)
TRAIN_DIR.mkdir(parents=True)
TEST_DIR.mkdir(parents=True)

count_train = 0
count_test = 0
missing = 0

# ─── PROCESS EACH HQ IMAGE ────────────────────────────────────────────
for hq_path in HQ_DIR.glob("*.png"):
    lq_dir = LQ_ROOT / hq_path.stem
    if not lq_dir.is_dir():
        print(f"❌ LQ folder missing for {hq_path.stem}")
        missing += 1
        continue

    # Load HQ image once
    hq_img = cv2.imread(str(hq_path), cv2.IMREAD_GRAYSCALE)
    if hq_img is None:
        print(f"⚠️  Couldn’t read {hq_path}")
        continue
    h, w = hq_img.shape

    # Iterate over each LQ image
    for lq_path in lq_dir.glob("*.png"):
        lq_img = cv2.imread(str(lq_path), cv2.IMREAD_GRAYSCALE)
        if lq_img is None:
            print(f"⚠️  Skipping unreadable {lq_path.name}")
            continue

        # Resize LQ to match HQ if needed
        if lq_img.shape != hq_img.shape:
            lq_img = cv2.resize(lq_img, (w, h), interpolation=cv2.INTER_CUBIC)

        # Concatenate side-by-side (A|B)
        ab = np.concatenate([lq_img, hq_img], axis=1)

        # Randomly assign to train or test
        if random.random() < TRAIN_SPLIT:
            out_dir = TRAIN_DIR
            count_train += 1
        else:
            out_dir = TEST_DIR
            count_test += 1

        # Save AB image with original LQ filename
        cv2.imwrite(str(out_dir / lq_path.name), ab)

# ─── SUMMARY ───────────────────────────────────────────────────────────
print(f"Built {count_train} train AB PNGs → {TRAIN_DIR}")
print(f"Built {count_test} test AB PNGs → {TEST_DIR}")
if missing:
    print(f" {missing} HQs had no matching LQ folder")

