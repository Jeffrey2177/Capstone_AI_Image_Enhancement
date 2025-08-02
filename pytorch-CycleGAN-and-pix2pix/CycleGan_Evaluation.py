#!/usr/bin/env python3
import os
import glob
import re
import argparse
import numpy as np
import cv2
from skimage.metrics import structural_similarity, peak_signal_noise_ratio, mean_squared_error

def parse_args():
    p = argparse.ArgumentParser(description="Groupwise evaluation of CycleGAN outputs")
    p.add_argument(
        "--results_dir",
        default=os.path.join(os.path.dirname(__file__), "results", "cycle_test", "test_latest", "images"),
        help="Where your *_fake_B.png and *_real_B.png live"
    )
    p.add_argument(
        "--imread_flag", type=int, default=cv2.IMREAD_GRAYSCALE,
        help="cv2.IMREAD_GRAYSCALE or IMREAD_COLOR"
    )
    p.add_argument(
        "--filter", type=str, default=None,
        help="Only evaluate files containing this substring"
    )
    p.add_argument(
        "--group_by", nargs="+",
        choices=["batch", "inpaint", "b"],
        default=["b"],
        help="Which metadata fields to group metrics by"
    )
    return p.parse_args()

def extract_metadata(fname):
    # e.g. batch_1_inpaint_3_..._b50_fake_B.png
    meta = {}
    parts = fname.split("_")
    for part in parts:
        if part.startswith("batch") and part != "batch":
            meta["batch"] = part.replace("batch", "")
        if part.startswith("inpaint"):
            meta["inpaint"] = part.replace("inpaint", "")
        m = re.match(r"b(\d+)", part)
        if m:
            meta["b"] = m.group(1)
    return meta

def main():
    args = parse_args()
    pattern = os.path.join(args.results_dir, "*_fake_B.*")
    fake_list = sorted(glob.glob(pattern))
    if args.filter:
        fake_list = [p for p in fake_list if args.filter in os.path.basename(p)]
    if not fake_list:
        print("No files found! Check your --results_dir and --filter.")
        return

    # metrics grouped into dict: key tuple -> list of (ssim, psnr, mse)
    groups = {}
    for fake_path in fake_list:
        fname = os.path.basename(fake_path)
        real_path = fake_path.replace("_fake_B", "_real_B")
        if not os.path.exists(real_path):
            print(f"⚠️  Missing real image for {fname}")
            continue

        # load and normalize
        F = cv2.imread(fake_path, args.imread_flag).astype(np.float32)/255.0
        R = cv2.imread(real_path, args.imread_flag).astype(np.float32)/255.0

        # compute
        s = structural_similarity(R, F, data_range=1.0)
        p = peak_signal_noise_ratio(R, F, data_range=1.0)
        m = mean_squared_error(R, F)

        # figure out group key
        meta = extract_metadata(fname)
        key = tuple(meta.get(f, "all") for f in args.group_by)
        groups.setdefault(key, []).append((s, p, m))

    # print results
    for key, vals in groups.items():
        ss = [v[0] for v in vals]
        ps = [v[1] for v in vals]
        ms = [v[2] for v in vals]
        label = ", ".join(f"{f}={k}" for f, k in zip(args.group_by, key))
        print(f"=== Group: {label} ({len(vals)} images) ===")
        print(f"  SSIM: {np.mean(ss):.4f} ± {np.std(ss):.4f}")
        print(f"  PSNR: {np.mean(ps):.2f} dB ± {np.std(ps):.2f} dB")
        print(f"  MSE : {np.mean(ms):.6f} ± {np.std(ms):.6f}")
        print()

if __name__ == "__main__":
    main()
