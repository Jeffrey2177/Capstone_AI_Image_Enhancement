# eval_metrics_with_fid_lpips.py
import os
import re
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# ─────────────────── MENU 1 ────────────────────────────────────────────────────
print("""
Choose which results to evaluate:
  [1] My own trained model outputs
  [2] Pretrained facades model outputs
  [3] Pretrained edges2shoes model outputs
""")
choice = input("Enter 1, 2 or 3: ").strip()

if choice == "1":
    RES_DIR = Path("results/pix2pix_main/test_latest/images")

    # ───────────────── MENU 2 (baseline band) ──────────────────────────────────
    print("""
    Evaluate which baseline band?
      [1] All baselines (no filter)
      [2] B0-300   → {b50,  b132,  b214,  b296}
      [3] B301-600 → {b378, b460,  b542}
      [4] B601-900 → {b625, b707,  b789,  b871}
      [5] B901+    → {b953, b1035, b1117, b1200}
    """)
    sel = input("Enter 1-5: ").strip()

    BAND_LOOKUP = {
        "1": None,  # no filtering
        "2": {50, 132, 214, 296},
        "3": {378, 460, 542},
        "4": {625, 707, 789, 871},
        "5": {953, 1035, 1117, 1200},
    }
    allowed_b_values = BAND_LOOKUP.get(sel, None)
else:
    allowed_b_values = None  # pretrained sets → evaluate everything
    if choice == "2":
        RES_DIR = Path("results/facades_label2photo/test_latest/images")
    elif choice == "3":
        RES_DIR = Path("results/edges2shoes/test_latest/images")
    else:
        raise SystemExit("Invalid choice — please rerun and enter 1, 2 or 3")

print(f"\nEvaluating images in: {RES_DIR}")
if allowed_b_values is not None:
    print(f"Baseline filter: {sorted(allowed_b_values)}")
print()

# ──────────────── RUNTIME CONFIG ───────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

fid_metric   = FrechetInceptionDistance(feature=2048, reset_real_features=False).to(DEVICE)
lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(DEVICE)

# gray (cv2) → 3-ch float tensor [0,1] resized to 256×256
to_float = T.Compose([
    T.ToPILImage(),
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Lambda(lambda x: x.repeat(3, 1, 1))
])

def to_uint8(t: torch.Tensor) -> torch.Tensor:
    return (t * 255).round().clamp(0, 255).to(torch.uint8)

def to_lpips(t: torch.Tensor) -> torch.Tensor:
    return t * 2.0 - 1.0

# ──────────────── HELPERS ──────────────────────────────────────────────────────
# Match 'b50_', 'B1200-', etc.  (digits not immediately followed by another digit)
BVAL_RE = re.compile(r"[bB](\d{1,4})(?!\d)")

def extract_bval(name: str) -> int | None:
    m = BVAL_RE.search(name)
    return int(m.group(1)) if m else None

# ──────────────── METRIC STORAGE ───────────────────────────────────────────────
ssim_vals, psnr_vals, mse_vals = [], [], []

# ──────────────── MAIN LOOP ────────────────────────────────────────────────────
for fname in sorted(os.listdir(RES_DIR)):
    if not fname.endswith("_fake_B.png"):
        continue

    # optional baseline filter (own-model case only)
    bval = extract_bval(fname)
    if allowed_b_values is not None:
        if bval is None or bval not in allowed_b_values:
            continue

    fake_path = RES_DIR / fname
    real_path = RES_DIR / fname.replace("_fake_B.png", "_real_B.png")

    fake = cv2.imread(str(fake_path), cv2.IMREAD_GRAYSCALE)
    real = cv2.imread(str(real_path), cv2.IMREAD_GRAYSCALE)
    if fake is None or real is None:
        print(f"Skipping {fname} (missing pair).")
        continue

    if real.shape != fake.shape:
        fake = cv2.resize(fake, (real.shape[1], real.shape[0]))

    # classical metrics
    rng = real.max() - real.min()
    ssim_vals.append(ssim(real, fake, data_range=rng))
    psnr_vals.append(psnr(real, fake, data_range=rng))
    mse_vals.append(mse(real, fake))

    # deep-feature metrics
    real_f32 = to_float(real)
    fake_f32 = to_float(fake)

    fid_metric.update(to_uint8(real_f32).unsqueeze(0).to(DEVICE), real=True)
    fid_metric.update(to_uint8(fake_f32).unsqueeze(0).to(DEVICE), real=False)

    lpips_metric.update(to_lpips(fake_f32).unsqueeze(0).to(DEVICE),
                        to_lpips(real_f32).unsqueeze(0).to(DEVICE))

# ──────────────── SAFEGUARD ────────────────────────────────────────────────────
if len(ssim_vals) == 0:
    print("No images matched the selected baseline band – nothing to evaluate.")
    raise SystemExit

# ──────────────── RESULTS ──────────────────────────────────────────────────────
print(f"Images evaluated: {len(ssim_vals)}")
print(f"Mean SSIM: {np.mean(ssim_vals):.4f} ± {np.std(ssim_vals):.4f}")
print(f"Mean PSNR: {np.mean(psnr_vals):.2f} dB ± {np.std(psnr_vals):.2f} dB")
print(f"Mean MSE:  {np.mean(mse_vals):.2f} ± {np.std(mse_vals):.2f}")
print(f"FID:   {fid_metric.compute().item():.2f}")
print(f"LPIPS: {lpips_metric.compute().item():.4f}")
