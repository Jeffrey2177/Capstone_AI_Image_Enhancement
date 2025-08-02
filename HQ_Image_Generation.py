# synth_many_inpaints_hq_dwi.py

from medigan import Generators
from pathlib import Path
import numpy as np
import shutil
import cv2


# ─── CONFIG ───────────────────────────────────────────────────────────────────────
TARGET_MODEL_ID         = "00007_INPAINT_BRAIN_MRI"
NUM_SAMPLES             = 1
NUM_INPAINTS_PER_SAMPLE = 14    # bump up from default 2 to get ~200+ inpaints
UPSAMPLE_SIZE           = (512, 512)

# 15 B-values from 50 to 1200 s/mm²
BVALS = list(np.linspace(50, 1200, 15, dtype=int))

# ─── STEP 1: GENERATE & UPSAMPLE MULTIPLE INPAINT T2 SLICES ───────────────────────
def generate_and_upsample_t2(model_id, out_dir, up_size):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    # generate with many inpaints
    Generators().generate(
        model_id=model_id,
        num_samples=NUM_SAMPLES,
        num_inpaints_per_sample=NUM_INPAINTS_PER_SAMPLE,
        output_path=str(out),
        save_images=True,
        install_dependencies=True
    )

    # filter & upsample *_T2_sample.png
    t2_list = []
    for p in sorted(out.iterdir()):
        if p.suffix == ".png" and p.stem.lower().endswith("_t2_sample"):
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            big = cv2.resize(img, up_size, interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(str(p), big)
            t2_list.append(p)
    return t2_list  # should be length = NUM_INPAINTS_PER_SAMPLE

# ─── STEP 2: SIMULATE 15 HQ DWI MAPS ──────────────────────────────────────────────
def simulate_hq_dwi(png_path, dwi_dir):
    dwi_dir = Path(dwi_dir); dwi_dir.mkdir(parents=True, exist_ok=True)
    import cv2, numpy as np
    img = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    h, w = img.shape

    for b in BVALS:
        decay    = np.exp(-b / 1200)
        dw       = img * decay
        sigma    = 2 + (b / max(BVALS)) * 8
        noise    = np.random.randn(h, w) * sigma
        dw_noisy = np.clip(dw + noise, 0, 255).astype(np.uint8)

        out_name = dwi_dir / f"{png_path.stem}_b{b}.png"
        cv2.imwrite(str(out_name), dw_noisy)

# ─── MAIN ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1) Generate & upsample your many T2 inpaints
    t2_slices = generate_and_upsample_t2(
        TARGET_MODEL_ID, "data/HQ_T2", UPSAMPLE_SIZE
    )

    # 2) Simulate 15 HQ DWI images from each inpaint
    for t2 in t2_slices:
        simulate_hq_dwi(t2, "data/HQ_DWI")


    shutil.rmtree("data/HQ_T2", ignore_errors=True)

    total = len(t2_slices) * len(BVALS)
    print("Done")
    print(f" Inpaints generated: {len(t2_slices)} → data/HQ_T2/")
    print(f" HQ DWI maps: {total} → data/HQ_DWI/")
    print(f" B-values: {BVALS}")

