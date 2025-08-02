import cv2, numpy as np, random
from pathlib import Path

# ─── CONFIG ───────────────────────────────────────────────────────────────────────
HQ_DIR = Path("data/HQ_DWI")
LQ_ROOT = Path("data/LQ_DWI")
NUM_LQ = 10

# ─── EFFECTS ──────────────────────────────────────────────────────────────────────
def down_up_sample(img):
    # downsample to 0.5–0.8× then back up
    h,w = img.shape
    scale = random.uniform(0.5, 0.8)
    small = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w,h), interpolation=cv2.INTER_LINEAR)

def gaussian_blur(img):
    k = random.choice([3,5,7])
    return cv2.GaussianBlur(img, (k,k), 0)

def noise_injection(img):
    h,w = img.shape
    sigma = random.uniform(5, 20)
    noise = np.random.randn(h, w) * sigma
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)

def motion_blur(img):
    # create motion blur kernel
    k = random.choice([9, 15, 21])  # kernel length
    angle = random.uniform(0, 360)
    # make an empty kernel
    kern = np.zeros((k, k), dtype=np.float32)
    kern[k//2, :] = np.ones(k, dtype=np.float32)
    # rotate kernel
    M = cv2.getRotationMatrix2D((k//2, k//2), angle, 1.0)
    kern = cv2.warpAffine(kern, M, (k, k))
    kern /= kern.sum() or 1.0
    return cv2.filter2D(img, -1, kern)

EFFECTS = [down_up_sample, gaussian_blur, noise_injection, motion_blur]

# ─── WORKFLOW ────────────────────────────────────────────────────────────────────
def apply_random_effects(img):
    funcs = random.sample(EFFECTS, k=random.randint(2,3))
    out = img.copy()
    for f in funcs:
        out = f(out)
    return out

def generate_lq_variants():
    for hq_path in HQ_DIR.glob("*.png"):
        stem = hq_path.stem
        out_dir = LQ_ROOT / stem
        out_dir.mkdir(parents=True, exist_ok=True)

        img = cv2.imread(str(hq_path), cv2.IMREAD_GRAYSCALE)
        for i in range(NUM_LQ):
            lq = apply_random_effects(img)
            fname = out_dir / f"{stem}_LQ_{i:02d}.png"
            cv2.imwrite(str(fname), lq)
        print(f"Generated {NUM_LQ} LQ variants for {stem}")

if __name__ == "__main__":
    LQ_ROOT.mkdir(parents=True, exist_ok=True)
    generate_lq_variants()
    print("Complete")

