from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision import transforms
import numpy as np

lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='alex')


def evaluate_metrics(img1_pil, img2_pil):
    # Convert PIL → grayscale numpy array
    img1 = np.array(img1_pil.convert("L"))  # predicted
    img2 = np.array(img2_pil.convert("L"))  # ground truth

    ssim_score = ssim(img1, img2, data_range=img2.max() - img2.min())
    psnr_score = psnr(img1, img2, data_range=img2.max() - img2.min())
    mse_score = mse(img1, img2)

    return ssim_score, psnr_score, mse_score


def evaluate_lpips(img1_pil, img2_pil):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),   # LPIPS prefers 256
        transforms.ToTensor()
    ])
    img1 = transform(img1_pil).unsqueeze(0)
    img2 = transform(img2_pil).unsqueeze(0)
    return lpips_metric(img1, img2).item()