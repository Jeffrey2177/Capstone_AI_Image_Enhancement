import streamlit as st
import numpy as np
import os
from PIL import Image
from io import BytesIO
from inference_utils import run_inference
from metrics_utils import evaluate_metrics, evaluate_lpips
import zipfile


# ─── CONFIG ───────────────────────────────────────────────────────
st.set_page_config(page_title="DWI GAN Enhancer", layout="wide")
st.markdown("<h1 style='text-align: center; color: #5a97cf;'>Diffusion-MRI Brain Image Enhancer</h1>", unsafe_allow_html=True)

# ─── ABOUT SECTION ────────────────────────────────────────────────
with st.expander("📘 About This App", expanded=False):
    st.markdown("""
    This app enhances low-Quality Diffusion-MRI brain images using a trained Pix2Pix GAN model.  
    Upload one or multiple degraded low-quality images and one high-quality baseline image.  
    The app will:
    - Generate high-quality predicted images
    - Compare them visually
    - Compute and average the metrics: SSIM, PSNR, MSE, LPIPS
    """)

# ─── SIDEBAR: INPUT ───────────────────────────────────────────────
st.sidebar.header("Upload Images")
lq_files = st.sidebar.file_uploader(
    "📂 Upload One or More Low-Quality Images",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
    help="Upload multiple degraded brain scan images. Each will be enhanced and compared against the same high-quality image."
)

hq_file = st.sidebar.file_uploader(
    "📂 Upload Ground Truth High-Quality Image",
    type=["png", "jpg", "jpeg"],
    help="Upload a single high-quality ground truth scan for comparison."
)

# ─── MODEL PATH (RELATIVE) ────────────────────────────────────────
MODEL_PATH = os.path.join("pytorch-CycleGAN-and-pix2pix", "checkpoints", "pix2pix_main", "latest_net_G.pth")

# ─── MAIN LOGIC ────────────────────────────────────────────────────
if lq_files and hq_file:
    # Load HQ once
    hq_img = Image.open(hq_file).convert("RGB").resize((512, 512))

    # Storage for metric averaging
    ssim_scores, psnr_scores, mse_scores, lpips_scores = [], [], [], []

    st.subheader("Image Comparisons")

    for idx, file in enumerate(lq_files):
        lq_img = Image.open(file).convert("RGB").resize((512, 512))
        pred_img = run_inference(lq_img, model_path=MODEL_PATH, device="cpu")

        # Compute metrics
        ssim, psnr, mse = evaluate_metrics(pred_img, hq_img)
        lpips = evaluate_lpips(pred_img, hq_img)
        ssim_scores.append(ssim)
        psnr_scores.append(psnr)
        mse_scores.append(mse)
        lpips_scores.append(lpips)

        # Display images in a row
        st.markdown(f"**Sample {idx+1}**")
        col1, col2, col3 = st.columns([1, 1, 1])  # Wider middle column
        with col1:
            st.image(lq_img, caption="LQ Input", use_column_width=True)
        with col2:
            st.image(pred_img, caption="Predicted HQ", use_column_width=True)
        with col3:
            st.image(hq_img, caption="Ground Truth HQ", use_column_width=True)

    # ─── METRICS SUMMARY ───────────────────────────────────────────
    avg_ssim = np.mean(ssim_scores)
    avg_psnr = np.mean(psnr_scores)
    avg_mse = np.mean(mse_scores)
    avg_lpips = np.mean(lpips_scores)

    st.subheader("Average Evaluation Metrics Across All Samples")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("SSIM ", f"{avg_ssim:.4f}")
    col2.metric("PSNR (dB) ", f"{avg_psnr:.2f}")
    col3.metric("MSE ", f"{avg_mse:.4f}")
    col4.metric("LPIPS ", f"{avg_lpips:.4f}")

    metric_option = st.selectbox(
        "Show Interpretation Table For:",
        ["SSIM", "PSNR", "MSE", "LPIPS"]
    )

    if metric_option == "SSIM":
        st.table({
            "Range": ["0.0 – 0.5", "0.51 – 0.7", "0.71 – 0.9", "0.91 – 1.0"],
            "Interpretation": ["Poor", "Fair", "Good", "Excellent"]
        })
    elif metric_option == "PSNR":
        st.table({
            "Range (dB)": ["<20", "20 – 30", "30 – 40", ">40"],
            "Interpretation": ["Unacceptable", "Acceptable", "Good", "Excellent"]
        })
    elif metric_option == "MSE":
        st.table({
            "Range": [">100", "10 – 100", "1 – 10", "<1"],
            "Interpretation": ["Very High Error", "High", "Moderate", "Low"]
        })
    elif metric_option == "LPIPS":
        st.table({
            "Range": [">0.3", "0.2 – 0.3", "0.1 – 0.2", "<0.1"],
            "Interpretation": ["Poor", "Fair", "Good", "Excellent"]
        })


    # ─── BUILD ZIP FILE OF ALL GENERATED IMAGES ──────────────────────
    zip_buf = BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zipf:
        for i, file in enumerate(lq_files):
            lq_img = Image.open(file).convert("RGB").resize((512, 512))
            pred_img = run_inference(lq_img, model_path=MODEL_PATH, device="cpu")

            img_buf = BytesIO()
            pred_img.save(img_buf, format="PNG")
            img_buf.seek(0)

            # Name the output after the original LQ filename
            original_name = os.path.splitext(file.name)[0]
            zipf.writestr(f"predicted_{original_name}.png", img_buf.read())

    zip_buf.seek(0)

    # ─── SIDEBAR ZIP DOWNLOAD BUTTON ─────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.download_button(
        label="⬇️ Download All Predicted Images (ZIP)",
        data=zip_buf,
        file_name="predicted_images.zip",
        mime="application/zip",
        key="zip_download_button"
    )

else:
    st.warning("Please upload at least one Low-quality image and one High-Quality image to begin.")

# ─── FOOTER ───────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; font-size: 0.9em; color: gray;'>
        © 2025 Jeffrey Mak · 
        <a href="https://github.com/Jeffrey2177/Capstone_AI_Image_Enhancement" target="_blank" style="text-decoration: none; color: #4CAF50;">
            GitHub Repo
        </a><br>
        Built using Streamlit 
    </div>
    """,
    unsafe_allow_html=True
)