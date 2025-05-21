import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pydicom
import torch
from monai.transforms import GaussianSmooth
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.util import img_as_float
from io import BytesIO

# ------------------------- UTILS -------------------------

def add_noise_to_dicom(img, noise_type='gaussian', noise_level=0.3):
    noisy_img = img.copy()
    
    if noise_type == 'gaussian':
        noise = np.random.normal(0, noise_level, img.shape)
        noisy_img = img + noise
        
    elif noise_type == 'salt_pepper':
        salt_mask = np.random.random(img.shape) < (noise_level / 2)
        pepper_mask = np.random.random(img.shape) < (noise_level / 2)
        noisy_img[salt_mask] = img.max()
        noisy_img[pepper_mask] = img.min()
        
    elif noise_type == 'speckle':
        noise = np.random.normal(1, noise_level, img.shape)
        noisy_img = img * noise

    noisy_img = np.clip(noisy_img, img.min(), img.max())
    return noisy_img

def denoise_dicom_with_monai(img, sigma=5.0):
    img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
    gaussian_smoother = GaussianSmooth(sigma=sigma)
    denoised_tensor = gaussian_smoother(img_tensor)
    return denoised_tensor.squeeze().cpu().numpy()

def denoise_dicom_with_skimage(img):
    img = img_as_float(img)
    sigma_est = np.mean(estimate_sigma(img, channel_axis=None))
    denoised = denoise_nl_means(
        img,
        h=1.5 * sigma_est,
        fast_mode=False,
        patch_size=7,
        patch_distance=15,
        channel_axis=None
    )
    return denoised

def calculate_metrics(original, denoised):
    original_float = img_as_float(original)
    denoised_float = img_as_float(denoised)
    psnr = peak_signal_noise_ratio(original_float, denoised_float, data_range=original_float.max() - original_float.min())
    ssim = structural_similarity(original_float, denoised_float, data_range=original_float.max() - original_float.min())
    return {'PSNR': psnr, 'SSIM': ssim}

def show_image_comparison(images, titles):
    fig, axes = plt.subplots(1, len(images), figsize=(5 * len(images), 5))
    if len(images) == 1:
        axes = [axes]
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    st.pyplot(fig)

# ------------------------- STREAMLIT UI -------------------------

st.set_page_config(page_title="DICOM Denoising App", layout="wide")
st.title("🩻 DICOM Denoising App (MONAI & Skimage)")

uploaded_file = st.file_uploader("📁 Upload a DICOM file (.dcm)", type=['dcm'])

if uploaded_file:
    data = pydicom.dcmread(BytesIO(uploaded_file.read()))
    img = data.pixel_array.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min())

    st.subheader("Original Image")
    st.image(img, caption="Original DICOM", use_column_width=True, clamp=True)

    # Noise parameters
    st.sidebar.title("🔧 Parameters")
    noise_type = st.sidebar.selectbox("Noise Type", ['gaussian', 'salt_pepper', 'speckle'])
    noise_level = st.sidebar.slider("Noise Level", 0.0, 1.0, 0.3, 0.05)
    sigma = st.sidebar.slider("MONAI Gaussian Sigma", 0.0, 10.0, 5.0, 0.5)

    # Processing
    if st.button("🔍 Run Denoising"):
        noisy_img = add_noise_to_dicom(img, noise_type=noise_type, noise_level=noise_level)
        denoised_monai = denoise_dicom_with_monai(noisy_img, sigma=sigma)
        denoised_skimage = denoise_dicom_with_skimage(noisy_img)

        # Metrics
        metrics = {
            "MONAI": calculate_metrics(img, denoised_monai),
            "Skimage": calculate_metrics(img, denoised_skimage),
            "Noisy": calculate_metrics(img, noisy_img)
        }

        # Show results
        st.subheader("📸 Denoising Results")
        show_image_comparison(
            [img, noisy_img, denoised_monai, denoised_skimage],
            ["Original", "Noisy", "MONAI Denoised", "Skimage Denoised"]
        )

        st.subheader("📊 Evaluation Metrics")
        st.markdown(f"""
        **Noisy Image**  
        - PSNR: `{metrics['Noisy']['PSNR']:.2f}`  
        - SSIM: `{metrics['Noisy']['SSIM']:.4f}`

        **MONAI Denoised**  
        - PSNR: `{metrics['MONAI']['PSNR']:.2f}`  
        - SSIM: `{metrics['MONAI']['SSIM']:.4f}`

        **Skimage Denoised**  
        - PSNR: `{metrics['Skimage']['PSNR']:.2f}`  
        - SSIM: `{metrics['Skimage']['SSIM']:.4f}`
        """)

else:
    st.info("👆 Upload a DICOM file to begin.")
