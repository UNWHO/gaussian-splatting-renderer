import cv2
from skimage import io, img_as_float
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
from PIL import Image
import numpy as np
import torch

# Load images
img1_path = './gt.png'
img2_path = './orig.png'
img3_path = './result.png'

img1 = img_as_float(io.imread(img1_path))
img2 = img_as_float(io.imread(img2_path))
img3 = img_as_float(io.imread(img3_path))

print(img1.shape)

# Calculate SSIM
ssim_index, _ = ssim(img1, img2, full=True, multichannel=True, channel_axis=2, data_range=1.0)
print(f"SSIM orig: {ssim_index}")
ssim_index, _ = ssim(img1, img3, full=True, multichannel=True, channel_axis=2, data_range=1.0)
print(f"SSIM result: {ssim_index}")
ssim_index, _ = ssim(img2, img3, full=True, multichannel=True, channel_axis=2, data_range=1.0)
print(f"SSIM : {ssim_index}")

# Calculate PSNR
psnr_value = psnr(img1, img2, data_range=img1.max() - img1.min())
print(f"PSNR orig: {psnr_value}")
psnr_value = psnr(img1, img3, data_range=img1.max() - img1.min())
print(f"PSNR result: {psnr_value}")
psnr_value = psnr(img2, img3, data_range=img1.max() - img1.min())
print(f"PSNR: {psnr_value}")


# # Calculate LPIPS
# # Convert images to tensors
# img1_tensor = torch.tensor(np.array(Image.open(img1_path)).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
# img2_tensor = torch.tensor(np.array(Image.open(img2_path)).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)

# # Initialize LPIPS model
# loss_fn = lpips.LPIPS(net='alex')

# # Calculate LPIPS score
# lpips_score = loss_fn(img1_tensor, img2_tensor)
# print(f"LPIPS: {lpips_score.item()}")