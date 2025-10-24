import argparse
import glob
import math

import numpy as np
import torch
from admm_utils import admm_pnp
from model.dncnn import DnCNN
from PIL import Image
from torch import nn
from torch.fft import fft2, fftshift, ifft2, ifftshift

from implementation.M3.M3_train import DnCNN

# %% 4. M3 validation
"""
You will validate your implementation on the images provided in the testing set.

Perform in a loop:
i)      read a new image
ii)     build a Fourier mask using a random seed to ensure the mask is
        different for each image
iii)    create a measurement operator and its adjoint (Phit and Phi)
iv)     compute the observed Fourier coefficients
v)      calculate value of a sigma for input SNR = 30dB
vi)     add noise to the Fourier coefficients to generate the observed data
vii)    generate a backprojected image and record its SNR and SSIM
iii)    fix the appropriate parameters for the PnP-ADMM algorithm
ix)     perform reconstruction by calling your PnP-ADMM implementation, 
        recording reconstruction time for each reconstruction
x)      calculate the SNR and SSIM and save the reconstructed image
"""


def create_mask(N1, N2):
    ft = 4  # Subsampling rate
    p = 0.08  # Width (in percent) of the central band
    N = N1 * N2
    num_meas = np.floor(N1 / ft)
    M = num_meas * N2  # Total number of measurements
    w = np.floor(N1 * p / 2)
    num_meas = num_meas - w

    # Building the Fourier mask
    mask = np.zeros((N1, N2))
    lines_int = np.random.randint(0, N1, int(num_meas))  # Sampling uniformly at random
    mask[int(np.floor(N1 / 2 - w) - 1) : int(np.floor(N2 / 2 + w) - 1), :] = 1
    mask[lines_int, :] = 1
    mask[0, :] = 0
    mask[N1 - 1, :] = 0
    mask = mask.T
    mask = torch.tensor(mask, dtype=torch.float32)
    return mask


def proj_l2(x, epsilon):
    norm_x = torch.norm(x)
    factor = torch.min(epsilon / norm_x, torch.ones_like(norm_x))
    return x * factor


def Phit(x, mask):
    return ifftshift(mask * fftshift(fft2(x))).ravel() / math.sqrt(x.numel())


def Phi(x, mask):
    return (ifft2(ifftshift(mask * fftshift(x.reshape(mask.shape)))) * math.sqrt(x.numel())).real


def gen_noise(x, mask, isnr):
    N = x.numel()
    sigma = torch.norm(x.flatten()) / math.sqrt(N) * 10 ** (-isnr / 20)
    noise = sigma / math.sqrt(2) * (torch.randn_like(x) + 1j * torch.randn_like(x))
    noise = ifftshift(mask * fftshift(noise)).ravel()

    return noise, sigma


def snr_single(output, target):
    # compute signal-to-noise ratio (SNR) for a single image
    return 20 * torch.log10(torch.norm(target.ravel()) / torch.norm(target.ravel() - output.ravel()))


def admm_pnp(
    y,
    epsilon,
    Phit,
    Phi,
    mask,
    denoiser,
    verbose=1,
    rel_tol=1e-4,
    rel_tol2=1e-4,
    max_iter=200,
    rho=1,
    delta=1,
    device=torch.device("cpu"),
):  # Added device parameter

    # Move input tensors to the specified device
    y = y.to(device)

    # TODO: implement the ADMM algorithm for PnP

    # return xsol, t


def parse_args():
    parser = argparse.ArgumentParser(description="Run the ADMM algorithm for M3.")
    parser.add_argument(
        "--img_path", type=str, default="data/testingset/", help="Path to the testing images."
    )
    parser.add_argument("--denoiser_ckpt_path", type=str, help="Path to the trained denoiser for M3.")
    parser.add_argument("--n_ch_in", type=int, default=1, help="Number of input channels")
    parser.add_argument("--n_ch_out", type=int, default=1, help="Number of output channels")
    parser.add_argument("--n_ch", type=int, default=32, help="Number of channels")
    parser.add_argument("--depth", type=int, default=17, help="Depth of network")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_paths = glob.glob(args.img_path + "*.tiff")
    # load the trained denoiser
    # denoiser =

    snrs = np.zeros(len(img_paths))

    for i, img_file in enumerate(img_paths):
        print(f"Processing image: {img_file.split('/')[-1]}")
        isnr = 30

        img = np.array(Image.open(img_file)).astype(np.float32)
        img = torch.tensor(img, dtype=torch.float32, device=device)

        mask = create_mask(*img.shape).to(device)

        # Compute the measurements with forward measurement operator Phit
        y0 = Phit(img, mask)

        # Compute noise given the image, mask, and iSNR
        noise, sigma = gen_noise(img, mask, isnr)

        # Add the noise to the measurements
        y = y0 + noise
        M = y.numel()

        # Compute the back-projected image with the adjoint measurement operator Phi
        y_bp = Phi(y, mask)

        # set the parameters for the ADMM algorithm
        lips = 1
        rel_tol = 1e-4
        rel_tol2 = 1e-4
        max_iter = 1000
        epsilon2 = sigma * math.sqrt(M + 2 * math.sqrt(M))
        rho = 1
        delta = 1 / lips

        print(f"epsilon2 = {epsilon2}, rho = {rho}")

        xsol1, niter1 = admm_pnp(
            y,
            epsilon2,
            Phit,
            Phi,
            mask,
            denoiser,
            verbose=1,
            rel_tol=rel_tol,
            rel_tol2=rel_tol2,
            max_iter=max_iter,
            rho=rho,
            delta=delta,
            device=device,
        )

        snrs[i] = snr_single(img, xsol1)
        print(f"SNR = {snrs[i]:.4f}")
    print("-" * 80)
    print(f"M3: Averaged SNR = {snrs.mean().item():.4f}")
