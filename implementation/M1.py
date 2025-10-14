"""
M1.py file
Please read carefully the comments below. Sections 1 and 2 contain useful
functions to generate the measurement and sparsity operators. Section 3
and 4 give guidelines for the implementation and validation of M1.
"""

# %% load libraries
from typing import Callable

import numpy as np
import torch
from torch.fft import fft2, fftshift, ifft2, ifftshift
import ptwt
from PIL import Image
import matplotlib.pyplot as plt
import tqdm
import os
import skimage.metrics as metrics

# %% 1. Generate measurement operators

# From a given groundtruth you will create the acquired Fourier data.
# To do so, you will create a Fourier mask that will be of use in the
# measurement operator. The acquired Fourier data consist in the masked
# Fourier transform of the groundtruth, to which noise is added.


# Function for creating the Fourier mask
def create_mask(N1: int, N2: int) -> torch.Tensor:
    """
    Create a Fourier mask for the measurement operators

    :param N1: height of the image
    :type N1: int
    :param N2: width of the image
    :type N2: int
    :return: Fourier mask
    :rtype: torch.Tensor
    """
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


# Create the measurement operators
def create_meas_op(mask: torch.Tensor) -> tuple[Callable, Callable]:
    """
    Create the measurement operators

    :param mask: Fourier mask
    :type mask: torch.Tensor
    :return: measurement operators
    :rtype: tuple[Callable, Callable]
    """

    # Adjoint measurement operator
    def Phit(x: torch.Tensor) -> torch.Tensor:
        """
        Compute the adjoint measurement operator

        :param x: input signal
        :type x: torch.Tensor
        :param mask: Fourier mask
        :type mask: torch.Tensor
        :return: measurements in Fourier domain
        :rtype: torch.Tensor
        """
        return ifftshift(mask * fftshift(fft2(x))).ravel() / np.sqrt(x.numel())

    # Forward measurement operator
    def Phi(x: torch.Tensor) -> torch.Tensor:
        """
        Compute the forward measurement operator

        :param x: input signal
        :type x: torch.Tensor
        :param mask: Fourier mask
        :type mask: torch.Tensor
        :return: back-projected measurements in image domain
        :rtype: torch.Tensor
        """

        return (
            ifft2(ifftshift(mask * fftshift(x.reshape(mask.shape))))
            * np.sqrt(x.numel())
        ).real

    return Phit, Phi


# generate noise vector in Fourier domain
def gen_noise(
    x: torch.Tensor, mask: torch.Tensor, isnr: float = 30
) -> tuple[torch.Tensor, float]:
    """
    Generate noise for a given input signal at a given SNR level

    :param x: input signal
    :type x: torch.Tensor
    :param mask: Fourier mask
    :type mask: torch.Tensor
    :param isnr: input SNR level
    :type isnr: float
    :return: noisy signal and noise level
    :rtype: tuple[torch.Tensor, float]
    """

    N = x.numel()
    sigma = torch.norm(x.flatten()) / np.sqrt(N) * 10 ** (-isnr / 20)
    noise = sigma / np.sqrt(2) * (torch.randn_like(x) + 1j * torch.randn_like(x))
    noise = ifftshift(mask * fftshift(noise)).ravel()

    return noise, sigma


# %% 2. Create sparsity operators
# Here you create a sparsity operator and its adjoint using the well known
# orthogonal Daubechies wavelet transforms.


# Create the sparsity operators
def gen_sparsity_op(
    wavelet: ptwt.Wavelet = "db4",
    mode: ptwt.constants.BoundaryMode = "periodic",
    level: int = 4,
) -> tuple[Callable, Callable]:
    """
    Create the sparsity operators

    :param wavelet: wavelet type, defaults to "db4"
    :type wavelet: ptwt.Wavelet, optional
    :param mode: boundary mode, defaults to "periodic"
    :type mode: ptwt.constants.BoundaryMode, optional
    :param level: decomposition level, defaults to 4
    :type level: int, optional
    :return: sparsity operators
    :rtype: tuple[Callable, Callable]
    """

    def Psit(x: torch.Tensor) -> ptwt.constants.WaveletCoeff2d:
        """
        Compute the adjoint sparsity operator

        :param x: input signal
        :type x: torch.Tensor
        :return: coefficients of the wavelet transform
        :rtype: ptwt.constants.WaveletCoeff2d
        """

        return ptwt.wavedec2(x, wavelet=wavelet, mode=mode, level=level)

    def Psi(alpha: ptwt.constants.WaveletCoeff2d) -> torch.Tensor:
        """
        Compute the forward sparsity operator

        :param alpha: coefficients of the wavelet transform
        :type alpha: ptwt.constants.WaveletCoeff2d
        :return: reconstructed image
        :rtype: torch.Tensor
        """

        return ptwt.waverec2(alpha, wavelet=wavelet)

    return Psit, Psi


# %% 3. M1 implementation
# You will here implement the ADMM algorithm for solving the constrained
# optimization problem using the appropriate functions provided above. It
# is highly recommended that you implement your algorithm in a separate
# file as a function with possibilities to pass the required inputs and
# parameters.

# For imposing the constraint on the data-fidelity term, you will use
# following value of the epsilon parameter.


def shrink_torch(x: torch.Tensor, thre: float) -> torch.Tensor:
    """
    Apply the shrinkage (soft-thresholding) operator over the input signal

    :param x: input signal
    :type x: torch.Tensor
    :param thre: threshold
    :type thre: float
    :return: output signal
    :rtype: torch.Tensor
    """
    return torch.sign(x) * torch.maximum(torch.abs(x) - thre, torch.zeros_like(x))


def shrinkage_ptwt(
    alpha: ptwt.constants.WaveletCoeff2d, thre: float
) -> ptwt.constants.WaveletCoeff2d:
    """
    Apply the shrinkage (soft-thresholding) operator over the
    wavelet coefficients

    :param alpha: input wavelet coefficients
    :type alpha: ptwt.constants.WaveletCoeff2d
    :param t: threshold
    :type t: float
    :return: output signal
    :rtype: ptwt.constants.WaveletCoeff2d
    """

    return tuple(
        [
            shrink_torch(alpha[0], thre),
            *[tuple(shrink_torch(sub, thre) for sub in lev) for lev in alpha[1:]],
        ]
    )


def proj_l2(x: torch.Tensor, epsilon: float) -> torch.Tensor:
    """
    Projection onto the L2 ball

    :param x: input signal
    :type x: torch.Tensor
    :param epsilon: radius of the L2 ball
    :type epsilon: float
    :return: output signal
    :rtype: torch.Tensor
    """
    return x * torch.min(epsilon / torch.linalg.vector_norm(x), torch.ones_like(torch.linalg.vector_norm(x)))

def rsnr(x: np.ndarray, x0: np.ndarray) -> float:
    """
    Compute the reconstruction signal-to-noise ratio (RSNR) in dB

    :param x: reconstruction
    :type x: np.ndarray
    :param x0: ground truth
    :type x0: np.ndarray
    :return: RSNR
    :rtype: float
    """
    return 20 * np.log10(np.linalg.norm(x0) / np.linalg.norm(x0 - x))

def admm_conbpdn(
    y: torch.Tensor,
    epsilon: float,
    Phit: Callable,
    Phi: Callable,
    Psit: Callable,
    Psi: Callable,
    verbose: int = 1,
    rel_tol: float = 1e-4,
    rel_tol2: float = 1e-4,
    max_iter: int = 200,
    rho: float = 1e2,
    delta: float = 1.0,
) -> tuple[torch.Tensor, float, int]:
    """
    Function to solve the Constrained Basis Pursuit DeNoising problem using the
    Alternating Direction Method of Multipliers (ADMM) algorithm with l-1 norm
    regularisation, as shown in equation (1) in the lab 2 guide.

    :param y: measurement vector
    :type y: torch.Tensor
    :param epsilon: residual error bound
    :type epsilon: float
    :param Phit: Forward measurement operator
    :type Phit: function
    :param Phi: Adjoint measurement operator
    :type Phi: function
    :param Psit: Forward sparsity operator
    :type Psit: function
    :param Psi: Adjoint sparsity operator
    :type Psi: function
    :param verbose: verbose, defaults to 1
    :type verbose: int, optional
    :param rel_tol: minimum relative change of the objective function value, defaults to 1e-4
    :type rel_tol: float, optional
    :param rel_tol2: minimum relative change of the iterates, defaults to 1e-4
    :type rel_tol2: float, optional
    :param max_iter: maximum number of iterations to run the algorithm, defaults to 200
    :type max_iter: int, optional
    :param rho: penalty parameter for constraint violation in the augmented Lagrangian problem,
                defaults to 1e2
    :type rho: float, optional
    :param delta: step size for the proximal gradient, optional, defaults to 1.0
    :type delta: float, optional
    :return: tuple(solution of the problem, final value of the objective function,
            final iteration number)
    :rtype: tuple[torch.Tensor, float, int]
    """

    # initialisation
    v = torch.zeros(np.shape(y))  # the dual variable
    s = -y  # the intermediate variable
    n = proj_l2(s, epsilon)
    xsol = torch.zeros_like(Phi(s))
    fval = 0
    flag = 0
    t = 0

    # main loop
    t = 0
    while flag == 0 and t < max_iter:
        # compute x_t+1
        xsol_old = xsol
        xsol = Psi(shrinkage_ptwt(Psit(xsol - delta * torch.real(Phi(s + n - v))), delta / rho))
        s = Phit(xsol) - y
        n = proj_l2(v - s, epsilon)
        v = v - (s + n)
        fval_old = fval
        fval = torch.linalg.norm(Psit(xsol)[0], 1).item()
        rel_fval = abs(fval - fval_old) / abs(fval)
        n_t = np.linalg.norm(y - Phit(xsol), 2)
        rel_x = np.linalg.norm(xsol - xsol_old, 2) / np.linalg.norm(xsol, 2)
        t += 1
        if rel_fval < rel_tol and n_t < epsilon or rel_x < rel_tol2:
            flag = 1
        if verbose > 1:
            print(
                f"Iteration {t}: fval = {fval}, rel_fval = {rel_fval}, rel_x = {rel_x}"
            )
        if verbose == 1:
            tqdm.tqdm.write(
                f"Iteration {t}: fval = {fval}, rel_fval = {rel_fval}, rel_x = {rel_x}"
            )

    return xsol, fval, t

def run_admm(img_file, rho=1e2):
    img = np.array(Image.open(img_file)).astype(np.float32)
    img = torch.tensor(img, dtype=torch.float32, device=device)

    # Create a Fourier mask
    mask = create_mask(*img.shape).to(device)
    Phit, Phi = create_meas_op(mask)

    # Compute the measurements with forward measurement operator Phit
    y0 = Phit(img)

    # Compute noise given the image, mask, and iSNR
    noise, sigma = gen_noise(img, mask)

    # Add the noise to the measurements
    y = y0 + noise
    M = y.numel()

    # Generate the sparsity operator
    Psit, Psi = gen_sparsity_op()

    # For imposing the constraint on the data-fidelity term, you will use
    # following value of the epsilon parameter.
    epsilon = sigma * np.sqrt(M + 2 * np.sqrt(M))

    # Reconstruct the image by calling our ADMM implementation
    xsol, fval, it = admm_conbpdn(
        y,
        epsilon,
        Phit,
        Phi,
        Psit,
        Psi,
        verbose=0,
        rel_tol=1e-4,
        rel_tol2=1e-4,
        max_iter=200,
        rho=rho,
        delta=1.0,
    )

    # Calculate SNR and SSIM of the reconstructed image
    snr = rsnr(xsol.cpu().numpy(), img.cpu().numpy())
    ssim = metrics.structural_similarity(img.cpu().numpy(), xsol.cpu().numpy(), data_range=(img.max() - img.min()).item())

    return xsol, snr, ssim

# %% 4. M1 validation

# You will validate your implementation on the images provided in the
# testing set.

# 1. Load the files in the folder containing the testing set
#    Hint: you can use functions: 'glob' and 'os.path.join'

# 2. Perform in a loop:
#    i) read a new image
#   ii) build a Fourier mask using a random seed to ensure the mask is
#   different for each image
#  iii) create a measurement operator and its adjoint (Phit and Phi)
#   iv) compute the observed Fourier coefficients
#    v) calculate value of a sigma for input SNR = 30dB
#   vi) add noise to the Fourier coefficients to generate the observed data
#  vii) generate a backprojected image and record its SNR and SSIM
# viii) define a sparsity operator and its adjoint (Psit and Psi)
#   ix) fix the appropriate parameters for the ADMM algorithm, including
#   the value of epsilon
#    x) perform reconstruction by calling your ADMM implementation
#   xi) calculate the SNR and SSIM and save the reconstructed image

if __name__ == "__main__":
    # Select to run on CUDA devcie if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the testing set images
    testing_set_dir = "../testing_set"
    if not os.path.isdir(testing_set_dir):
        raise FileNotFoundError(f"Testing set directory not found: {testing_set_dir}")
    testing_files = [os.path.join(testing_set_dir, f) for f in os.listdir(testing_set_dir)]

    # Initialize lists to store overall SNR and SSIM results
    overall_snr_list = []
    overall_ssim_list = []

    # Define rho values to test
    #rho_values = [1, 5, 10, 20, 50]
    rho_values = list(range(1, 50))

    # Loop through multiple rho values to find the best one
    for rho in rho_values:
        print(f"\nRunning ADMM with rho = {rho}")

        # Initialize lists to store per run results
        snr_list = []
        ssim_list = []

        # Loop through all images in the testing set with a progress bar
        for img_file in tqdm.tqdm(testing_files, desc=f"Processing (rho={rho})", unit="img"):
            xsol, snr, ssim = run_admm(img_file, rho=rho)

            # Append results to the lists
            snr_list.append(snr)
            ssim_list.append(ssim)

        # Store average SNR and SSIM
        overall_snr_list.append(np.mean(snr_list))
        overall_ssim_list.append(np.mean(ssim_list))

    # Plot Average SNR
    plt.figure()
    plt.plot(rho_values, overall_snr_list, marker="o", linestyle="-")
    plt.xlabel("Rho")
    plt.ylabel("Average SNR (dB)")
    plt.title("Average SNR vs Rho")
    plt.grid(True)
    plt.savefig("Average_SNR_vs_Rho.pdf")
    plt.show()

    # Plot Average SSIM
    plt.figure()
    plt.plot(rho_values, overall_ssim_list, marker="s", linestyle="-")
    plt.xlabel("Rho")
    plt.ylabel("Average SSIM")
    plt.title("Average SSIM vs Rho")
    plt.grid(True)
    plt.savefig("Average_SSIM_vs_Rho.pdf")
    plt.show()

    # img = np.array(Image.open(img_file)).astype(np.float32)
    # img = torch.tensor(img, dtype=torch.float32, device=device)

    # # Create a Fourier mask
    # mask = create_mask(*img.shape).to(device)
    # Phit, Phi = create_meas_op(mask)

    # # Compute the measurements with forward measurement operator Phit
    # y0 = Phit(img)

    # # Compute noise given the image, mask, and iSNR
    # noise, sigma = gen_noise(img, mask)

    # # Add the noise to the measurements
    # y = y0 + noise
    # M = y.numel()

    # # In order to visualise the measurements in the image domain, you can apply
    # # the adjoint of the measurement operator then take the real part. The
    # # resulting image is called the backprojected image.
    # # y_bp = Phi(y)
    # # plt.figure()
    # # plt.imshow(y_bp.cpu().numpy(), cmap="gray")
    # # plt.title("Backprojected image")
    # # plt.show()

    # # Generate the sparsity operator
    # Psit, Psi = gen_sparsity_op()

    # # For imposing the constraint on the data-fidelity term, you will use
    # # following value of the epsilon parameter.
    # epsilon = sigma * np.sqrt(M + 2 * np.sqrt(M))

    # # Reconstruct the image by calling our ADMM implementation
    # xsol, fval, it = admm_conbpdn(
    #     y,
    #     epsilon,
    #     Phit,
    #     Phi,
    #     Psit,
    #     Psi,
    #     verbose=1,
    #     rel_tol=1e-4,
    #     rel_tol2=1e-4,
    #     max_iter=200,
    #     rho=1e2,
    #     delta=1.0,
    # )

    # # Calculate SNR and SSIM of the reconstructed image
    # snr = rsnr(xsol.cpu().numpy(), img.cpu().numpy())
    # ssim = metrics.structural_similarity(img.cpu().numpy(), xsol.cpu().numpy(), data_range=(img.max() - img.min()).item())

    # # Show and save the reconstructed image
    # # plt.figure()
    # # plt.imshow(xsol.cpu().numpy(), cmap="gray")
    # # plt.title("Reconstructed image - SNR: %.2f dB - SSIM: %.4f" % (snr, ssim))
    # # plt.axis("off")
    # # plt.savefig("reconstructed_image.png")
    # # plt.show()


