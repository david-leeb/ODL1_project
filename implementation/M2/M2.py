"""
M2.m file
Please read carefully the comments below. Sections 1 and 2 contain useful
functions to generate the training data and ImageDatastores. Section 3
contains a template for the implementation of the U-net. Sections 4 and 5
are reserved for training the U-net and validating the method.
"""

# %% load libraries
import argparse
import os

# import pprint
import glob

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from implementation.M1.M1 import create_meas_op, gen_noise, create_mask

# %% Section 1: Generate backprojected data

# Generate a dataset of backprojected images and save them in a new folder
# called 'trainingset_dirty' with the same name as the groundtruth image.
# To keep the integrity (scale and floating point format) of backprojected images save
# them in 'tif' file format. The codes below does this for you. You may
# need to adjust directories to read and save images.


def gen_backprojected_image(path_gdth: str, path_dirty: str, isnr: float = 30):
    """
    Generate backprojected image from ground truth image.

    :param path_gdth: path to the ground truth image
    :type path_gdth: str
    :param path_dirty: path where the dirty images will be saved
    :type path_dirty: str
    :param isnr: input signal-to-noise ratio
    :type isnr: float
    """
    list_gdth_files = glob.glob(os.path.join(path_gdth, "*.tiff"))
    if not os.path.exists(path_dirty):
        os.makedirs(path_dirty)
    for gdth_file in list_gdth_files:
        # Load the ground truth image
        gdth = np.array(Image.open(gdth_file)).astype(np.float32)
        gdth = torch.tensor(gdth, dtype=torch.float32)

        # Generate backprojected image
        N1, N2 = gdth.shape
        mask = create_mask(N1, N2)
        noise, _ = gen_noise(gdth, mask, isnr)
        Phit, Phi = create_meas_op(mask)
        y = Phit(gdth) + noise
        bp_y = Phi(y)

        # Save the backprojected image
        gdth_name = os.path.basename(gdth_file)
        dirty_file = os.path.join(path_dirty, gdth_name)
        Image.fromarray(bp_y.numpy()).save(dirty_file)


# %% Section 2: Create Dataset class for training

# To handle a large number of files, PyTorch provides the Dataset class to
# efficiently load data for training. Here, we define a custom Dataset
# class to load the backprojected images and groundtruths from their
# respective folders.

# For more help, please visit:
# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html


class M2Dataset(Dataset):
    """
    PyTorch Dataset class for M2
    """

    def __init__(self, path_gdth: str, path_dirty: str):

        self._path_gdth = path_gdth
        self._path_dirty = path_dirty
        self._ground_truth_files = glob.glob(os.path.join(path_gdth, "*.tiff"))

    def __len__(self):
        """
        Return the number of samples in the dataset
        """
        return len(self._ground_truth_files)

    def __getitem__(self, idx: int):
        """
        Return the training pair at the given index
        """
        data = {}
        data["ground_truth"] = np.array(
            Image.open(self._ground_truth_files[idx])
        ).astype(np.float32)
        data["ground_truth"] = torch.tensor(
            data["ground_truth"], dtype=torch.float32
        ).view(1, *data["ground_truth"].shape)
        data["dirty"] = np.array(
            Image.open(
                os.path.join(
                    self._path_dirty, os.path.basename(self._ground_truth_files[idx])
                )
            )
        ).astype(np.float32)
        data["dirty"] = torch.tensor(data["dirty"], dtype=torch.float32).view(
            1, *data["dirty"].shape
        )

        return data


# %% Section 3: U-net architecture and implementation

# See project document Appendix A for architecture details.
# The U-net architecture is implemented in the following class.


# A typical U-net architecture consists of two main parts: the down-sampling
# part and the up-sampling part. The down-sampling part can been seen as a series
# of blocks, each block containing a max-pooling layer followed by two parirs of convolutional
# layers with ReLU Layer.
class UNetDownBlock(torch.nn.Module):
    """
    UNet down sampling block
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_block = torch.nn.Sequential(
            # add proper layers here
        )

    def forward(self, x_in: torch.Tensor):
        """
        Forward pass of UNetDownBlock
        """
        return self.conv_block(x_in)


# The up-sampling part of the U-net architecture can be seen as a series of blocks as well.
# Each block contains a transposed convolutional layer and a ReLU layer followed by two
# pairs of convolutional layers with ReLU activation.
class UNetUpBlock(torch.nn.Module):
    """
    UNet up sampling block
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up_block = torch.nn.Sequential(
            # add proper layers here
        )
        self.conv_block = torch.nn.Sequential(
            # add proper layers here
        )

    def forward(self, x_in: torch.Tensor, x_bridge: torch.Tensor):
        """
        Forward pass of UNetUpBlock
        """
        x_out = self.up_block(x_in)
        # concatenate with the feature maps from down-sampling part
        x_out = torch.cat([x_out, x_bridge], dim=1)
        x_out = self.conv_block(x_out)
        return x_out


# Now, we can define the U-net architecture by connecting the down-sampling
# and up-sampling blocks.
class UNet(torch.nn.Module):
    """
    UNet model
    """

    def __init__(self, n_ch_in=1, n_ch_out=1, n_ch=64, depth=3):
        """
        Initialize layers in UNet model
        """
        super().__init__()

        self.n_ch_in = n_ch_in
        self.n_ch_out = n_ch_out
        self.n_ch = n_ch
        self.depth = depth

        self.conv_init = torch.nn.Sequential(
            # add proper layers here
        )

        self.down_blocks = torch.nn.ModuleList(
            # add down-sampling blocks here
        )

        self.up_blocks = torch.nn.ModuleList(
            # add up-sampling blocks here
        )

        self.conv_final = torch.nn.Sequential(
            # add proper layers here
        )

    def forward(self, x_in):
        """
        Forward pass of UNet model
        """

        # connect the blocks here

        return x_in


# %% Section 4: U-net training

# You can use the following class to train the network.
# It is highly recommended to load this class in a separate script and then
# execute the training process.

# Note: As per standard procedure, a validation step is introduced during
# training to evaluate the performance of neural networks. The dataset will be
# split into two sets, one for training and one for the validation step.
# This validation is not to be confused with the general validation of the method
# M2, which is performed on the testing set.


class M2Trainer:
    """
    M2 Trainer class
    """

    def __init__(
        self,
        path_gdth: str = "../dataset/trainingset_gdth",
        path_dirty: str = "../dataset/ground_truth",
        batch_size: int = 1,
        num_epochs: int = 50,
        n_ch_in: int = 1,
        n_ch_out: int = 1,
        n_ch: int = 64,
        depth: int = 3,
        lr_init: float = 1e-4,
        model_save_path: str = "m2_models",
        model_save_interval: int = 10,
        verbose: bool = True,
        verbose_interval: int = 100,
        device: torch.device = torch.device("cpu"),
    ):
        self._path_gdth = path_gdth
        self._path_dirty = path_dirty
        self._num_epochs = num_epochs
        self._verbose = verbose
        self._verbose_interval = verbose_interval
        self._device = device
        self._model_save_path = model_save_path
        self._model_save_interval = model_save_interval
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        # Load dataset into DataLoader
        _dataset = M2Dataset(path_gdth=path_gdth, path_dirty=path_dirty)
        # Split dataset into training and validation
        _train_dataset, _val_dataset = torch.utils.data.random_split(
            _dataset,
            [
                int(np.floor(0.8 * len(_dataset))),
                len(_dataset) - int(np.floor(0.8 * len(_dataset))),
            ],
        )
        # Create DataLoaders for training and validation
        self._train_data_loader = DataLoader(
            dataset=_train_dataset, batch_size=batch_size
        )
        self._val_data_loader = DataLoader(dataset=_val_dataset, batch_size=batch_size)

        # Initialize network
        self._model = UNet(n_ch_in=n_ch_in, n_ch_out=n_ch_out, n_ch=n_ch, depth=depth)

        # Move model to device
        self._model.to(self._device)

        # set up optimiser
        self._optimiser = torch.optim.Adam(self._model.parameters(), lr=lr_init)

    def loss(self, output, target):
        """
        Define loss function
        """
        return torch.nn.functional.mse_loss(output, target)

    def snr(self, output, target):
        """
        Compute signal-to-noise ratio (SNR)
        """
        # compute signal-to-noise ratio (SNR)
        output = output.squeeze(1)
        target = target.squeeze(1)
        snrs = torch.zeros(output.shape[0])
        for i in range(output.shape[0]):
            snrs[i] = 20 * torch.log10(
                torch.norm(target[i].ravel())
                / torch.norm(target[i].ravel() - output[i].ravel())
            )
        return snrs.mean()

    def forward_step(self, data):
        """
        Forward pass of the model
        """
        for k, v in data.items():
            if "torch" in str(type(v)):
                data[k] = v.to(self._device)
        self._optimiser.zero_grad()
        return self._model(data["dirty"])

    def training_step(self, data):
        """
        Training step.
        Forward pass, loss computation, and then backpropagation.
        """
        _output = self.forward_step(data)
        loss = self.loss(_output, data["ground_truth"])
        loss.backward()
        self._optimiser.step()
        return loss

    def training_epoch(self, cur_iter):
        """
        Training epoch.
        For each batch, run the training_step.
        """
        self._model.train()
        pbar = tqdm(self._train_data_loader)
        losses = []
        for i, data in enumerate(pbar):
            loss = self.training_step(data)
            losses.append(loss)
            if (
                i % self._verbose_interval == 0 or i == len(pbar) - 1
            ) and self._verbose:
                print(
                    f"Epoch {cur_iter+1} -",
                    f"batch {i+1} / {len(pbar)}:",
                    f"Training loss={loss.mean().item():.4e}",
                    flush=True,
                )
        if self._verbose:
            print(
                f"Epoch {cur_iter+1} -",
                f"Averaged training loss={torch.tensor(losses).mean().item():.4e}",
            )

    def validation_step(self, data):
        """
        Validation step.
        Forward pass and compute loss and SNR.
        """
        _output = self.forward_step(data)
        loss = self.loss(_output, data["ground_truth"])
        snr = self.snr(_output, data["ground_truth"])
        return loss, snr

    def validation_epoch(self, cur_iter):
        """
        Validation epoch.
        For each batch, run the validation_step.
        """
        self._model.eval()
        pbar = tqdm(self._val_data_loader)
        losses = []
        snrs = []
        for i, data in enumerate(pbar):
            loss, snr = self.validation_step(data)
            losses.append(loss)
            snrs.append(snr)
            if (
                i % self._verbose_interval == 0 or i == len(pbar) - 1
            ) and self._verbose:
                print(
                    f"Epoch {cur_iter+1} - batch {i+1} / {len(pbar)}:",
                    f"Validation loss={loss.mean().item():.4e};",
                    f"SNR={snr.mean().item():.4f}",
                    flush=True,
                )
        if self._verbose:
            print(
                f"Epoch {cur_iter+1} -"
                f"Averaged validation loss={torch.tensor(losses).mean().item():.4e},",
                f"SNR={torch.tensor(snrs).mean().item():.4f}",
            )

    def train(self):
        """
        Train the model.
        """
        for i in tqdm(range(self._num_epochs)):
            self.training_epoch(i)
            # validation
            with torch.no_grad():
                self.validation_epoch(i)
            # save model
            if (i + 1) % self._model_save_interval == 0 or i == self._num_epochs - 1:
                torch.save(
                    self._model.state_dict(),
                    os.path.join(self._model_save_path, f"model_epoch_{i+1}.ckpt"),
                )
            print("-" * 20)


# This function can parse the arguments from the command line.
# It would be useful to run the training script from the command line and change
# the hyperparameters without modifying the code.
def parse_args():
    """
    Parse arguments from command line.
    """
    parser = argparse.ArgumentParser(description="Train M2 model")
    parser.add_argument(
        "--ground_truth_path",
        type=str,
        default="data/ground_truth",
        help="Path to ground truth data",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for training"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=50, help="Number of epochs to train"
    )
    parser.add_argument(
        "--n_ch_in", type=int, default=1, help="Number of input channels"
    )
    parser.add_argument(
        "--n_ch_out", type=int, default=1, help="Number of output channels"
    )
    parser.add_argument("--n_ch", type=int, default=64, help="Number of channels")
    parser.add_argument("--depth", type=int, default=3, help="Depth of network")
    parser.add_argument(
        "--lr_init",
        type=float,
        default=1e-4,
        help="Initial learning rate of the optimiser",
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        default="M2_trained_models",
        help="Path to save model",
    )
    parser.add_argument(
        "--model_save_interval",
        type=int,
        default=10,
        help="Saving the model every n epochs",
    )
    parser.add_argument(
        "--verbose_interval", type=int, default=100, help="Interval for verbose output"
    )

    return parser.parse_args()


# %% M2 validation

# You will validate your implementation of M2 on the images provided in the
# testing set.

# 1. Load the files in the folder containing the testing set
#    Hint: you can use built-in functions: 'glob.glob' and 'os.path.join'

# 2. Perform in a loop:
#    i) read a new image
#   ii) build a Fourier mask using a random seed to ensure the mask is
#   different for each image
#  iii) create a measurement operator and its adjoint (Phit and Phi)
#   iv) compute the observed Fourier coefficients
#    v) calculate value of a sigma for ISNR = 30dB
#   vi) add noise to the Fourier coefficients to generate the observed data
#  vii) generate a backprojected image and record its SNR and SSIM
# viii) perform the reconstruction by inputing the backprojection to the
#       U-net
#   ix) calculate the SNR and SSIM and save the reconstructed image

if __name__ == "__main__":
    # create the training set
    gen_backprojected_image(
        path_gdth="../dataset/trainingset_gdth",
        path_dirty="../dataset/trainingset_dirty",
        isnr=30,
    )

    # training
    # hyperparameters should be adapted properly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training using device: {device}")
    trainer = M2Trainer(
        path_gdth="../dataset/trainingset_gdth",
        path_dirty="../dataset/trainingset_dirty",
        num_epochs=2,
        verbose_interval=1,
        device=device,
    )
    trainer.train()

    # validation
    # load the model
    model = UNet()
    model.load_state_dict(torch.load("./m2_models/model_epoch_2.ckpt", weights_only=True))
    model.to(device)
    # inference ...
