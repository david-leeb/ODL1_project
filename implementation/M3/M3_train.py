import argparse
import os
import pathlib
import pprint

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from tqdm import tqdm

# %% Useful functions
def snr_single(output, target):
    # compute signal-to-noise ratio (SNR) for a single image
    return 20 * torch.log10(torch.norm(target.ravel()) / torch.norm(target.ravel() - output.ravel()))

def add_noise(img, sigma):
    # add Gaussian noise to the image
    if "torch" in str(type(img)):
        noise = sigma * torch.randn_like(img)
    elif "numpy" in str(type(img)):
        noise = sigma * np.random.randn(*img.shape)
    return img + noise

# %% 1. Create PyTorch Dataset for training DnCNN

"""
Training the DnCNN needs pairs of groundtruth and noisy images. Use the
images in the training set as groundtruths. It is advisable to train
DnCNN on image patches rather than on the full image. The size of patches
should be 64x64 pixels and number of patches can be taken out of each
image. For noisy patches, you will add Gaussian noise with the standard
deviation sigma = 0.06, around 1 order of magnitude above the standard
deviation of the noise on the backprojected images (given an input SNR of
30dB). As in M2, it is advisable to split your training image patches
into two sets, one for training and one for an "in-training" validation
step.

Hint 1: You can use torchvision.transforms.random_crop to create patches
Hint 2: You can duplicate the ground truth images by the number of patches
then crop them randomly to the patch size


"""
class M3Dataset(Dataset):
    def __init__(self):
        pass # TODO: Implement the M3Dataset class
    
    def __len__(self):
        pass
    
    def __getitem__(self, i):
        pass

# %% 2. DnCNN architecture and implementation
"""
Here, you will create your DnCNN architecture
"""

class DnCNN(nn.Module):
    """
    DnCNN model
    """

    def __init__(self, n_ch_in=1, n_ch_out=1, n_ch=32, depth=20):
        """
        Initialize layers in DnCNN model

        :param n_ch_in: number of input channels, defaults to 1
        :type n_ch_in: int, optional
        :param n_ch_out: number of output channels, defaults to 1
        :type n_ch_out: int, optional
        :param n_ch: number of channels in DnCNN, defaults to 32
        :type n_ch: int, optional
        :param depth: depth of DnCNN, defaults to 20
        :type depth: int, optional
        """
        super().__init__()

        self.n_ch_in = n_ch_in
        self.n_ch_out = n_ch_out
        self.n_ch = n_ch
        self.depth = depth

        self.in_conv = nn.Conv2d(self.n_ch_in, self.n_ch, kernel_size=3, stride=1, padding=1, bias=True)

        # TODO: Implement the missing DnCNN layers
        # self.conv_list =

        self.out_conv = nn.Conv2d(self.n_ch, self.n_ch_out, kernel_size=3, stride=1, padding=1, bias=True)

        self.relu_list = nn.ModuleList([nn.ReLU() for _ in range(self.depth)])

    def forward(self, x_in):
        """
        Forward pass of DnCNN model
        """
        x_out = self.in_conv(x_in)
        x_out = self.relu_list[0](x_out)

        # TODO: Implement the forward function for the missing DnCNN layers
        
        # TODO: Implement the skip connection
        # x_out = 
        x_out = self.relu_list[-1](x_out)

        return x_out

# %% 3. Training the DnCNN

class M3Trainer:
    def __init__(
        self,
        ground_truth_path: str = "data/ground_truth",
        sigma: float = 0.06,
        patch_size_x: int = 64,
        patch_size_y: int = 64,
        num_patches: int = 5,
        batch_size: int = 1,
        num_epochs: int = 50,
        n_ch_in: int = 1,
        n_ch_out: int = 1,
        n_ch: int = 64,
        depth: int = 20,
        lr_init: float = 1e-4,
        model_save_path: str = "models",
        model_save_interval: int = 10,
        verbose: bool = True,
        verbose_interval: int = 100,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Trainer class for M3

        :param ground_truth_path: path to the ground truth images, defaults to "data/ground_truth"
        :type ground_truth_path: str, optional
        :param sigma: standard deviation of the noise, defaults to 0.06
        :type sigma: float, optional
        :param patch_size_x: patch size in x direction, defaults to 64
        :type patch_size_x: int, optional
        :param patch_size_y: patch size in y direction, defaults to 64
        :type patch_size_y: int, optional
        :param num_patches: number of patches per ground truth image, defaults to 5
        :type num_patches: int, optional
        :param batch_size: batch size, defaults to 1
        :type batch_size: int, optional
        :param num_epochs: total number of training epochs, defaults to 50
        :type num_epochs: int, optional
        :param n_ch_in: number of input channels to DnCNN, defaults to 1
        :type n_ch_in: int, optional
        :param n_ch_out: number of output channels of DnCNN, defaults to 1
        :type n_ch_out: int, optional
        :param n_ch: number of channels in DnCNN, defaults to 64
        :type n_ch: int, optional
        :param depth: depth of DnCNN, defaults to 20
        :type depth: int, optional
        :param lr_init: initial learning rate, defaults to 1e-4
        :type lr_init: float, optional
        :param model_save_path: path to save trained checkpoints, defaults to "models"
        :type model_save_path: str, optional
        :param model_save_interval: epoch interval to save trained checkpoints, defaults to 10
        :type model_save_interval: int, optional
        :param verbose: verbose, defaults to True
        :type verbose: bool, optional
        :param verbose_interval: verbose epoch interval, defaults to 100
        :type verbose_interval: int, optional
        :param device: device to train on, defaults to torch.device("cpu")
        :type device: torch.device, optional
        """
        self._ground_truth_path = ground_truth_path
        self._num_epochs = num_epochs
        self._verbose = verbose
        self._verbose_interval = verbose_interval
        self._device = device
        self._model_save_path = model_save_path
        self._model_save_interval = model_save_interval
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        # Load dataset into DataLoader
        # TODO: Define M3Dataset
        _dataset = M3Dataset()
        
        
        # TODO: Split dataset into training and validation
        # _train_dataset, _val_dataset =

        # Create DataLoaders for training and validation
        self._train_data_loader = DataLoader(dataset=_train_dataset, batch_size=batch_size, shuffle=True)
        self._val_data_loader = DataLoader(dataset=_val_dataset, batch_size=batch_size)

        # Initialize network
        self._model = DnCNN(n_ch_in=n_ch_in, n_ch_out=n_ch_out, n_ch=n_ch, depth=depth)
        print(f"DnCNN model with {sum(p.numel() for p in self._model.parameters())} parameters")
        # Move model to device
        self._model.to(self._device)

        # set up optimiser
        # TODO: define optimiser
        # self._optimiser = 

    def loss(self, output, target):
        # TODO: define loss function
        pass

    def snr(self, output, target):
        # compute signal-to-noise ratio (SNR)
        output = output.squeeze(1)
        target = target.squeeze(1)
        snrs = torch.zeros(output.shape[0])
        for i in range(output.shape[0]):
            snrs[i] = snr_single(output[i], target[i])
        return snrs.mean()

    def forward_step(self, data):
        # TODO: Implement forward step
        pass

    def training_step(self, data):
        # TODO: Implement trainig step
        pass

    def training_epoch(self, cur_iter):
        # TODO: Implement trainig epoch
        pass

    def validation_step(self, data):
        # TODO: Implement validation step
        pass

    def validation_epoch(self, cur_iter):
        # TODO: Implement validation epoch
        pass

    def train(self):
        for i in tqdm(range(self._num_epochs)):
            self.training_epoch(i)
            # validation
            with torch.no_grad():
                self.validation_epoch(i)
            # save model
            if (i + 1) % self._model_save_interval == 0 or i == self._num_epochs - 1:
                torch.save(
                    self._model.state_dict(), os.path.join(self._model_save_path, f"model_epoch_{i+1}.ckpt")
                )
            print("-" * 20)


def parse_args():
    parser = argparse.ArgumentParser(description="Train M3 model")
    parser.add_argument(
        "--ground_truth_path",
        type=str,
        default="data/ground_truth",
        help="Path to ground truth data",
    )
    parser.add_argument("--sigma", type=float, default=0.06, help="Standard deviation of image domain noise")
    parser.add_argument("--patch_size_x", type=int, default=64, help="Size of random patches in x-direction")
    parser.add_argument("--patch_size_y", type=int, default=64, help="Size of random patches in y-direction")
    parser.add_argument(
        "--num_patches", type=int, default=5, help="Number of random patches to extract per image"
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs to train")
    parser.add_argument("--n_ch_in", type=int, default=1, help="Number of input channels")
    parser.add_argument("--n_ch_out", type=int, default=1, help="Number of output channels")
    parser.add_argument("--n_ch", type=int, default=32, help="Number of channels")
    parser.add_argument("--depth", type=int, default=20, help="Depth of network")
    parser.add_argument("--lr_init", type=float, default=1e-4, help="Initial learning rate of the optimiser")
    parser.add_argument("--model_save_path", type=str, default="M3_trained_models", help="Path to save model")
    parser.add_argument("--model_save_interval", type=int, default=10, help="Saving the model every n epochs")
    parser.add_argument("--verbose_interval", type=int, default=100, help="Interval for verbose output")

    return parser.parse_args()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_args()
    pprint.pprint(vars(args))
    print(f"Training using device: {device}")
    # TODO: train M3
