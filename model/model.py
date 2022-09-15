"""This is unofficial implementation of YeNet:
Deep Learning Hierarchical Representation for Image Steganalysis.
"""

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

import numpy as np


class SRMConv(nn.Module):
    """This class computes convolution of input tensor with 30 SRM filters"""

    def __init__(self) -> None:
        """Constructor."""
        super().__init__()
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.srm = torch.from_numpy(np.load(".\\srm.npy")).to(
            self.device, dtype=torch.float
        )
        self.tlu = nn.Hardtanh(min_val=-3.0, max_val=3.0)

    def forward(self, inp: Tensor) -> Tensor:
        """Returns output tensor after convolution with 30 SRM filters
        followed by TLU activation."""
        return self.tlu(F.conv2d(inp, self.srm))


class ConvBlock(nn.Module):
    """This class returns building block for YeNet class."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        use_pool: bool = False,
        pool_size: int = 3,
        pool_padding: int = 0,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )
        self.activation = nn.ReLU()
        self.pool = nn.AvgPool2d(
            kernel_size=pool_size, stride=2, padding=pool_padding
        )
        self.use_pool = use_pool

    def forward(self, inp: Tensor) -> Tensor:
        """Returns conv->gaussian->average pooling."""
        if self.use_pool:
            return self.pool(self.activation(self.conv(inp)))
        return self.activation(self.conv(inp))


class YeNet(nn.Module):
    """This class returns YeNet model."""

    def __init__(self) -> None:
        super().__init__()
        self.layer1 = ConvBlock(30, 30, kernel_size=3)
        self.layer2 = ConvBlock(30, 30, kernel_size=3)
        self.layer3 = ConvBlock(
            30, 30, kernel_size=3, use_pool=True, pool_size=2, pool_padding=0
        )
        self.layer4 = ConvBlock(
            30,
            32,
            kernel_size=5,
            padding=0,
            use_pool=True,
            pool_size=3,
            pool_padding=0,
        )
        self.layer5 = ConvBlock(
            32, 32, kernel_size=5, use_pool=True, pool_padding=0
        )
        self.layer6 = ConvBlock(32, 32, kernel_size=5, use_pool=True)
        self.layer7 = ConvBlock(32, 16, kernel_size=3)
        self.layer8 = ConvBlock(16, 16, kernel_size=3, stride=3)
        self.fully_connected = nn.Sequential(
            nn.Linear(in_features=16 * 3 * 3, out_features=2),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, image: Tensor) -> Tensor:
        """Returns logit for the given tensor."""
        out = SRMConv()(image)
        out = nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.layer5,
            self.layer6,
            self.layer7,
            self.layer8,
        )(out)
        out = out.view(out.size(0), -1)
        out = self.fully_connected(out)
        return out


if __name__ == "__main__":
    net = YeNet()
    print(net)
    inp_image = torch.randn((1, 1, 256, 256))
    print(net(inp_image))
