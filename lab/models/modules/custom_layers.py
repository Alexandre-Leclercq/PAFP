from typing import Optional
from torch import nn
import torch


class ChannelAffine(nn.Module):
    def __init__(self, num_channels):
        super().__init__()

        # Initialize gamma to 1.0 (Identity multiplier)
        # Shape: (1, C, 1, 1) allows automatic broadcasting over (B, C, H, W)
        self.gamma = nn.Parameter(torch.ones(1, num_channels, 1, 1))

        # Initialize beta to 0.0 (Identity offset)
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

    def forward(self, x):
        # Optimal element-wise operation
        return self.gamma * x + self.beta

class HEADAffine(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Initialize gamma to 1.0 (Identity multiplier)
        # Shape: (1, C) allows automatic broadcasting over (B, C)
        self.gamma = nn.Parameter(torch.ones(1, num_classes))

        # Initialize beta to 0.0 (Identity offset)
        self.beta = nn.Parameter(torch.zeros(1, num_classes))

    def forward(self, x):
        # Optimal element-wise operation
        return self.gamma * x + self.beta

def id_init_conv(m):
    """
    Initializes a Conv2d layer to be an identity mapping.
    The weight kernel is initialized to a Dirac delta function.
    The bias is initialized to zero.
    """
    # Check if the module is a 2D convolutional layer
    if isinstance(m, nn.Conv2d):
        # Get the weight tensor
        weight = m.weight.data

        # Ensure that the number of input and output channels are the same.
        # IDInit is only well-defined when channels are preserved.
        # if m.in_channels != m.out_channels:
        #     raise ValueError(
        #         "IDInit requires the number of input and output channels to be equal."
        #     )

        # Set all weights to zero
        weight.fill_(0)

        # Find the center of the kernel
        mid_h = m.kernel_size[0] // 2
        mid_w = m.kernel_size[1] // 2

        # Find the minimum of in_channels and out_channels
        min_channels = min(m.in_channels, m.out_channels)

        # This creates the Dirac delta function for each feature map
        for i in range(min_channels):
            weight[i, i, mid_h, mid_w] = 1.0

        # Initialize bias to zero
        if m.bias is not None:
            m.bias.data.fill_(0)

class ChannelLayerNorm(nn.Module):
    def __init__(self, num_channels: int, eps: Optional[float] = None):
        super(ChannelLayerNorm, self).__init__()
        if eps is None:
            self.layer_norm = nn.LayerNorm(num_channels)
        else:
            self.layer_norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):
        # Permute: (N, C, H, W) -> (N, H, W, C)
        x = x.permute(0, 2, 3, 1)
        # Apply Norm
        x = self.layer_norm(x)
        # Permute back: (N, H, W, C) -> (N, C, H, W)
        x = x.permute(0, 3, 1, 2)
        return x
