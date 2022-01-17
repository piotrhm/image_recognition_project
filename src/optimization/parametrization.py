from typing import Callable, Optional

import torch
import math
from torch import nn, Tensor


# These activations and implementation of CPPN are adapted from
# colab link in https://distill.pub/2018/differentiable-parameterizations/


def composite_activation(x):
    x = torch.atan(x)
    # Coefficients computed by:
    #   def rms(x):
    #     return np.sqrt((x*x).mean())
    #   a = np.arctan(np.random.normal(0.0, 1.0, 10**6))
    #   print(rms(a), rms(a*a))
    return torch.concat([x/0.67, (x*x)/0.6], -1)


def composite_activation_unbiased(x):
    x = torch.atan(x)
    # Coefficients computed by:
    #   a = np.arctan(np.random.normal(0.0, 1.0, 10**6))
    #   aa = a*a
    #   print(a.std(), aa.mean(), aa.std())
    return torch.concat([x/0.67, (x*x-0.45)/0.396], -1)


def relu_normalized(x):
    x = nn.ReLU()(x)
    # Coefficients computed by:
    #   a = np.random.normal(0.0, 1.0, 10**6)
    #   a = np.maximum(a, 0.0)
    #   print(a.mean(), a.std())
    return (x-0.40)/0.58


def fourier_map(fourier_features, coords):
    sin = torch.sin(2 * math.pi * fourier_features(coords))
    cos = torch.cos(2 * math.pi * fourier_features(coords))
    return torch.cat([sin, cos], dim=-1)


class Lambda(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def __call__(self, x):
        return self.f(x)


class ImageCPPN(nn.Module):
    """
    Implementation of Compositional Pattern Producing Networks.
    Takes pixel coordinates and outputs rgb values.
    Can use fourier features to first embed coordinates.
    """

    def __init__(self, width: int,
                 height: Optional[int] = None,
                 num_output_channels: int = 3,
                 num_hidden_channels: int = 24,
                 num_layers: int = 8,
                 activation_fn: Callable = Lambda(relu_normalized),
                 normalize: bool = False,
                 embedding_size: int = 256,
                 scale: int = 10,
                 use_fourier_features: bool = False):
        """
        paramterers:
            width: width of an image
            height: height of an image, if None then equal to width
            num_output_channels: number of output channels
            num_hidden_channels: number of hidden channels
            num_layers: number of layers
            activation_fn: activation function for all layers except last one
            normalize: use instance norm or not
            embedding_size: fourier feature embedding size
            scale: std for normal distribution from which fourier features are drawn
            use_fourier_features: embed coordinates using fourier features or not
        """
        super().__init__()

        self.width = width
        self.height = height if height is not None else width
        self.embedding_size = embedding_size
        self.use_fourier_features = use_fourier_features

        r = 3.0**0.5  # std(coord_range) == 1.0
        width_range = torch.linspace(-r, r, self.width)
        height_range = torch.linspace(-r, r, self.height)
        grid_x, grid_y = torch.meshgrid(width_range, height_range, indexing='ij')
        self.grid = torch.stack([grid_x, grid_y]).unsqueeze(0)

        if use_fourier_features:
            self.fourier_features = nn.Linear(2, embedding_size)
            self.fourier_features.requires_grad = False
            nn.init.normal_(self.fourier_features.weight, std=scale)

        layers = []
        for i in range(num_layers - 1):
            if i == 0:
                in_ch = 2*self.embedding_size if self.use_fourier_features else 2
                layer = nn.Conv2d(in_ch, num_hidden_channels, 1)
            else:
                layer = nn.Conv2d(num_hidden_channels, num_hidden_channels, 1)
            layers.append(layer)
            if normalize:
                layers.append(nn.InstanceNorm2d(num_hidden_channels))
            layers.append(activation_fn)

        final = nn.Conv2d(num_hidden_channels, num_output_channels, 1)
        with torch.no_grad():
            final.weight.zero_()
        layers.append(final)
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def to(self, device: str) -> nn.Module:
        self.model = self.model.to(device)
        self.grid = self.grid.to(device)
        if self.use_fourier_features:
            self.fourier_features = self.fourier_features.to(device)
        return self

    def __call__(self, grid: Tensor = None) -> Tensor:
        grid = self.grid if grid is None else grid
        if grid.dim() == 3:
            grid = grid.unsqueeze(0)
        if self.use_fourier_features:
            emb = fourier_map(self.fourier_features, self.grid.view(-1, 2))
            emb = emb.view(grid.shape[0], 2*self.embedding_size, self.height, self.width)
            grid = emb
        return self.model(grid)


# TODO: FFT generator
