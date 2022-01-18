from typing import Callable, Optional, Tuple

import torch
import math
import numpy as np
from torch import nn, Tensor

from src.optimization.decorrelation import _rfft2d_freqs, _linear_decorelate_color, \
        IMAGENET_COLOR_CORRELATION_SVD_SQRT, IMAGENET_COLOR_MEAN, _to_valid_rgb


def fourier_map(fourier_features, coords):
    sin = torch.sin(2 * math.pi * fourier_features(coords))
    cos = torch.cos(2 * math.pi * fourier_features(coords))
    return torch.cat([sin, cos], dim=-1)


class Lambda(nn.Module):
    """For using a standalone function as an activation function in nn.Sequential."""

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
    Implementation of CPPN are adapted from colab link in
    https://distill.pub/2018/differentiable-parameterizations/
    """

    def __init__(self, width: int,
                 height: Optional[int] = None,
                 num_output_channels: int = 3,
                 num_hidden_channels: int = 24,
                 num_layers: int = 8,
                 activation_fn: Callable = nn.ReLU(),
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
            final.bias.zero_()
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


class FFTGenerator(nn.Module):
    """Parametrization using fourier space. Adapted from lucid."""

    def __init__(self, shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
                 std: float = 0.01, decay_power: float = 1.0,
                 sigmoid: bool = False,
                 decorrelate: bool = True,
                 color_correlation_svd_sqrt: np.array = IMAGENET_COLOR_CORRELATION_SVD_SQRT,
                 color_mean: np.array = IMAGENET_COLOR_MEAN):
        self.shape = shape
        self.decay_power = decay_power
        self.std = std
        self.sigmoid = sigmoid
        self.decorrelate = decorrelate
        self.color_correlation_svd_sqrt = color_correlation_svd_sqrt
        self.color_mean = color_mean

        batch_size, channels, width, height = self.shape  # flipped to pytorch convention
        self.freqs = _rfft2d_freqs(height, width)
        init_value_size = (2, batch_size, channels) + self.freqs.shape
        spectrum_real_tensor = std * torch.randn(init_value_size)
        self.spectrum_tensor = torch.complex(spectrum_real_tensor[0], spectrum_real_tensor[1])
        self.spectrum_tensor.requires_grad = True

    def to(self, device):
        with torch.no_grad():
            self.spectrum_tensor.data = self.spectrum_tensor.to(device)
        return self

    def get_image(self) -> torch.tensor:
        batch_size, channels, width, height = self.shape  # flipped to pytorch convention

        scale = 1.0 / np.maximum(self.freqs, 1.0 / max(width, height)) ** self.decay_power
        scale *= np.sqrt(width * height)
        scale = torch.tensor(scale).float().to(self.spectrum_tensor.device)
        scaled_spectrum_t = scale.view(1, scale.shape[-2], scale.shape[-1]) * self.spectrum_tensor

        image_tensor = torch.fft.irfft2(scaled_spectrum_t)

        # in case of odd spatial input dimensions we need to crop
        image_tensor = image_tensor[:batch_size, :channels, :height, :width]
        image_tensor = image_tensor / 4.0  # some magic number
        return image_tensor

    def __call__(self) -> Tensor:
        image = self.get_image()
        image = _to_valid_rgb(image, sigmoid=self.sigmoid, decorrelate=self.decorrelate,
                              color_correlation_svd_sqrt=self.color_correlation_svd_sqrt,
                              color_mean=self.color_mean)
        return image
