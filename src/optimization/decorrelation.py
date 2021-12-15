# Code adapted from https://github.com/tensorflow/lucid/

from typing import Tuple

import numpy as np
import torch
from torch import Tensor

IMAGENET_COLOR_CORRELATION_SVD_SQRT = np.asarray([[0.26, 0.09, 0.02],
                                                  [0.27, 0.00, -0.05],
                                                  [0.27, -0.09, 0.03]]).astype("float32")
IMAGENET_COLOR_MEAN = np.asarray([0.48, 0.46, 0.41]).astype("float32")


class DecorrelatedImagesGenerator:
    """
    A class for generating decorrelated images.
    """
    def __init__(self, *,
                 shape: Tuple[int, int, int] = (3, 224, 224),
                 std: float = 0.1,
                 decay_power: float = 1.5,
                 magic_number: float = 4.0,
                 sigmoid: bool = True,
                 decorrelate: bool = True,
                 color_correlation_svd_sqrt: np.array = IMAGENET_COLOR_CORRELATION_SVD_SQRT,
                 color_mean: np.array = IMAGENET_COLOR_MEAN):
        self.shape = shape
        self.decay_power = decay_power
        self.std = std
        self.magic_number = magic_number
        self.sigmoid = sigmoid
        self.decorrelate = decorrelate
        self.color_correlation_svd_sqrt = color_correlation_svd_sqrt
        self.color_mean = color_mean

    def generate(self) -> Tensor:
        shape = self.shape
        fft_shape = (1, shape[1], shape[2], shape[0])
        x = _generate_fft_image(fft_shape, std=self.std, decay_power=self.decay_power)
        x = _to_valid_rgb(x, sigmoid=self.sigmoid, decorrelate=self.decorrelate,
                          color_correlation_svd_sqrt=self.color_correlation_svd_sqrt,
                          color_mean=self.color_mean)

        # restore wanted shape
        x = x.squeeze(0).permute(2, 0, 1).contiguous()

        return x


def _generate_fft_image(shape: Tuple[int, int, int, int],
                        std: float = 0.01,
                        decay_power: float = 1.0,
                        magic_number: float = 4.0) -> Tensor:
    """An image parametrization using 2D Fourier coefficients."""

    batch_size, height, width, channels = shape
    freqs = _rfft2d_freqs(height, width)
    init_value_size = (2, batch_size, channels) + freqs.shape

    init_value = np.random.normal(size=init_value_size, scale=std).astype(np.float32)
    spectrum_real_tensor = torch.tensor(init_value).float()
    spectrum_tensor = torch.complex(spectrum_real_tensor[0], spectrum_real_tensor[1])

    # Scale the spectrum. First normalize energy, then scale by the square-root
    # of the number of pixels to get a unitary transformation.
    # This allows to use similar learning rates to pixel-wise optimisation.
    scale = 1.0 / np.maximum(freqs, 1.0 / max(width, height)) ** decay_power
    scale *= np.sqrt(width * height)
    scale = torch.tensor(scale).float()
    scaled_spectrum_t = scale * spectrum_tensor

    image_tensor = torch.fft.irfft2(scaled_spectrum_t)
    image_tensor = image_tensor.permute((0, 2, 3, 1))

    # in case of odd spatial input dimensions we need to crop
    image_tensor = image_tensor[:batch_size, :height, :width, :channels]
    image_tensor = image_tensor / magic_number
    return image_tensor


def _rfft2d_freqs(h: int, w: int) -> np.array:
    """Computes 2D spectrum frequencies."""

    fy = np.fft.fftfreq(h)[:, None]
    # when we have an odd input dimension we need to keep one additional frequency and later cut off 1 pixel
    if w % 2 == 1:
        fx = np.fft.fftfreq(w)[: w // 2 + 2]
    else:
        fx = np.fft.fftfreq(w)[: w // 2 + 1]
    return np.sqrt(fx * fx + fy * fy)


def _to_valid_rgb(x: Tensor, *,
                  decorrelate: bool, sigmoid: bool,
                  color_correlation_svd_sqrt: np.array,
                  color_mean: np.array):
    if decorrelate:
        x = _linear_decorelate_color(x, color_correlation_svd_sqrt)
    if decorrelate and not sigmoid:
        x += torch.tensor(color_mean).float()
    if sigmoid:
        return torch.sigmoid(x)
    else:
        return _constrain_L_inf(2 * x - 1) / 2 + 0.5


def _linear_decorelate_color(image: Tensor, color_correlation_svd_sqrt: np.array) -> Tensor:
    max_norm_svd_sqrt = np.max(np.linalg.norm(color_correlation_svd_sqrt, axis=0))
    color_correlation_normalized = color_correlation_svd_sqrt / max_norm_svd_sqrt
    color_correlation_normalized = torch.tensor(color_correlation_normalized).float()

    flat = image.view(-1, 3)
    flat = torch.matmul(flat, color_correlation_normalized.T)
    image = torch.reshape(flat, image.shape)
    return image


def _constrain_L_inf(x):
    return x / torch.maximum(torch.ones_like(x), torch.abs(x))
