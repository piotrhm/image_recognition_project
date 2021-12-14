import torch
import subprocess
import sys

import torchvision.transforms as tfs

from typing import Tuple, Union

if torch.cuda.is_available():
    import os
    print('Installing bilateral-filter...')
    ret = subprocess.check_output([sys.executable, '-m', 'pip', 'install', 'git+git://github.com/adriansuwala/bilateral-filter.git@main'])
    print(ret.decode('utf-8'))
    import bilateral_cuda
else:
    print('Cuda not available, bilateral filter not imported')



class CustomTransforms:
    """
    Custom wrapper on torchvision.transforms
    """

    def __init__(
            self,
            padding: int = 0,
            color_jitter_args: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 0.25),
            random_crop_args: Tuple[int, Tuple[float, float]] = (224, (0.9, 1.1)),
            rotate: int = 5,
            gaussian_blur_args: Union[Tuple[int, float], None] = (7, 2)
    ):
        """
        Parameters:
            color_jitter_args: Quadruple of (brightness, contrast, saturation, hue)
            rotate: Max value for range of degrees to select
            random_crop_args: Tripe of (size, (scale_min, scale_max))
            gaussian_blur_args: Tuple of (kernel_size, sigma)
        """

        if gaussian_blur_args is None:
            tfs_blurring = tfs.Lambda(lambda x: x)
        else:
            tfs_blurring = tfs.GaussianBlur(kernel_size=gaussian_blur_args[0], sigma=gaussian_blur_args[1])

        self.transforms = tfs.Compose([
            tfs.ToPILImage(),
            tfs.Pad(padding=padding),
            tfs.ColorJitter(brightness=color_jitter_args[0],
                            contrast=color_jitter_args[1],
                            saturation=color_jitter_args[2],
                            hue=color_jitter_args[3]),
            tfs.RandomRotation((-rotate, rotate)),
            tfs.RandomResizedCrop(size=random_crop_args[0], scale=random_crop_args[1]),
            tfs_blurring,
            tfs.ToTensor(),
        ])

    def __call__(self, x):
        return self.transforms(x)

# Predefined transforms


transforms_base = tfs.Compose([
    tfs.ToPILImage(),
    tfs.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0),
    tfs.RandomRotation((-1, 1)),
    tfs.RandomResizedCrop(size=224, scale=(0.99, 1)),
    tfs.ToTensor(),
])

class BilateralFilter:
    def __init__(self, kernel_size: int, sigma_v: Union[torch.tensor, float], sigma_s: Union[torch.tensor, float]):
        """
        Parameters:
            kernel_size: kernel size
            sigma_v: range sigma (increases blurring), tensor with value for each channel or single value for all
            sigma_s: spatial sigma, tensor with value for each channel or single value for all
        """
        self.sigma_v = sigma_v if isinstance(sigma_v, torch.Tensor) else torch.tensor([sigma_v])
        self.sigma_s = sigma_s if isinstance(sigma_s, torch.Tensor) else torch.tensor([sigma_s])
        self.sigma_v = self.sigma_v.float().cuda()
        self.sigma_s = self.sigma_s.float().cuda()
        self.kernel_size = kernel_size

    def __call__(self, x: torch.tensor) -> torch.tensor:
        squeeze = False
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
            squeeze = True
        self.sigma_v = self.sigma_v if self.sigma_v.shape[0] == x.shape[1] else self.sigma_v.repeat(x.shape[1]).float().cuda()
        self.sigma_s = self.sigma_s if self.sigma_s.shape[0] == x.shape[1] else self.sigma_s.repeat(x.shape[1]).float().cuda()
        assert self.sigma_v.shape[0] == x.shape[1]
        assert self.sigma_s.shape[0] == x.shape[1]
        self.input = x
        [ret, self.numerator, self.denominator] = bilateral_cuda.forward(x, self.kernel_size, self.sigma_v, self.sigma_s)
        return ret.squeeze(0) if squeeze else ret

    def backward(self, grad_output: torch.tensor) -> torch.tensor:
        return bilateral_cuda.backward(grad_output.clone(), self.kernel_size, self.sigma_v, self.sigma_s, self.numerator, self.denominator)
