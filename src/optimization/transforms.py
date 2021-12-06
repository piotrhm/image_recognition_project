import torchvision.transforms as tfs

from typing import Tuple


class CustomTransforms:
    def __init__(
        self,
        brightness: float = 1.0,
        contrast: float = 1.0,
        saturation: float = 1.0,
        hue: float = 0.25,
        rotate: int = 5,
        scale: Tuple[float, float] = (0.9, 1.1),
        size: int = 224,
        kernel_size: int = 7,
        sigma: float = 2
    ):

        """
        Custom wrapper on torchvision.transforms
        """
        self.transforms = tfs.Compose([
            tfs.ColorJitter(brightness, contrast, saturation, hue),
            tfs.RandomResizedCrop(size=size, scale=scale),
            tfs.RandomRotation(range(-rotate, rotate + 1)),
            tfs.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
        ])

    def __call__(self, x):
        return self.transforms(x)
