import torchvision.transforms as tfs

from typing import Tuple


class CustomTransforms:
    """
    Custom wrapper on torchvision.transforms
    """

    def __init__(
            self,
            color_jitter_args: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 0.25),
            random_crop_args: Tuple[int, Tuple[float, float]] = (224, (0.9, 1.1)),
            rotate: int = 5,
            gaussian_blur_args: Tuple[int, float] = (7, 2)
    ):
        """
        Parameters:
            color_jitter_args: Quadruple of (brightness, contrast, saturation, hue)
            rotate: Max value for range of degrees to select
            random_crop_args: Tripe of (size, (scale_min, scale_max))
            gaussian_blur_args: Tuple of (kernel_size, sigma)
        """

        self.transforms = tfs.Compose([
            tfs.ColorJitter(color_jitter_args[0], color_jitter_args[1], color_jitter_args[2], color_jitter_args[3]),
            tfs.RandomResizedCrop(size=random_crop_args[0], scale=random_crop_args[1]),
            tfs.RandomRotation(range(-rotate, rotate + 1)),
            tfs.GaussianBlur(kernel_size=gaussian_blur_args[0], sigma=gaussian_blur_args[1])
        ])

    def __call__(self, x):
        return self.transforms(x)

# Predefined transforms


transforms_base = tfs.Compose([
    tfs.ColorJitter(brightness=1.0, contrast=1.0, saturation=1.0, hue=0.25),
    tfs.RandomResizedCrop(size=224, scale=(0.9, 1.1)),
    tfs.RandomRotation(range(-5, 6)),
    tfs.GaussianBlur(kernel_size=7, sigma=2)
])
