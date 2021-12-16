import torchvision.transforms as tfs

from typing import Tuple, Union


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
