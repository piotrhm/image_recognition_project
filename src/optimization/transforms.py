import logging
import numbers
import random
from abc import abstractmethod, ABC
from typing import Tuple, Union, Callable, List, Any

import numpy as np
import torch
import torchvision.transforms as tfs
from torch import Tensor

if torch.cuda.is_available():
    try:
        import bilateral_cuda
    except ImportError:
        import subprocess
        import sys

        logging.info('Installing bilateral-filter...')
        ret = subprocess.check_output(
            [sys.executable, '-m', 'pip', 'install', 'git+git://github.com/adriansuwala/bilateral-filter.git@main'])
        logging.info(ret.decode('utf-8'))
    import bilateral_cuda
else:
    logging.warning('Cuda not available, bilateral filter not imported')


class ReversibleTransform(ABC):
    """
    Transform that can be reversed.
    """

    @abstractmethod
    def __call__(self, x: Tensor) -> Tensor:
        ...

    @abstractmethod
    def reverse_transform(self, x: Tensor) -> Tensor:
        ...


class ReversibleCompose(ReversibleTransform):
    def __init__(self, transforms: List[Union[ReversibleTransform, Callable[[Tensor], Tensor]]]):
        self.transforms = transforms

    def __call__(self, x: Tensor) -> Tensor:
        for fn in self.transforms:
            x = fn(x)
        return x

    def reverse_transform(self, x: Tensor) -> Tensor:
        for fn in reversed(self.transforms):
            if isinstance(fn, ReversibleTransform):
                x = fn.reverse_transform(x)
        return x


class ReversibleRandomTransforms(ReversibleTransform):
    """
    Random transform that can be reversed. It stores its random parameters and uses them to reverse the transformation.
    Only the last transformation call can be reversed (and can be reversed only once).
    """

    def __init__(self, p: float):
        """
        Arguments:
            p: probability of performing a random transformation.
        """
        self.p = p
        self.params = None

    @abstractmethod
    def _get_random_params(self) -> Any:
        ...

    @abstractmethod
    def _transform_with_params(self, x: Tensor, params: Any) -> Tensor:
        ...

    @abstractmethod
    def _reverse_with_params(self, x: Tensor, params: Any) -> Tensor:
        ...

    def __call__(self, x: Tensor) -> Tensor:
        if np.random.binomial(1, self.p):
            self.params = self._get_random_params()
            x = self._transform_with_params(x, self.params)
        return x

    def reverse_transform(self, x: Tensor) -> Tensor:
        if self.params is not None:
            x = self._reverse_with_params(x, self.params)
            self.params = None
        return x


class ReversibleRandomRotate90(ReversibleRandomTransforms):
    def _get_random_params(self) -> Any:
        return random.choice([1, 2, 3])

    def _transform_with_params(self, x: Tensor, params: Any) -> Tensor:
        return torch.rot90(x, k=self.params, dims=(-2, -1))

    def _reverse_with_params(self, x: Tensor, params: Any) -> Tensor:
        return torch.rot90(x, k=-self.params, dims=(-2, -1))


class ReversibleRandomHorizontalFlip(ReversibleRandomTransforms):
    def _get_random_params(self) -> Any:
        return True

    def _transform_with_params(self, x: Tensor, params: Any) -> Tensor:
        return tfs.functional.hflip(x)

    def _reverse_with_params(self, x: Tensor, params: Any) -> Tensor:
        return tfs.functional.hflip(x)


class ReversibleRandomVerticalFlip(ReversibleRandomTransforms):
    def _get_random_params(self) -> Any:
        return True

    def _transform_with_params(self, x: Tensor, params: Any) -> Tensor:
        return tfs.functional.vflip(x)

    def _reverse_with_params(self, x: Tensor, params: Any) -> Tensor:
        return tfs.functional.vflip(x)


class ReversibleRandomColorJitter(ReversibleRandomTransforms):
    """
    Randomly change the brightness, contrast, saturation and hue of an image.
    Adapted from https://pytorch.org/vision/main/_modules/torchvision/transforms/transforms.html#ColorJitter.
    Works as tfs.RandomColorJittering if ``p=1.0``, but can be reversed.
    Args:
        p: probability of performing color jittering.
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """

    def __init__(self, p: float,
                 brightness: Union[float, Tuple[float, float]] = 0,
                 contrast: Union[float, Tuple[float, float]] = 0,
                 saturation: Union[float, Tuple[float, float]] = 0,
                 hue: Union[float, Tuple[float, float]] = 0):
        super().__init__(p)
        self.brightness = self._check_input(brightness, "brightness")
        self.contrast = self._check_input(contrast, "contrast")
        self.saturation = self._check_input(saturation, "saturation")
        self.hue = self._check_input(hue, "hue", center=0, bound=(-0.5, 0.5), clip_first_on_zero=False)

    def _check_input(self, value: Any, name, center=1, bound=(0, float("inf")), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(f"If {name} is a single number, it must be non negative.")
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError(f"{name} values should be between {bound}")
        else:
            raise TypeError(f"{name} should be a single number or a list/tuple with length 2.")

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    def _get_random_params(self) -> Any:
        fn_idx = torch.randperm(4)

        b = None if self.brightness is None else float(torch.empty(1).uniform_(self.brightness[0], self.brightness[1]))
        c = None if self.contrast is None else float(torch.empty(1).uniform_(self.contrast[0], self.contrast[1]))
        s = None if self.saturation is None else float(torch.empty(1).uniform_(self.saturation[0], self.saturation[1]))
        h = None if self.hue is None else float(torch.empty(1).uniform_(self.hue[0], self.hue[1]))

        return fn_idx, b, c, s, h

    def _transform_with_params(self, x: Tensor, params: Any) -> Tensor:
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = params
        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                x = tfs.functional.adjust_brightness(x, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                x = tfs.functional.adjust_contrast(x, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                x = tfs.functional.adjust_saturation(x, saturation_factor)
            elif fn_id == 3 and hue_factor is not None:
                x = tfs.functional.adjust_hue(x, hue_factor)

        return x

    def _reverse_with_params(self, x: Tensor, params: Any) -> Tensor:
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = params
        for fn_id in reversed(fn_idx):
            if fn_id == 0 and brightness_factor is not None:
                x = tfs.functional.adjust_brightness(x, 1 / brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                x = tfs.functional.adjust_contrast(x, 1 / contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                x = tfs.functional.adjust_saturation(x, 1 / saturation_factor)
            elif fn_id == 3 and hue_factor is not None:
                x = tfs.functional.adjust_hue(x, -hue_factor)
        return x


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

    def __call__(self, x: Tensor) -> Tensor:
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
    def __init__(self, kernel_size: int, sigma_v: Union[Tensor, float], sigma_s: Union[Tensor, float]):
        """
        Parameters:
            kernel_size: kernel size
            sigma_v: range sigma (increases blurring), tensor with value for each channel or single value for all
            sigma_s: spatial sigma, tensor with value for each channel or single value for all
        """
        self.sigma_v = sigma_v if isinstance(sigma_v, Tensor) else torch.tensor([sigma_v])
        self.sigma_s = sigma_s if isinstance(sigma_s, Tensor) else torch.tensor([sigma_s])
        self.sigma_v = self.sigma_v.float().cuda()
        self.sigma_s = self.sigma_s.float().cuda()
        self.kernel_size = kernel_size

    def __call__(self, x: Tensor) -> Tensor:
        squeeze = False
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
            squeeze = True
        self.sigma_v = self.sigma_v if self.sigma_v.shape[0] == x.shape[1] else self.sigma_v.repeat(
            x.shape[1]).float().cuda()
        self.sigma_s = self.sigma_s if self.sigma_s.shape[0] == x.shape[1] else self.sigma_s.repeat(
            x.shape[1]).float().cuda()
        assert self.sigma_v.shape[0] == x.shape[1]
        assert self.sigma_s.shape[0] == x.shape[1]
        self.input = x
        [ret, self.numerator, self.denominator] = bilateral_cuda.forward(x, self.kernel_size, self.sigma_v,
                                                                         self.sigma_s)
        return ret.squeeze(0) if squeeze else ret

    def backward(self, grad_output: Tensor) -> Tensor:
        return bilateral_cuda.backward(grad_output.clone(), self.input, self.kernel_size, self.sigma_v, self.sigma_s,
                                       self.numerator, self.denominator)


Transform = Callable[[Tensor], Tensor]
