import numpy as np
import torch
import torchvision.transforms as tfs
from torch import Tensor


class ClampingMinMax:
    """
    A class used to define clamping into the range [min, max].
    """

    def __init__(self, min: float = 0, max: float = 1):
        """
        Parameters:
            min: lower-bound of the range to be clamped to
            max: upper-bound of the range to be clamped to
        """
        self.min = min
        self.max = max

    def __call__(self, t: Tensor):
        """
        Clamps all elements in input into the range [min, max].
        """
        return torch.clamp(t, self.min, self.max)


class ClampingMeanStd:
    """
    A class used to define clamping into the range [-mean/std, mean/std] separately for each channel.
    """

    def __init__(self,
                 mean: np.array = np.array([0.485, 0.456, 0.406]),
                 std: np.array = np.array([0.229, 0.224, 0.225])):
        """
        Parameters:
            mean: mean
            std: std
        """
        self.mean = mean
        self.std = std

    def __call__(self, t: Tensor) -> Tensor:
        """
        Clamps all elements in input into the range [-mean/std, mean/std].
        """
        t_clamped = torch.zeros(t.shape).to(t.device)
        for c in range(3):
            m, s = self.mean[c], self.std[c]
            t_clamped[c, :, :] = torch.clamp(t[c, :, :], -m / s, (1 - m) / s)
        return t_clamped


class NormalizationMinMax:
    """
    A class used to define normalization to the range [min, max].
    """

    def __init__(self, min: float = 0, max: float = 1):
        """
        Parameters:
            min: lower-bound of the range to be normalized to
            max: upper-bound of the range to be normalized to
        """
        self.min = min
        self.max = max

    def __call__(self, t: Tensor) -> Tensor:
        """
        Normalizes all elements in input into the range [min, max].
        """
        t_min = torch.min(t)
        t_max = torch.max(t)
        t_norm = (self.max - self.min) * (t - t_min) / (t_max - t_min) + self.min
        return t_norm


class NormalizationMeanStd:
    """
    A class used to define normalization with mean and std.
    """

    def __init__(self,
                 mean: np.array = np.array([0.485, 0.456, 0.406]),
                 std: np.array = np.array([0.229, 0.224, 0.225])):
        """
        Parameters:
            mean: mean used for normalization
            std: standard deviation used for normalization
        """
        self.transform = tfs.Normalize(mean, std)

    def __call__(self, t: Tensor) -> Tensor:
        """
        Normalizes tensor image with mean and standard deviation.
        """
        return self.transform(t)


class DenormalizationMeanStd:
    """
    A class used to define denormalization with mean and std.
    """

    def __init__(self,
                 mean: np.array = np.array([0.485, 0.456, 0.406]),
                 std: np.array = np.array([0.229, 0.224, 0.225])):
        """
        Parameters:
            mean: mean used for normalization
            std: standard deviation used for normalization
        """
        self.mean = torch.tensor(mean).float().view(-1, 1, 1)
        self.std = torch.tensor(std).float().view(-1, 1, 1)

    def __call__(self, t: Tensor) -> Tensor:
        """
        Normalizes tensor image with mean and standard deviation.
        """
        return (t * self.std) + self.mean
