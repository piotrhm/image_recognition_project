import torch
import numpy as np
import torchvision.transforms as tfs
from typing import List, Callable


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

    def __call__(self, t):
        """
        Clamps all elements in input into the range [min, max].
        """
        t.data = torch.clamp(t.data, self.min, self.max)
        return t


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

    def __call__(self, t):
        """
        Clamps all elements in input into the range [-mean/std, mean/std].
        """
        t_clamped = torch.zeros(t.shape)
        for c in range(3):
            m, s = self.mean[c], self.std[c]
            t_clamped[c, :, :] = torch.clamp(t[c, :, :], -m / s, (1 - m) / s)
        t.data = t_clamped.data
        return t


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

    def __call__(self, t):
        """
        Normalizes all elements in input into the range [min, max].
        """
        t_min = torch.min(t)
        t_max = torch.max(t)
        t_norm = (self.max - self.min) * (t - t_min) / (t_max - t_min) + self.min
        t.data = t_norm.data
        return t


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
        self.mean = mean
        self.std = std

    def __call__(self, t):
        """
        Normalizes tensor image with mean and standard deviation.
        """
        t.data = tfs.Normalize(self.mean, self.std)(t).data
        return t


class BeforeOptimFn:
    """
    A class used to define a sequence of transformations on the input.
    """

    def __init__(self, methods: List[Callable]):
        """
        Parameters:
            methods: list of transformations
        """
        self.methods = methods

    def __call__(self, t):
        """
        Applies all transformations to the input.
        """
        for m in self.methods:
            t.data = m(t).data
        return t