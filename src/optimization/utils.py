import torch
import torchvision.transforms as tfs
import numpy as np


class BeforeOptim:
    def __init__(self, method: str):
        if method == "clamp":
            self.method = lambda x: torch.clamp(x, 0, 1)
        elif method == "normalize":
            self.method = self.__normalize
        elif method == "sigmoid":
            self.method = lambda x: torch.sigmoid(x)
        elif method == "clamp_dd":
            self.method = self.__clamp_dd
        else:
            raise ValueError("Value Error!")

    def __normalize(self, t):
        t_min = torch.min(t)
        t_max = torch.max(t)
        t_norm = (t - t_min) / (t_max - t_min)
        return t_norm

    def __clamp_dd(self, t):
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        t_clamped = torch.zeros((3, 224, 224))
        for c in range(3):
            m, s = mean[c], std[c]
            t_clamped[c, :, :] = torch.clamp(t[c, :, :], -m / s, (1 - m) / s)
        return t_clamped

    def __call__(self, t):
        tt = self.method(t)
        t.data = tfs.GaussianBlur(7, 2)(tt).data
        return t
