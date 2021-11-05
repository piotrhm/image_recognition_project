import torch.nn as nn


def download_checkpoint(checkpoint_url: str, checkpoint_target_path: str) -> str:
    # Return checkpoint_target_path
    raise NotImplementedError()


def load_model(checkpoint_path: str, device: str = "cuda") -> nn.Module:
    raise NotImplementedError()
