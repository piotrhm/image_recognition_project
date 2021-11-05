from typing import List, Tuple

import torch
import torch.nn as nn


def get_output_mask_from_prototypes_list(model: nn.Module, prototypes_list: List[Tuple[int, int]]) -> torch.tensor:
    # prototypes_list consist of pairs (class_idx, prototype_idx)
    raise NotImplementedError()


def prepare_model_for_prototype_optimization(model: nn.Module) -> nn.Module:
    # Freeze the entire model and somehow transform (cut?) it so it outputs the propotype-similarity layer
    raise NotImplementedError()
