from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_output_mask_from_prototypes_list(model: nn.Module, prototypes_list: List[Tuple[int, int]]) -> torch.tensor:
    """Takes prototypes as a list of pairs (class index, prototype index)
    and returns a boolean tensor that masks out everything else."""
    proto_per_class = model.num_prototypes // model.num_classes
    ret = torch.zeros(model.num_prototypes, dtype=bool).to(next(model.parameters()).device)
    for i, j in prototypes_list:
        ret[i * proto_per_class + j] = True
    return ret


def prepare_model_for_prototype_optimization(model: nn.Module) -> nn.Module:
    """Takes a ProtoPNet model, freezes all its parameters
    and transforms it to only output prototype similarity scores."""

    def _forward(x):
        distances = model.prototype_distances(x)
        # global min pooling
        min_distances = -F.max_pool2d(-distances,
                                      kernel_size=(distances.size()[2],
                                                   distances.size()[3]))
        min_distances = min_distances.view(-1, model.num_prototypes)
        prototype_activations = model.distance_2_similarity(min_distances)
        return prototype_activations

    for f in model.parameters():
        f.requires_grad = False
    model.forward = _forward
    return model
