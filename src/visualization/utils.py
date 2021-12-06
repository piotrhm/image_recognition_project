from typing import List, Tuple

import torch
import torch.nn as nn
import torchvision.transforms as tfs
import numpy as np


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
    and transforms it to only output prototype similarity scores for a single patch."""

    def _compute_similarity(x):
        """Return some kind of similarity, which is to be maximized"""
        distances = model.prototype_distances(x)
        distances = distances.view(distances.shape[0], model.num_prototypes, -1)

        # Log and then mean
        prototype_activations = model.distance_2_similarity(distances)
        prototype_activations = torch.mean(prototype_activations, dim=-1)

        # Mean and then log
        # distances = torch.mean(distances, dim=-1)
        # prototype_activations = model.distance_2_similarity(distances)

        # Fu*k the log
        # distances = torch.mean(distances, dim=-1)
        # prototype_activations = -distances

        return prototype_activations

    for f in model.parameters():
        f.requires_grad = False
    model.forward = _compute_similarity
    return model


mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

preprocess = tfs.Compose([tfs.ToTensor(), tfs.Normalize(mean, std)])


def deprocess(image_np):
    image_np = image_np.squeeze().transpose(1, 2, 0)
    image_np = image_np * std.reshape((1, 1, 3)) + mean.reshape((1, 1, 3))
    image_np = np.clip(image_np, 0.0, 255.0)
    return image_np
