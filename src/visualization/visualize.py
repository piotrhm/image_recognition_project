from typing import List, Tuple, Callable, Type, Dict, Any, Optional

import torch
import torch.nn as nn

from src.optimization.optimize import optimize_model
from .utils import prepare_model_for_prototype_optimization, get_output_mask_from_prototypes_list


def visualize_prototypes(model: nn.Module,
                         prototypes_list: List[Tuple[int, int]],
                         loss_agg_fn: Callable[[torch.tensor], torch.tensor] = torch.mean,
                         input_tensor: torch.tensor = None,
                         optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.Adam,
                         optimizer_kwargs: Optional[Dict[str, Any]] = None,
                         optimization_steps: int = 20
                         ) -> torch.tensor:
    """
    Optimizes a tensor to minimize the given loss.

    Parameters:
        model: model to use
        prototypes_list: prototypes to optimize the activation of. List of pairs (class index, prototype index)
        loss_agg_fn: loss aggregation function that calculates loss from model's masked output
        input_tensor: starting point for the optimization
        optimizer_cls: optimizer class
        optimizer_kwargs: arguments for the optimizer
        optimization_steps: number of steps to optimize for
    Returns:
        Optimized tensor
    """
    size = (3, model.img_size, model.img_size)
    input_tensor = input_tensor if input_tensor is not None else torch.zeros(size=size).uniform_()
    model = prepare_model_for_prototype_optimization(model)
    output_mask = get_output_mask_from_prototypes_list(model, prototypes_list)
    optimizer_kwargs = optimizer_kwargs if optimizer_kwargs is not None else {'lr': 0.01}
    optimized_input = optimize_model(model=model,
                                     output_mask=output_mask,
                                     loss_agg_fn=loss_agg_fn,
                                     input_tensor=input_tensor,
                                     optimizer_cls=optimizer_cls,
                                     optimizer_kwargs=optimizer_kwargs,
                                     optimization_steps=optimization_steps)
    return optimized_input
