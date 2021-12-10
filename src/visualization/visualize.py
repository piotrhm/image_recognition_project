from typing import List, Tuple, Callable, Type, Dict, Any, Optional

import torch
import torch.nn as nn

from src.optimization.aggregate import AggregationFn
from src.optimization.optimize import optimize_model
from .utils import prepare_model_for_prototype_optimization, get_prototypes_mask_from_prototypes_list


def visualize_prototypes(model: nn.Module,
                         prototypes_list: List[Tuple[int, int]],
                         input_tensor: torch.tensor,
                         loss_agg_fn: AggregationFn = AggregationFn(),
                         optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.Adam,
                         optimizer_kwargs: Optional[Dict[str, Any]] = None,
                         optimization_steps: int = 20,
                         before_optim_step: Callable[[torch.tensor], None] = None,
                         print_interval: int = 100,
                         display_interval: Optional[int] = 500
                         ) -> torch.tensor:
    """
    Optimizes a tensor to minimize the given loss.

    Parameters:
        model: model to use
        prototypes_list: prototypes to optimize the activation of. List of pairs (class index, prototype index)
        input_tensor: an initial tensor
        loss_agg_fn: AggregationFn, outputs aggregated loss
        prototypes_agg_fn: takes loss_agg_fn output for each of prototype, outputs aggregated loss
        optimizer_cls: optimizer class
        optimizer_kwargs: arguments for the optimizer
        optimization_steps: number of steps to optimize for
        before_optim_step: called after gradients are calculated, but before optimizer step
        print_interval: prints logs every `print_interval` steps
        display_interval: displays input_tensor every `display_interval` steps
    Returns:
        Optimized tensor
    """
    model = prepare_model_for_prototype_optimization(model)
    prototypes_mask = get_prototypes_mask_from_prototypes_list(model, prototypes_list)
    optimizer_kwargs = optimizer_kwargs if optimizer_kwargs is not None else {'lr': 0.01}
    optimized_input = optimize_model(model=model,
                                     prototypes_mask=prototypes_mask,
                                     loss_agg_fn=loss_agg_fn,
                                     input_tensor=input_tensor,
                                     optimizer_cls=optimizer_cls,
                                     optimizer_kwargs=optimizer_kwargs,
                                     optimization_steps=optimization_steps,
                                     before_optim_step=before_optim_step,
                                     print_interval=print_interval,
                                     display_interval=display_interval)
    return optimized_input
