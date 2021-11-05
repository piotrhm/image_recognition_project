from typing import List, Tuple, Callable, Type, Dict, Any, Optional

import torch
import torch.nn as nn

from src.optimization.optimize import optimize_model
from .utils import prepare_model_for_prototype_optimization, get_output_mask_from_prototypes_list


def visualize_prototypes(model: nn.Module,
                         prototypes_list: List[Tuple[int, int]],
                         loss_agg_fn: Callable[[torch.tensor], torch.tensor] = torch.mean,
                         input_init_fn: Callable[[torch.tensor], None] = torch.nn.init.uniform_,
                         optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.Adam,
                         optimizer_kwargs: Optional[Dict[str, Any]] = None,
                         optimization_steps: int = 20
                         ) -> torch.tensor:
    model = prepare_model_for_prototype_optimization(model)
    output_mask = get_output_mask_from_prototypes_list(model, prototypes_list)
    optimizer_kwargs = optimizer_kwargs if optimizer_kwargs is not None else {'lr': 0.01}
    optimized_input = optimize_model(model=model,
                                     output_mask=output_mask,
                                     loss_agg_fn=loss_agg_fn,
                                     input_init_fn=input_init_fn,
                                     optimizer_cls=optimizer_cls,
                                     optimizer_kwargs=optimizer_kwargs,
                                     optimization_steps=optimization_steps)
    return optimized_input
