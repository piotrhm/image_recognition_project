from typing import Callable, Type, Dict, Any, Optional

import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from torch import Tensor
from torch.optim import Optimizer


def optimize_model(model: nn.Module,
                   prototypes_mask: Tensor,
                   input_tensor: Tensor,
                   loss_agg_fn: Callable[[Tensor, Tensor], Tensor],
                   optimizer_cls: Type[Optimizer],
                   optimizer_kwargs: Dict[str, Any],
                   optimization_steps: int,
                   before_optim_step: Callable[[Tensor], None],
                   print_interval: int,
                   display_interval: Optional[int]
                   ) -> torch.tensor:
    """
    Optimizes a tensor to minimize loss given by loss_agg_fn.

    Parameters:
        model: model to use
        prototypes_mask: boolean tensor that filters only desired elements from model's output
        input_tensor: an initial tensor
        loss_agg_fn: AggregationFn, outputs aggregated loss
        optimizer_cls: optimizer class
        optimizer_kwargs: arguments for the optimizer
        optimization_steps: number of steps to optimize for
        before_optim_step: called after gradients are calculated, but before optimizer step
        print_interval: prints logs every `print_interval` steps
        display_interval: displays `input_tensor` every `display_interval` steps
    Returns:
        optimized tensor
    """

    input_tensor = input_tensor.to(next(model.parameters()).device)
    input_tensor.requires_grad_()
    optimizer = optimizer_cls(params=[input_tensor], **optimizer_kwargs)
    for i in range(optimization_steps):
        optimizer.zero_grad()
        loss = loss_agg_fn(model, input_tensor.unsqueeze(0), prototypes_mask.unsqueeze(0))
        if i % print_interval == 0:
            print(f'step: {i}/{optimization_steps}, loss: {loss}')
        if display_interval and i % display_interval == 0:
            display(F.to_pil_image(input_tensor))
        loss.backward()
        if before_optim_step:
            before_optim_step(input_tensor)
        optimizer.step()

    return input_tensor.detach()
