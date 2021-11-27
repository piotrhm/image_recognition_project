from typing import Callable, Type, Dict, Any, Optional

import torch
import torch.nn as nn
import torchvision.transforms.functional as F


def optimize_model(model: nn.Module,
                   output_mask: torch.tensor,
                   input_tensor: torch.tensor,
                   loss_agg_fn: Callable[[torch.tensor, torch.tensor], torch.tensor],
                   optimization_direction: str,
                   optimizer_cls: Type[torch.optim.Optimizer],
                   optimizer_kwargs: Dict[str, Any],
                   optimization_steps: int,
                   before_optim_step: Callable[[torch.tensor], None],
                   print_interval: int,
                   display_interval: Optional[int]
                   ) -> torch.tensor:
    """
    Optimizes a tensor to minimize loss given by loss_agg_fn.

    Parameters:
        model: model to use
        output_mask: boolean tensor that filters only desired elements from model's output
        input_tensor: an initial tensor
        loss_agg_fn: takes input_tensor and model's masked output, outputs aggregated loss
        optimization_direction: direction of optimization. Can be `minimize` or 'maximize'
        optimizer_cls: optimizer class
        optimizer_kwargs: arguments for the optimizer
        optimization_steps: number of steps to optimize for
        before_optim_step: called after gradients are calculated, but before optimizer step
        print_interval: prints logs every `print_interval` steps
        display_interval: displays `input_tensor` every `display_interval` steps
    Returns:
        optimized tensor
    """
    assert optimization_direction in ['maximize', 'minimize']

    input_tensor = input_tensor.to(next(model.parameters()).device)
    input_tensor.requires_grad_()
    optimizer = optimizer_cls(params=[input_tensor], **optimizer_kwargs)
    for i in range(optimization_steps):
        optimizer.zero_grad()
        output = model(input_tensor.unsqueeze(0))
        interesting_output = output[output_mask.unsqueeze(0)]
        loss = loss_agg_fn(input_tensor, interesting_output)
        loss = loss if optimization_direction == 'minimize' else -loss
        if i % print_interval == 0:
            print(f'step: {i}/{optimization_steps}, loss: {loss}')
        if display_interval and i % display_interval == 0:
            display(F.to_pil_image(input_tensor))
        loss.backward()
        if before_optim_step:
            before_optim_step(input_tensor)
        optimizer.step()

    return input_tensor.detach()
