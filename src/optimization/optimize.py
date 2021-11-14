from typing import Callable, Type, Dict, Any

import torch
import torch.nn as nn


def optimize_model(model: nn.Module,
                   output_mask: torch.tensor,
                   loss_agg_fn: Callable[[torch.tensor], torch.tensor],
                   input_tensor: torch.tensor,
                   optimizer_cls: Type[torch.optim.Optimizer],
                   optimizer_kwargs: Dict[str, Any],
                   optimization_steps: int,
                   ) -> torch.tensor:
    input_tensor = input_tensor.to(next(model.parameters()).device).float()
    input_tensor.requires_grad_()
    optimizer = optimizer_cls(params=[input_tensor], **optimizer_kwargs)
    for i in range(optimization_steps):
        optimizer.zero_grad()
        output = model(input_tensor.unsqueeze(0))  # probably should be batched or something
        interesting_output = output[output_mask.unsqueeze(0)]
        loss = loss_agg_fn(interesting_output)
        loss.backward()
        optimizer.step()

    return input_tensor.detach()
