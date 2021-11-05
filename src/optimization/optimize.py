from typing import Callable, Type, Dict, Any

import torch
import torch.nn as nn


def optimize_model(model: nn.Module,
                   output_mask: torch.tensor,
                   loss_agg_fn: Callable[[torch.tensor], torch.tensor],
                   input_init_fn: Callable[[torch.tensor], None],
                   optimizer_cls: Type[torch.optim.Optimizer],
                   optimizer_kwargs: Dict[str, Any],
                   optimization_steps: int,
                   ) -> torch.tensor:
    size = ()  # TODO: set size
    input_tensor = torch.zeros(size=size, requires_grad=True).float().to(model.device)
    input_init_fn(input_tensor)
    optimizer = optimizer_cls(params=[input_tensor], **optimizer_kwargs)
    for i in range(optimization_steps):
        optimizer.zero_grad()
        output = model(input_tensor)  # probably should be batched or something
        interesting_output = output.data[output_mask]
        loss = loss_agg_fn(interesting_output)
        loss.backward()
        optimizer.step()

    return input_tensor.detach()
