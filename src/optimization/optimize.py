from typing import Type, Dict, Any, Optional, Union, List, Tuple

import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, SequentialLR
from torchvision.transforms import Compose

from src.optimization import AggregationFn
from src.optimization.transforms import ReversibleTransform, ReversibleCompose, Transform


def optimize_model(model: nn.Module,
                   prototypes_mask: Tensor,
                   input_tensor: Tensor,
                   loss_agg_fn: AggregationFn,
                   optimizer_cls: Type[Optimizer],
                   optimizer_parameters: Tensor,
                   optimizer_kwargs: Dict[str, Any],
                   optimization_steps: int,
                   lr_scheduler_cls: _LRScheduler,
                   lr_scheduler_kwargs: Dict[str, Any],
                   lr_scheduler_step_interval: int,
                   transforms: Optional[List[Transform]],
                   robustness_transforms: Optional[List[Union[ReversibleTransform, Transform]]],
                   parametrization_transforms: Optional[List[Transform]],
                   gradient_transforms: Optional[List[Transform]],
                   denormalization_transforms: Optional[List[Transform]],
                   reverse_reversible_robustness_transforms: bool,
                   print_interval: Optional[int],
                   display_interval: Optional[int],
                   return_optimized_input: bool = False,
                   ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """
    Optimizes a tensor to minimize loss given by loss_agg_fn.

    Parameters:
        model: model to use
        prototypes_mask: boolean tensor that filters only desired elements from model's output
        input_tensor: an initial tensor
        loss_agg_fn: AggregationFn, outputs aggregated loss
        optimizer_cls: optimizer class
        optimizer_parameters: what to optimize
        optimizer_kwargs: arguments for the optimizer
        optimization_steps: number of steps to optimize for
        lr_scheduler_cls: lr scheduler class
        lr_scheduler_kwargs: arguments for the lr scheduler
        lr_scheduler_step_interval: make lr scheduler step every `lr_scheduler_step_interval` steps
        transforms: list of transformations that are applied to the input before applying robustness
            transformations. The transformations are applied as in-place operations.
        robustness_transforms: list of transformations that are applied to the input and may be reversed
            at the end of the epoch. The transformations are applied as in-place operations.
        parametrization_transforms: list of differentiable transformations that are applied to the (possibly previously
            transformed) input. Applying these transformations should end up with a tensor that can be fed to the model:
            it must have appropriate size and should be a vector from standardized (normalized) ImageNet distribution.
            The transformations are applied as gradient-flow operations.
        gradient_transforms: list of transformations that are applied to the input gradient after the gradient
            backpropagation, but before the optimizer update. The transformations are applied as in-place operations.
        denormalization_transforms: list of transformations that are applied to the parametrized input to obtain
            an image with all pixel values in [0, 1]. It should always be set to [``DenormalizationMeanStd()``].
        reverse_reversible_robustness_transforms: whether to reverse all reversible transformations used as robustness
            transformations
        print_interval: prints logs every `print_interval` steps
        display_interval: displays optimized image every `display_interval` steps
        return_optimized_input: whether to additionally return optimized input.
    Returns:
        Optimized image and optimized input (optionally).
    """
    transforms = transforms if transforms else []
    robustness_transforms = robustness_transforms if robustness_transforms else []
    parametrization_transforms = parametrization_transforms if parametrization_transforms else []
    gradient_transforms = gradient_transforms if gradient_transforms else []
    denormalization_transforms = denormalization_transforms if denormalization_transforms else []
    transform_fn = Compose(transforms)
    robustness_transform_fn = ReversibleCompose(robustness_transforms)
    parametrization_transform_fn = Compose(parametrization_transforms)
    gradient_transform_fn = Compose(gradient_transforms)
    denormalization_transform_fn = Compose(denormalization_transforms)

    @torch.no_grad()
    def _transform_to_image(x: Tensor) -> Tensor:
        x = parametrization_transform_fn(x)
        x = x.detach().cpu().clone()
        x.data = transform_fn(x.data)
        x = denormalization_transform_fn(x)
        return x

    input_tensor = input_tensor.to(next(model.parameters()).device)
    input_tensor.requires_grad_()
    optimizer = optimizer_cls(params=optimizer_parameters, **optimizer_kwargs)
    if lr_scheduler_cls is not None:
        lr_scheduler = lr_scheduler_cls(optimizer, **lr_scheduler_kwargs)
    for i in range(optimization_steps):
        optimizer.zero_grad()
        parametrized_input = parametrization_transform_fn(input_tensor)
        parametrized_input.data = transform_fn(parametrized_input.data)
        parametrized_input.data = robustness_transform_fn(parametrized_input.data)
        loss = loss_agg_fn(model, parametrized_input.unsqueeze(0), prototypes_mask.unsqueeze(0))
        loss.backward()
        parametrized_input.grad = gradient_transform_fn(parametrized_input.grad)
        optimizer.step()
        if reverse_reversible_robustness_transforms:
            parametrized_input.data = robustness_transform_fn.reverse_transform(parametrized_input.data)

        if print_interval and i % print_interval == 0:
            if lr_scheduler_cls is not None:
                print(f'step: {i}/{optimization_steps}, loss: {loss}, lr: {lr_scheduler.get_last_lr()[0]}')
            else:
                print(f'step: {i}/{optimization_steps}, loss: {loss}')

        if display_interval and i % display_interval == 0:
            output_image = _transform_to_image(input_tensor)
            display(F.to_pil_image(output_image))
        if lr_scheduler_cls is not None and i % lr_scheduler_step_interval == 0:
            lr_scheduler.step()

    image = _transform_to_image(input_tensor)
    if return_optimized_input:
        return image, input_tensor.detach().cpu()
    else:
        return image
