from typing import List, Tuple, Type, Dict, Any, Optional, Union

import numpy as np
import scipy.ndimage as nd
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from src.optimization.aggregate import AggregationFn
from src.optimization.optimize import optimize_model
from .utils import prepare_model_for_prototype_optimization, get_prototypes_mask_from_prototypes_list
from ..optimization import ReversibleTransform, Transform, DenormalizationMeanStd


def visualize_prototypes(model: nn.Module,
                         prototypes_list: List[Tuple[int, int]],
                         input_tensor: torch.tensor,
                         loss_agg_fn: AggregationFn = AggregationFn(),
                         optimizer_cls: Type[Optimizer] = torch.optim.Adam,
                         optimizer_parameters: Tensor = None,
                         optimizer_kwargs: Optional[Dict[str, Any]] = None,
                         optimization_steps: int = 20,
                         lr_scheduler_cls: Optional[_LRScheduler] = None,
                         lr_scheduler_kwargs: Optional[Dict[str, Any]] = None,
                         lr_scheduler_step_interval: int = 1,
                         transforms: Optional[List[Transform]] = None,
                         robustness_transforms: Optional[List[Union[ReversibleTransform, Transform]]] = None,
                         parametrization_transforms: Optional[List[Transform]] = None,
                         gradient_transforms: Optional[List[Transform]] = None,
                         denormalization_transforms: Optional[List[Transform]] = (DenormalizationMeanStd(),),
                         reverse_reversible_robustness_transforms: bool = True,
                         print_interval: Optional[int] = 100,
                         display_interval: Optional[int] = 500
                         ) -> Tensor:
    """
    Optimizes a tensor to minimize the given loss.

    Parameters:
        model: model to use
        prototypes_list: prototypes to optimize the activation of. List of pairs (class index, prototype index)
        input_tensor: an initial tensor
        loss_agg_fn: AggregationFn, outputs aggregated loss
        optimizer_cls: optimizer class
        optimizer_parameters: what to optimize
        optimizer_kwargs: arguments for the optimizer
        optimization_steps: number of steps to optimize for
        lr_scheduler_cls: lr scheduler class
        lr_scheduler_kwargs: arguments for the lr scheduler
        lr_scheduler_step_interval: make lr scheduler step every `lr_scheduler_step_interval` steps
        transforms: list of transformations that get composed and applied to input_tensor before processing by model
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
        display_interval: displays input_tensor every `display_interval` steps
    Returns:
        Optimized tensor
    """
    model = prepare_model_for_prototype_optimization(model)
    prototypes_mask = get_prototypes_mask_from_prototypes_list(model, prototypes_list)
    optimizer_parameters = input_tensor if optimizer_parameters is None else optimizer_parameters
    optimizer_kwargs = optimizer_kwargs if optimizer_kwargs is not None else {'lr': 0.01}
    optimized_input = optimize_model(model=model,
                                     prototypes_mask=prototypes_mask,
                                     loss_agg_fn=loss_agg_fn,
                                     input_tensor=input_tensor,
                                     optimizer_cls=optimizer_cls,
                                     optimizer_parameters=optimizer_parameters,
                                     optimizer_kwargs=optimizer_kwargs,
                                     optimization_steps=optimization_steps,
                                     lr_scheduler_cls=lr_scheduler_cls,
                                     lr_scheduler_kwargs=lr_scheduler_kwargs,
                                     lr_scheduler_step_interval=lr_scheduler_step_interval,
                                     transforms=transforms,
                                     robustness_transforms=robustness_transforms,
                                     parametrization_transforms=parametrization_transforms,
                                     gradient_transforms=gradient_transforms,
                                     denormalization_transforms=denormalization_transforms,
                                     reverse_reversible_robustness_transforms=reverse_reversible_robustness_transforms,
                                     print_interval=print_interval,
                                     display_interval=display_interval)
    return optimized_input


def visualize_prototypes_octaves(model: nn.Module,
                                 prototypes_list: List[Tuple[int, int]],
                                 input_tensor: torch.tensor,
                                 num_octaves: int = 10,
                                 octave_scale: float = 1.4,
                                 loss_agg_fn: AggregationFn = AggregationFn(),
                                 optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.Adam,
                                 optimizer_kwargs: Optional[Dict[str, Any]] = None,
                                 optimization_steps: int = 20,
                                 lr_scheduler_cls: Optional[Union[_LRScheduler, List[_LRScheduler]]] = None,
                                 lr_scheduler_kwargs: Optional[Union[Dict[str, Any],List[Dict[str, Any]]]] = None,
                                 lr_scheduler_step_interval: int = 1,
                                 transforms: Optional[List[Transform]] = None,
                                 robustness_transforms: Optional[List[Union[ReversibleTransform, Transform]]] = None,
                                 parametrization_transforms: Optional[List[Transform]] = None,
                                 gradient_transforms: Optional[List[Transform]] = None,
                                 denormalization_transforms: Optional[List[Transform]] = (DenormalizationMeanStd(),),
                                 reverse_reversible_robustness_transforms: bool = True,
                                 print_interval: int = 100,
                                 display_interval: Optional[int] = 500
                                 ) -> torch.tensor:
    """
    Optimizes a tensor to minimize the given loss.

    Parameters:
        model: model to use
        prototypes_list: prototypes to optimize the activation of. List of pairs (class index, prototype index)
        input_tensor: an initial tensor
        num_octaves: number of octaves
        octave_scale: image scale between octaves
        loss_agg_fn: takes input_tensor and model's masked output, outputs aggregated loss
        optimizer_cls: optimizer class
        optimizer_kwargs: arguments for the optimizer
        optimization_steps: number of steps to optimize for per octave
        lr_scheduler_cls: lr scheduler class list of classes for SequentialLR
        lr_scheduler_kwargs: arguments for the lr scheduler or list of kwargs for SequentialLR
        lr_scheduler_step_interval: make lr scheduler step every `lr_scheduler_step_interval` steps
        transforms: list of transformations that get composed and applied to input_tensor before processing by model
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
        print_interval: prints logs every `print_interval` steps per octave
        display_interval: displays input_tensor every `display_interval` steps per octave
    Returns:
        Optimized tensor
    """
    model = prepare_model_for_prototype_optimization(model)
    prototypes_mask = get_prototypes_mask_from_prototypes_list(model, prototypes_list)
    optimizer_kwargs = optimizer_kwargs if optimizer_kwargs is not None else {'lr': 0.01}

    input_arr = input_tensor.data.numpy()
    input_arr_scaled = [input_arr]
    for _ in range(num_octaves - 1):
        input_arr_scaled.append(nd.zoom(input_arr_scaled[-1], (1, 1 / octave_scale, 1 / octave_scale), order=1))

    detail = np.zeros_like(input_arr_scaled[-1])
    for octave, base in enumerate(input_arr_scaled[::-1]):
        if octave > 0:
            detail = nd.zoom(detail, np.array(base.shape) / np.array(detail.shape), order=1)
        image = base + detail
        input_tensor = torch.from_numpy(image)
        optimized_image, optimized_input = optimize_model(model=model,
                                                          prototypes_mask=prototypes_mask,
                                                          loss_agg_fn=loss_agg_fn,
                                                          input_tensor=input_tensor,
                                                          optimizer_cls=optimizer_cls,
                                                          optimizer_kwargs=optimizer_kwargs,
                                                          optimization_steps=optimization_steps,
                                                          lr_scheduler_cls=lr_scheduler_cls,
                                                          lr_scheduler_kwargs=lr_scheduler_kwargs,
                                                          lr_scheduler_step_interval=lr_scheduler_step_interval,
                                                          transforms=transforms,
                                                          robustness_transforms=robustness_transforms,
                                                          parametrization_transforms=parametrization_transforms,
                                                          gradient_transforms=gradient_transforms,
                                                          denormalization_transforms=denormalization_transforms,
                                                          reverse_reversible_robustness_transforms=
                                                          reverse_reversible_robustness_transforms,
                                                          print_interval=print_interval,
                                                          display_interval=display_interval,
                                                          return_optimized_input=True)
        detail = optimized_input - base

    return optimized_image
