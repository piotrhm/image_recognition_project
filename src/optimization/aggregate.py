from typing import Optional

import torch


class AggregationFn:
    def __init__(
        self,
        metric: Optional[str] = "distance",
        agg_fn: Optional[str] = "mean",
        ptype_lvl_agg_fn: Optional[str] = "mean",
        exponent: Optional[float] = 1.0,
        patches_mask: Optional[torch.tensor] = torch.ones(7, 7, dtype=bool),
    ):

    """
    Defines loss that will be minimized.

    Parameters:
        metric: metric used in loss. `distance` or `similarity`
        agg_fn: function that aggregates feature map for one prototype. `mean` or `prod`
        ptype_lvl_agg_fn: function that aggregates outputs of `agg_fn` for each prototype. `mean` or `prod`
        exponent: exponent to which each element of feature map is raised
        patches_mask: determines which patch is included in calculations, same shape as feature map
    """
        if metric == "distance":
            self.similarity = False
        elif metric == "similarity":
            self.similarity = True
        else:
            raise ValueError("Expected `distance` or `similarity`")

        if agg_fn == "mean":
            self.agg_fn = lambda x: torch.mean(x, dim=-1)
        elif agg_fn == "prod":
            self.agg_fn = lambda x: torch.prod(x, dim=-1)
        else:
            raise ValueError("Expected `mean` or `prod`")

        if ptype_lvl_agg_fn == "mean":
            self.ptype_lvl_agg_fn = torch.mean
        elif ptype_lvl_agg_fn == "prod":
            self.ptype_lvl_agg_fn = torch.prod
        else:
            raise ValueError("Expected `mean` or `prod`")

        self.exponent = exponent
        self.patches_mask = patches_mask

    def to(self, device):
        """Moves object to device"""
        self.patches_mask = self.patches_mask.to(device)
        return self

    def __call__(self, model, x, prototypes_mask):
        """Evaluates and returns loss"""
        metric = model.prototype_distances(x)
        if self.similarity:
            metric = -model.distance_2_similarity(metric)
        metric = metric[prototypes_mask]
        metric = metric[:, self.patches_mask]
        agg_metric = self.agg_fn(torch.pow(metric, self.exponent))
        return self.ptype_lvl_agg_fn(agg_metric)
