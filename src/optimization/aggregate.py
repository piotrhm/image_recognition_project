import torch
import torch.nn.functional as F
from typing import Union


class AggregationFn:
    def __init__(
        self,
        metric: str = "distance",
        agg_fn: str = "mean",
        ptype_lvl_agg_fn: str = "mean",
        exponent: float = 1.0,
        patches_mask: Union[torch.tensor, bool] = True,
    ):

        """
        Defines loss that will be minimized.

        Parameters:
            metric: metric used in loss. `distance` or `similarity`
            agg_fn: function that aggregates feature map for one prototype. `mean` or `mean_log`
            ptype_lvl_agg_fn: function that aggregates outputs of `agg_fn` for each prototype. `mean` or `mean_log`
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
        elif agg_fn == "mean_log":
            self.agg_fn = lambda x: torch.mean(torch.log(x), dim=-1)
        else:
            raise ValueError("Expected `mean` or `mean_log`")

        if ptype_lvl_agg_fn == "mean":
            self.ptype_lvl_agg_fn = torch.mean
        elif ptype_lvl_agg_fn == "mean_log":
            self.ptype_lvl_agg_fn = lambda x: torch.mean(torch.log(x))
        else:
            raise ValueError("Expected `mean` or `mean_log`")

        self.exponent = exponent
        self.patches_mask = patches_mask

    def to(self, device):
        """Moves object to device"""
        if type(self.patches_mask) is not bool:
            self.patches_mask = self.patches_mask.to(device)
        return self

    @staticmethod
    def diversity_term_func(x):
        batch, channels, _, _ = x.shape
        flattened = x.view(batch, channels, -1)
        grams = torch.matmul(flattened, torch.transpose(flattened, 1, 2))
        grams = F.normalize(grams, p=2, dim=(1, 2))
        return -sum([sum([(grams[i] * grams[j]).sum()
                          for j in range(batch) if j != i]) for i in range(batch)]) / batch

    def __call__(self, model, x, prototypes_mask, diversity_term=False, alfa=1.0):
        """Evaluates and returns loss"""
        metric = model.prototype_distances(x)
        if self.similarity:
            metric = -model.distance_2_similarity(metric)
        metric = metric[prototypes_mask]
        metric = metric[:, self.patches_mask]
        agg_metric = self.agg_fn(torch.pow(metric, self.exponent))
        if diversity_term:
            return self.ptype_lvl_agg_fn(agg_metric) - alfa*self.diversity_term_func(x)
        return self.ptype_lvl_agg_fn(agg_metric)
