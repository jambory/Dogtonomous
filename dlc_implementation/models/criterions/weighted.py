import torch
import torch.nn as nn
import torch.nn.functional as F

from dlc_implementation.models.criterions.base import (
    BaseCriterion,
)

class WeightedCriterion(BaseCriterion):
    """Base class for weighted criterions"""

    def __init__(self, criterion: nn.Module):
        super().__init__()
        self.criterion = criterion

    def forward(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor | float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            output: predicted tensor
            target: target tensor
            weights: weights for each element in the loss calculation. If a float,
                weights all elements by that value. Defaults to 1.

        Returns:
            the weighted loss
        """
        # shape of loss: (batch_size, n_kpts, heatmap_size, heatmap_size)
        loss = self.criterion(output, target)
        n_elems = count_nonzero_elems(loss, weights)
        if n_elems == 0:
            n_elems = 1

        return torch.sum(loss * weights) / n_elems

class WeightedMSECriterion(WeightedCriterion):
    """
    Weighted Mean Squared Error (MSE) Loss.

    This loss computes the Mean Squared Error between the prediction and target tensors,
    but it also incorporates weights to adjust the contribution of each element in the loss
    calculation. The loss is computed element-wise, and elements with a weight of 0 (masked items)
    are excluded from the loss calculation.
    """

    def __init__(self) -> None:
        super().__init__(nn.MSELoss(reduction="none"))

    def forward(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor | float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            output: predicted tensor
            target: target tensor
            weights: weights for each element in the loss calculation. If a float,
                weights all elements by that value. Defaults to 1.

        Returns:
            the weighted loss
        """
        # shape of loss: (batch_size, n_kpts, h, w)
        loss = self.criterion(output, target)
        n_elems = count_nonzero_elems(loss, weights)
        if n_elems == 0:
            n_elems = 1

        return torch.sum(loss * weights) / n_elems

class WeightedHuberCriterion(WeightedCriterion):
    """
    Weighted Huber Loss.

    This loss computes the Huber loss between the prediction and target tensors,
    but it also incorporates weights to adjust the contribution of each element in the loss
    calculation. The loss is computed element-wise, and elements with a weight of 0 are
    excluded from the loss calculation.
    """

    def __init__(self) -> None:
        super().__init__(nn.HuberLoss(reduction="none"))

def count_nonzero_elems(
    losses: torch.Tensor, weights: float | torch.Tensor, per_batch: bool = False
):
    """
    Compute the number of elements in the loss function induced by `weights`.
    This is a torch implementation of https://github.com/tensorflow/tensorflow/blob/4dacf3f368eb7965e9b5c3bbdd5193986081c3b2/tensorflow/python/ops/losses/losses_impl.py#L89

    Args:
        losses (Tensor): Tensor of shape [batch_size, d1, ... dN].
        weights (Tensor): Tensor of shape [], [batch_size] or [batch_size, d1, ... dK], where K < N.
        per_batch (bool): Whether to return the number of elements per batch or as a sum total.

    Returns:
        Tensor: The number of present (non-zero) elements in the losses tensor.
    """
    if isinstance(weights, float):
        if weights != 0.0:
            return losses.numel()
        else:
            return torch.tensor(0)

    weights = torch.as_tensor(weights, dtype=torch.float32)

    # Check for non-zero weights and broadcast to match losses
    present = torch.where(
        weights == 0.0, torch.zeros_like(weights), torch.ones_like(weights)
    )
    present = present.expand_as(losses)

    # Reduce sum across the desired dimensions
    if per_batch:
        reduction_dims = tuple(range(1, present.dim()))
        return torch.sum(present, dim=reduction_dims, keepdim=True)
    else:
        return torch.sum(present)