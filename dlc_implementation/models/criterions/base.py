from abc import ABC, abstractmethod

import torch
import torch.nn as nn

class BaseCriterion(ABC, nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(
        self, output: torch.Tensor, target: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """
        Args:
            output: the output from which to compute the loss
            target: the target for the loss

        Returns:
            the different losses for the module, including one "total_loss" key which
            is the loss from which to start backpropagation
        """
        raise NotImplementedError


class BaseLossAggregator(ABC, nn.Module):
    @abstractmethod
    def forward(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            losses: the losses to aggregate

        Returns:
            the aggregate loss
        """
        raise NotImplementedError

class WeightedLossAggregator(BaseLossAggregator):
    def __init__(self, weights: dict[str, float]) -> None:
        super().__init__()
        self.weights = weights

    def forward(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        weighted_losses = [
            weight * losses[loss_name] for loss_name, weight in self.weights.items()
        ]
        return torch.mean(torch.stack(weighted_losses))
