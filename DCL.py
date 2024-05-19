import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Callable, Optional

def supervised_negative_mises_fisher_weights(
    out0: torch.Tensor, out1: torch.Tensor, labels0: torch.Tensor, labels1: torch.Tensor, sigma: float = 0.5
) -> torch.Tensor:
    """Weighting function for supervised Decoupled Contrastive Learning,
    taking into account class labels.
    
    Args:
        out0 (torch.Tensor): Output projections of the first set.
        out1 (torch.Tensor): Output projections of the second set.
        labels0 (torch.Tensor): Labels corresponding to out0.
        labels1 (torch.Tensor): Labels corresponding to out1.
        sigma (float): Scaling factor for the similarity.

    Returns:
        torch.Tensor: Weights for the loss components.
    """
    similarity = torch.einsum('nm,nm->n', out0.detach(), out1.detach()) / sigma
    weights = 2 - out0.shape[0] * F.softmax(similarity, dim=0)
    mask = labels0 == labels1.unsqueeze(1)  # Only weight same class pairs
    return weights * mask.float()

class SupDCLLoss(nn.Module):
    """Supervised version of the Decoupled Contrastive Learning Loss."""

    def __init__(
        self,
        num_classes: int,
        temperature: float = 0.1,
        sigma: float = 0.5,
        gather_distributed: bool = False,
    ):
        super().__init__()
        self.temperature = temperature
        self.sigma = sigma
        self.num_classes = num_classes
        self.gather_distributed = gather_distributed
        self.weight_fn = partial(supervised_negative_mises_fisher_weights, sigma=sigma)

        if gather_distributed and not torch.distributed.is_available():
            raise ValueError(
                "gather_distributed is True but torch.distributed is not available. "
                "Please set gather_distributed=False or install a torch version with "
                "distributed support."
            )

    def forward(self, out0: torch.Tensor, out1: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Supervised DCL loss.

        Args:
            out0 (torch.Tensor): Projections from the first transformation.
            out1 (torch.Tensor): Projections from the second transformation.
            labels (torch.Tensor): Ground truth labels.

        Returns:
            torch.Tensor: Computed loss.
        """
        out0 = F.normalize(out0, dim=1)
        out1 = F.normalize(out1, dim=1)

        if self.gather_distributed and torch.distributed.is_initialized():
            out0_all = torch.cat(torch.distributed.all_gather(out0), 0)
            out1_all = torch.cat(torch.distributed.all_gather(out1), 0)
            labels_all = torch.cat(torch.distributed.all_gather(labels), 0)
        else:
            out0_all, out1_all, labels_all = out0, out1, labels

        loss0 = self._loss(out0, out1, out0_all, out1_all, labels, labels_all)
        loss1 = self._loss(out1, out0, out1_all, out0_all, labels, labels_all)
        return 0.5 * (loss0 + loss1)

    def _loss(self, out0, out1, out0_all, out1_all, labels0, labels_all):
        batch_size = out0.shape[0]
        sim_00 = torch.einsum("nc,mc->nm", out0, out0_all) / self.temperature
        sim_01 = torch.einsum("nc,mc->nm", out0, out1_all) / self.temperature

        # Apply label mask for positive and negative samples
        positive_mask = labels0 == labels_all.unsqueeze(0)
        negative_mask = ~positive_mask

        if self.weight_fn:
            weights = self.weight_fn(out0, out1, labels0, labels0)

        positive_loss = -sim_01[positive_mask].mean()
        negative_loss_00 = torch.logsumexp(sim_00[negative_mask], dim=1).mean()
        negative_loss_01 = torch.logsumexp(sim_01[negative_mask], dim=1).mean()

        return positive_loss + negative_loss_00 + negative_loss_01
