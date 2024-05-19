import torch
import torch.distributed as dist
import torch.nn.functional as F
from typing import Tuple

class SupBarlowTwinsLoss(torch.nn.Module):
    """
    Supervised version of the Barlow Twins Loss.

    This version uses a dynamic class center for embeddings to encourage class-specific features
    while reducing inter-class redundancy in a distributed training environment.
    """

    def __init__(self, num_classes: int, feature_dim: int, lambda_param: float = 5e-3, gather_distributed: bool = False):
        """
        Initialize the Supervised Barlow Twins Loss module.
        
        Args:
            num_classes (int): Number of classes in the dataset.
            feature_dim (int): Dimensionality of the feature vectors.
            lambda_param (float): Importance of redundancy reduction term. Defaults to 5e-3.
            gather_distributed (bool): If True, gather and sum matrices across all GPUs.
        """
        super(SupBarlowTwinsLoss, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.lambda_param = lambda_param
        self.gather_distributed = gather_distributed

        self.register_buffer('class_centers', torch.zeros(num_classes, feature_dim))

        if gather_distributed and not dist.is_available():
            raise ValueError(
                "gather_distributed is True but torch.distributed is not available. "
                "Please set gather_distributed=False or install a torch version with "
                "distributed support."
            )

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the supervised Barlow Twins Loss.

        Args:
            z (torch.Tensor): Embeddings of the batch. Shape (batch_size, feature_dim).
            labels (torch.Tensor): Ground truth labels for the batch. Shape (batch_size,).

        Returns:
            torch.Tensor: Computed loss for the batch.
        """
        # Update class centers
        self._update_class_centers(z, labels)

        # Normalize embeddings
        z_norm = F.normalize(z, dim=1)

        # Normalize class centers
        centers_norm = F.normalize(self.class_centers, dim=1)

        # Compute cross-correlation matrix between normalized embeddings and class centers
        c = z_norm @ centers_norm.T
        c /= z.size(0)

        # Gather cross-correlation matrix if distributed
        if self.gather_distributed and dist.is_initialized():
            dist.all_reduce(c)

        # Calculate losses
        invariance_loss = torch.diagonal(c).add_(-1).pow_(2).sum()
        redundancy_reduction_loss = self._off_diagonal(c).pow_(2).sum()
        loss = invariance_loss + self.lambda_param * redundancy_reduction_loss

        return loss

    def _update_class_centers(self, z: torch.Tensor, labels: torch.Tensor):
        """
        Update class centers based on current batch embeddings.

        Args:
            z (torch.Tensor): Current batch embeddings.
            labels (torch.Tensor): Labels corresponding to each embedding.
        """
        for i in range(self.num_classes):
            mask = labels == i
            if mask.any():
                self.class_centers[i] = z[mask].mean(dim=0)

    def _off_diagonal(self, x):
        """
        Return a flattened view of the off-diagonal elements of a square matrix.

        Args:
            x (torch.Tensor): Square matrix.

        Returns:
            torch.Tensor: Flattened view of off-diagonal elements.
        """
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()