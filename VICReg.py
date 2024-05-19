import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from lightly.utils.dist import gather

class SupervisedVICRegLoss(nn.Module):
    """Implementation of the Supervised VICReg loss with class discriminative term."""

    def __init__(self, lambda_param: float = 25.0, mu_param: float = 25.0, 
                 nu_param: float = 1.0, alpha_param: float = 50.0, gather_distributed: bool = False, eps=0.0001):
        """
        Args:
            lambda_param (float): Scaling coefficient for the invariance term.
            mu_param (float): Scaling coefficient for the variance term.
            nu_param (float): Scaling coefficient for the covariance term.
            alpha_param (float): Scaling coefficient for the class discriminative term.
            gather_distributed (bool): If True then gather tensors from all GPUs.
            eps (float): Epsilon for numerical stability in variance computation.
        """
        super(SupervisedVICRegLoss, self).__init__()
        self.lambda_param = lambda_param
        self.mu_param = mu_param
        self.nu_param = nu_param
        self.alpha_param = alpha_param
        self.gather_distributed = gather_distributed
        self.eps = eps

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.gather_distributed and dist.is_initialized():
            world_size = dist.get_world_size()
            z_a = torch.cat(gather(z_a), dim=0)
            z_b = torch.cat(gather(z_b), dim=0)
            labels = torch.cat(gather(labels), dim=0)

        inv_loss = invariance_loss(z_a, z_b)
        var_loss = 0.5 * (variance_loss(z_a, self.eps, labels) + variance_loss(z_b, self.eps, labels))
        cov_loss = 0.5 * (covariance_loss(z_a, labels) + covariance_loss(z_b, labels))
        class_loss = class_discriminative_loss(z_a, z_b, labels, self.alpha_param)

        total_loss = (self.lambda_param * inv_loss + self.mu_param * var_loss +
                      self.nu_param * cov_loss + self.alpha_param * class_loss)
        return total_loss

def invariance_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(x, y)

def variance_loss(x: torch.Tensor, eps: float, labels: torch.Tensor) -> torch.Tensor:
    # Compute variance for each class separately and average
    unique_labels = torch.unique(labels)
    var_losses = []
    for label in unique_labels:
        x_class = x[labels == label]
        std = torch.sqrt(x_class.var(dim=0) + eps)
        var_losses.append(torch.mean(F.relu(1.0 - std)))
    return torch.mean(torch.stack(var_losses))

def covariance_loss(x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    unique_labels = torch.unique(labels)
    cov_losses = []
    for label in unique_labels:
        x_class = x[labels == label]
        x_class_mean = x_class - x_class.mean(dim=0)
        cov = torch.einsum('bi,bj->ij', x_class_mean, x_class_mean) / (x_class.size(0) - 1)
        nondiag_mask = ~torch.eye(cov.size(0), device=x.device, dtype=torch.bool)
        cov_losses.append(cov[nondiag_mask].pow(2).sum() / cov.size(0))
    return torch.mean(torch.stack(cov_losses))

def class_discriminative_loss(z_a: torch.Tensor, z_b: torch.Tensor, labels: torch.Tensor, alpha: float) -> torch.Tensor:
    # Calculate mean embedding for each class and enforce separation
    unique_labels = torch.unique(labels)
    means = {label: (z_a[labels == label].mean(dim=0) + z_b[labels == label].mean(dim=0)) / 2
             for label in unique_labels}
    losses = []
    for i, label_i in enumerate(unique_labels):
        for label_j in unique_labels[i+1:]:
            mean_dist = F.relu(alpha - torch.norm(means[label_i] - means[label_j]))
            losses.append(mean_dist.pow(2))
    return torch.mean(torch.stack(losses)) if losses else torch.tensor(0.0)