import torch
import torch.nn as nn
from lightly.utils import dist
import torch.nn.functional as F
from lightly.utils.dist import gather
from functools import partial
from typing import Callable, Optional, Tuple

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



class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.5, lambda_val: float = 0.5, use_distributed: bool = False, device: torch.device = torch.device('cuda')):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.lambda_val = lambda_val
        self.use_distributed = use_distributed and dist.world_size() > 1
        self.cross_entropy = nn.CrossEntropyLoss()
        self.device = device

    def forward(self, out0: torch.Tensor, out1: torch.Tensor, labels: torch.Tensor, coarse_labels: torch.Tensor = None):
        out0, out1, labels = out0.to(self.device), out1.to(self.device), labels.to(self.device)
        if coarse_labels is not None:
            coarse_labels = coarse_labels.to(self.device)
        out0 = F.normalize(out0, dim=1)
        out1 = F.normalize(out1, dim=1)

        if self.use_distributed:
            # Prepare lists for gathering across nodes
            out0_gathered = [torch.empty_like(out0) for _ in range(dist.get_world_size())]
            out1_gathered = [torch.empty_like(out1) for _ in range(dist.get_world_size())]
            labels_gathered = [torch.empty_like(labels) for _ in range(dist.get_world_size())]
            coarse_labels_gathered = [torch.empty_like(coarse_labels) for _ in range(dist.get_world_size())] if coarse_labels is not None else None

            # Gather all tensors from all nodes
            dist.all_gather(out0_gathered, out0)
            dist.all_gather(out1_gathered, out1)
            dist.all_gather(labels_gathered, labels)
            if coarse_labels_gathered is not None:
                dist.all_gather(coarse_labels_gathered, coarse_labels)

            # Concatenate all gathered tensors
            out0 = torch.cat(out0_gathered, dim=0)
            out1 = torch.cat(out1_gathered, dim=0)
            labels = torch.cat(labels_gathered, dim=0)
            if coarse_labels_gathered is not None:
                coarse_labels = torch.cat(coarse_labels_gathered, dim=0)


        # Compute the cosine similarity matrix
        logits = torch.matmul(out0, out1.T) / self.temperature
        logits_exp = torch.exp(logits)
        # Construct a mask for positive samples
        positive_mask = labels.unsqueeze(1) == labels.unsqueeze(0)
        # Calculate the logits for positive samples
        positive_logits = logits_exp * positive_mask
        # Calculate the sum of the exponentiated logits for all samples (the denominator in the softmax)
        all_logits_sum = logits_exp.sum(dim=1, keepdim=True)
        sup_contrastive_loss = -torch.log(positive_logits.sum(dim=1) / all_logits_sum.squeeze()).mean()

        return sup_contrastive_loss



#TODO: add distributed computation
class OARLoss(nn.Module):
    """
    Orthogonal Anchor Regression Loss with SVD-initialized anchors.
    """
    def __init__(self, num_classes: int, embedding_dim: int):
        """
        Args:
            num_classes (int): Number of classes, and thus the number of anchors.
            embedding_dim (int): Dimensionality of the embeddings and each anchor.
        """
        super(OARLoss, self).__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        
        # Initialize anchors using SVD to ensure they are orthogonal
        random_matrix = torch.randn(num_classes, embedding_dim)
        _, _, v = torch.svd(random_matrix)
        self.anchors = nn.Parameter(v[:num_classes], requires_grad=False)  # Use the right singular vectors

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute the Orthogonal Anchor Regression Loss.

        Args:
            embeddings (torch.Tensor): Batch of embeddings, shape (batch_size, embedding_dim)
            labels (torch.Tensor): Corresponding labels for each embedding, shape (batch_size,)
        
        Returns:
            torch.Tensor: The computed loss.
        """
        # Normalize embeddings to unit vectors
        embeddings_norm = nn.functional.normalize(embeddings, p=2, dim=1)
        
        # Gather the corresponding anchors for each embedding in the batch
        anchors_selected = self.anchors[labels]
        
        # Compute cosine similarity
        cosine_similarity = torch.sum(embeddings_norm * anchors_selected, dim=1)
        
        # Compute the loss as the mean of (1 - cosine similarity)
        loss = 1 - cosine_similarity
        return torch.mean(loss)