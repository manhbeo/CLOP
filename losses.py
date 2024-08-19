import torch
import torch.nn as nn
from lightly.utils import dist
import torch.nn.functional as F
from typing import Callable, Optional

class NTXentLoss(nn.Module):
    def __init__(self, temperature: float = 0.5, gather_distributed: bool = False):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.use_distributed = gather_distributed and dist.world_size() > 1
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, out0: torch.Tensor, out1: torch.Tensor, labels: torch.Tensor, coarse_labels: torch.Tensor = None):
        out0 = F.normalize(out0, dim=1)
        out1 = F.normalize(out1, dim=1)

        if self.use_distributed and dist.world_size() > 1:
            # Gather hidden representations from other processes
            out0_large = torch.cat(dist.gather(out0), 0)
            out1_large = torch.cat(dist.gather(out1), 0)
            labels_large = torch.cat(dist.gather(labels), 0)
            if coarse_labels is not None:
                coarse_labels_large = torch.cat(dist.gather(coarse_labels), 0)
                
        else:
            # Use the current process's data if not using distributed mode or single process
            out0_large = out0
            out1_large = out1
            labels_large = labels
            coarse_labels_large = coarse_labels

        # Use the gathered data for further processing
        out0, out1, labels = out0_large, out1_large, labels_large
        if coarse_labels is not None:
            coarse_labels = coarse_labels_large


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



class BarlowTwinsLoss(nn.Module):
    def __init__(self, lambda_param: float = 5e-3, gather_distributed: bool = False):
        super(BarlowTwinsLoss, self).__init__()
        self.lambda_param = lambda_param
        self.use_distributed = gather_distributed and dist.world_size() > 1
        self.epsilon = 1e-10  # Small epsilon term for numerical stability

    def off_diagonal(self, x):
        # Returns the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, z_a, z_b, labels):
        # Normalize the representations with epsilon for numerical stability
        z_a = (z_a - z_a.mean(0)) / (z_a.std(0) + self.epsilon)
        z_b = (z_b - z_b.mean(0)) / (z_b.std(0) + self.epsilon)
        
        # Gather the representations from all distributed processes if enabled
        if self.use_distributed:
            z_a = self.gather_from_all(z_a)
            z_b = self.gather_from_all(z_b)
            labels = self.gather_from_all(labels)

        # Compute the cross-correlation matrix
        c = torch.mm(z_a.T, z_b) / z_a.size(0)

        # Apply the supervised component: Only consider same-label pairs
        label_mask = labels.unsqueeze(1) == labels.unsqueeze(0)
        c = c * label_mask.float()

        # Compute the loss
        on_diag = torch.diagonal(c).add_(-1).pow(2).sum()
        off_diag = self.off_diagonal(c).pow(2).sum()
        loss = on_diag + self.lambda_param * off_diag
        return loss

    def gather_from_all(self, tensor):
        # Gather tensor from all processes
        gathered_tensors = [torch.zeros_like(tensor) for _ in range(dist.world_size())]
        dist.gather(gathered_tensors, tensor)
        gathered_tensors = torch.cat(gathered_tensors, dim=0)
        return gathered_tensors



class DCLLoss(nn.Module):
    def __init__(
        self,
        temperature: float = 0.1,
        weight_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        gather_distributed: bool = False,
    ):
        super().__init__()
        self.temperature = temperature
        self.weight_fn = weight_fn
        self.gather_distributed = gather_distributed

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.gather_distributed and dist.is_initialized():
            # Gather features and labels from all processes
            features_gathered = [torch.zeros_like(features) for _ in range(dist.world_size())]
            labels_gathered = [torch.zeros_like(labels) for _ in range(dist.world_size())]
            
            dist.all_gather(features_gathered, features)
            dist.all_gather(labels_gathered, labels)
            
            features = torch.cat(features_gathered, dim=0)
            labels = torch.cat(labels_gathered, dim=0)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Apply weight function if provided
        if self.weight_fn is not None:
            weights = self.weight_fn(features, labels)
            similarity_matrix = similarity_matrix * weights
        
        # Create mask to exclude self-comparisons
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float()

        # Compute the log_prob using logsumexp for numerical stability
        log_prob = similarity_matrix - torch.logsumexp(similarity_matrix, dim=1, keepdim=True)
        
        # Only consider positive pairs for the loss
        loss = -torch.sum(mask * log_prob, dim=1) / (torch.sum(mask, dim=1) + 1e-8)
        loss = loss.mean()

        return loss




class VICRegLoss(nn.Module):
    def __init__(
        self,
        lambda_param: float = 25.0,
        mu_param: float = 25.0,
        nu_param: float = 1.0,
        gather_distributed: bool = False,
        eps: float = 0.0001,
    ):
        super(VICRegLoss, self).__init__()
        self.lambda_param = lambda_param
        self.mu_param = mu_param
        self.nu_param = nu_param
        self.gather_distributed = gather_distributed
        self.eps = eps

    def forward(self, z_a, z_b):
        if self.gather_distributed and dist.is_initialized():
            z_a = self.gather_embeddings(z_a)
            z_b = self.gather_embeddings(z_b)

        # Invariance loss (mean squared error between representations)
        invariance_loss = nn.functional.mse_loss(z_a, z_b)

        # Variance loss (standard deviation should be close to 1)
        std_z_a = torch.sqrt(z_a.var(dim=0) + self.eps)
        std_z_b = torch.sqrt(z_b.var(dim=0) + self.eps)
        variance_loss = torch.mean(nn.functional.relu(1 - std_z_a)) + torch.mean(nn.functional.relu(1 - std_z_b))

        # Covariance loss (off-diagonal elements of covariance matrix should be small)
        N, D = z_a.size()
        z_a_centered = z_a - z_a.mean(dim=0)
        z_b_centered = z_b - z_b.mean(dim=0)

        cov_z_a = (z_a_centered.T @ z_a_centered) / (N - 1)
        cov_z_b = (z_b_centered.T @ z_b_centered) / (N - 1)

        # Mask the diagonal to zero out the diagonal elements
        cov_loss_a = self.off_diagonal(cov_z_a).pow(2).sum().div(D)
        cov_loss_b = self.off_diagonal(cov_z_b).pow(2).sum().div(D)
        covariance_loss = cov_loss_a + cov_loss_b

        # Total loss
        loss = (
            self.lambda_param * invariance_loss +
            self.mu_param * variance_loss +
            self.nu_param * covariance_loss
        )
        
        return loss

    def off_diagonal(self, matrix):
        n, m = matrix.shape
        assert n == m
        off_diag_elements = matrix.flatten()[1:].view(n - 1, n + 1)[:, :-1].flatten()
        return off_diag_elements

    def gather_embeddings(self, embeddings):
        gathered_embeddings = [torch.zeros_like(embeddings) for _ in range(dist.world_size())]
        dist.all_gather(gathered_embeddings, embeddings)
        gathered_embeddings = torch.cat(gathered_embeddings, dim=0)
        return gathered_embeddings

