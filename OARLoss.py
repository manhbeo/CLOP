import torch
import torch.nn as nn
import torch.distributed as dist

class OARLoss(nn.Module):
    """
    Orthogonal Anchor Regression Loss with SVD-initialized anchors. Add this to a (main) loss function
    """
    def __init__(self, num_classes: int = 100, embedding_dim: int = 128, lambda_value: float = 1.0):
        """
        Args:
            num_classes (int): Number of classes, and thus the number of anchors.
            embedding_dim (int): Dimensionality of the embeddings and each anchor.
        """
        super(OARLoss, self).__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.lambda_value = lambda_value
        
        # Initialize anchors using SVD to ensure they are orthogonal
        random_matrix = torch.randn(num_classes, embedding_dim)
        _, _, V = torch.svd(random_matrix)
        self.anchors = nn.Parameter(V[:, :num_classes].t(), requires_grad=False)  # Use the right singular vectors
        
        # Broadcast anchors to ensure all processes have the same anchors
        if dist.is_initialized():
            dist.broadcast(self.anchors, 0)

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute the Orthogonal Anchor Regression Loss.

        Args:
            embeddings (torch.Tensor): Batch of embeddings, shape (batch_size, embedding_dim)
            labels (torch.Tensor): Corresponding labels for each embedding, shape (batch_size,)
        
        Returns:
            torch.Tensor: The computed loss.
        """
        # Normalize embeddings to unit vectors
        z_i = nn.functional.normalize(z_i, p=2, dim=1)
        z_j = nn.functional.normalize(z_j, p=2, dim=1)
        
        # Gather the corresponding anchors for each embedding in the batch
        anchors_selected = self.anchors[labels]
        
        # Compute cosine similarity
        cosine_similarity = torch.sum(z_i * anchors_selected, dim=1) + torch.sum(z_j * anchors_selected, dim=1)
        cosine_similarity /= 2
        
        # Compute the loss as the mean of (1 - cosine similarity)
        loss = torch.mean(1 - cosine_similarity)
        loss *= self.lambda_value
        
        # Use distributed reduce to average the loss across all processes
        if dist.is_initialized():
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss /= dist.get_world_size()
        
        return loss