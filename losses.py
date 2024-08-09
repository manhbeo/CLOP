import torch
import torch.nn as nn
import torch.distributed as dist

class OARLoss(nn.Module):
    """
    Orthogonal Anchor Regression Loss with SVD-initialized anchors. Add this to a (main) loss function
    """
    def __init__(self, num_classes: int = 100, embedding_dim: int = 128):
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
        
        # Broadcast anchors to ensure all processes have the same anchors
        if dist.is_initialized():
            dist.broadcast(self.anchors, 0)

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
        loss = torch.mean(loss)
        
        # Use distributed reduce to average the loss across all processes
        if dist.is_initialized():
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss /= dist.get_world_size()
        
        return loss