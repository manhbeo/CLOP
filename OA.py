import torch
import torch.nn as nn

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
