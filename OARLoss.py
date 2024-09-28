import torch
import torch.nn as nn
import torch.distributed as dist
import random

class OARLoss(nn.Module):
    """
    Orthogonal Anchor Regression Loss with SVD-initialized anchors. Add this to a (main) loss function
    """
    def __init__(self, num_classes: int = 100, embedding_dim: int = 128, lambda_value: float = 1.0, distance: str = "cosine", label_por=1.0):
        """
        Args:
            num_classes (int): Number of classes, and thus the number of anchors.
            embedding_dim (int): Dimensionality of the embeddings and each anchor.
        """
        super(OARLoss, self).__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.lambda_value = lambda_value
        self.distance = distance
        self.label_por = label_por
        
        # Initialize anchors using SVD to ensure they are orthogonal
        random_matrix = torch.randn(num_classes, embedding_dim)
        _, _, V = torch.svd(random_matrix)
        self.anchors = nn.Parameter(V[:, :num_classes].t(), requires_grad=False)  # Use the right singular vectors
        
        # Broadcast anchors to ensure all processes have the same anchors
        if dist.is_initialized():
            dist.broadcast(self.anchors, 0)

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor, z_weak: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute the Orthogonal Anchor Regression Loss using a certain percentage of labels.

        Args:
            z_i (torch.Tensor): Batch of embeddings, shape (batch_size, embedding_dim)
            z_j (torch.Tensor): Batch of embeddings, shape (batch_size, embedding_dim)
            z_weak (torch.Tensor): Batch of weak embeddings, shape (batch_size, embedding_dim)
            labels (torch.Tensor): Corresponding labels for each embedding, shape (batch_size,) or None if loss is "supcon"
            percentage (float): Percentage of labels to use (value between 0 and 1).
        
        Returns:
            torch.Tensor: The computed loss.
        """
        # Normalize embeddings to unit vectors
        z_i = nn.functional.normalize(z_i, dim=1)
        z_j = nn.functional.normalize(z_j, dim=1)
                
        # Select a percentage of the batch
        batch_size = z_i.size(0)
        num_selected = int(batch_size * self.label_por)
        
        if num_selected < batch_size:
            # Randomly select indices of the batch
            selected_indices = random.sample(range(batch_size), num_selected)
            selected_indices = torch.tensor(selected_indices, device=labels.device)
        else:
            # Use all indices if percentage is 1.0 or higher
            selected_indices = torch.arange(batch_size, device=labels.device)

        # Gather the corresponding embeddings and labels
        z_i_selected = z_i[selected_indices]
        z_j_selected = z_j[selected_indices]
        if labels is None: 
            #TODO: Update on every 10 epoch
            z_weak = nn.functional.normalize(z_weak, dim=1)
            I = torch.diag(z_weak @ (z_i).T)

            sorted_I, _ = torch.sort(I, descending=True) 
            num_anchors = max(1, int(0.1 * len(sorted_I)))

            anchors_selected = sorted_I[:num_anchors] #TODO: size mismatch. Fix

        else:
            labels_selected = labels[selected_indices]
            # Gather the corresponding anchors for each selected label
            anchors_selected = self.anchors[labels_selected]

        # Compute cosine similarity
        if self.distance == "cosine": 
            cosine_similarity = torch.sum(z_i_selected * anchors_selected, dim=1) + torch.sum(z_j_selected * anchors_selected, dim=1)
            cosine_similarity /= 2
            # Compute the loss as the mean of (1 - cosine similarity)
            loss = torch.mean(1 - cosine_similarity)
            
        elif self.distance == "euclidean":
            # Euclidean distance
            euclidean_distance = torch.norm(z_i_selected - anchors_selected, p=2, dim=1) + torch.norm(z_j_selected - anchors_selected, p=2, dim=1)
            euclidean_distance /= 2
            # Compute the loss as the mean of the Euclidean distance
            loss = torch.mean(euclidean_distance)

        elif self.distance == "manhattan":
            # Manhattan distance
            manhattan_distance = torch.sum(torch.abs(z_i_selected - anchors_selected), dim=1) + torch.sum(torch.abs(z_j_selected - anchors_selected), dim=1)
            manhattan_distance /= 2
            # Compute the loss as the mean of the Manhattan distance
            loss = torch.mean(manhattan_distance)

        loss *= self.lambda_value
        
        # Use distributed reduce to average the loss across all processes
        if dist.is_initialized():
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss /= dist.get_world_size()
        
        return loss
