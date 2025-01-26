import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional

class CLOPLoss(nn.Module):
    '''
    Orthogonal Prototype Loss with SVD-initialized anchors. Add this to another loss function ("ntx_ent" or "supcon")
    '''

    def __init__(self, num_classes: int = 100, embedding_dim: int = 128, lambda_value: float = 1.0,
                 distance: str = "cosine", etf=False):
        '''
        Parameters:
            num_classes (int): Number of classes, and thus the number of anchors.
            embedding_dim (int): Dimensionality of the embeddings and each anchor.
            lambda_value (float): Scaling factor for the loss.
            distance (str): The type of distance metric to use ("cosine", "euclidean", and "manhattan")
        '''
        super(CLOPLoss, self).__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.lambda_value = lambda_value
        self.distance = distance

        if etf:
            I = torch.eye(embedding_dim)
            offset = torch.ones((embedding_dim, embedding_dim)) / embedding_dim
            etf_mat = I - offset
            self.anchors = nn.Parameter(etf_mat[:num_classes, :], requires_grad=False)
        else:
            # Initialize anchors using SVD
            random_matrix = torch.randn(num_classes, embedding_dim)
            _, _, V = torch.svd(random_matrix)
            self.anchors = nn.Parameter(V[:, :num_classes].t(), requires_grad=False)

        if dist.is_initialized():
            dist.broadcast(self.anchors, 0)

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor, labels: Optional[torch.Tensor]):
        '''
        Compute the Orthogonal Prototype Loss, skipping samples with label -1.

        Parameters:
            z_i (torch.Tensor): Batch of embeddings from one view, shape (batch_size, embedding_dim).
            z_j (torch.Tensor): Batch of embeddings from another view, shape (batch_size, embedding_dim).
            labels (torch.Tensor): Corresponding labels for each embedding, shape (batch_size,).
                                 Samples with label -1 will be skipped.

        Returns:
            torch.Tensor: The computed Orthogonal Prototype Loss, or zero tensor if no valid labels.
        '''
        # If labels is None, return zero loss
        if labels is None:
            return torch.tensor(0.0, device=z_i.device)

        # Create mask for valid labels (not equal to -1)
        valid_mask = labels != -1
        if not valid_mask.any():
            return torch.tensor(0.0, device=z_i.device)

        # Normalize embeddings
        z_i = nn.functional.normalize(z_i, dim=1)
        z_j = nn.functional.normalize(z_j, dim=1)

        # Select only samples with valid labels
        z_i_valid = z_i[valid_mask]
        z_j_valid = z_j[valid_mask]
        labels_valid = labels[valid_mask]

        # Get corresponding anchors for valid labels
        anchors_valid = self.anchors[labels_valid]

        # Compute loss based on selected distance metric
        if self.distance == "cosine":
            cosine_similarity = torch.sum(z_i_valid * anchors_valid, dim=1) + \
                              torch.sum(z_j_valid * anchors_valid, dim=1)
            cosine_similarity /= 2
            loss = torch.mean(1 - cosine_similarity)

        elif self.distance == "euclidean":
            euclidean_distance = torch.norm(z_i_valid - anchors_valid, p=2, dim=1) + \
                               torch.norm(z_j_valid - anchors_valid, p=2, dim=1)
            euclidean_distance /= 2
            loss = torch.mean(euclidean_distance)

        elif self.distance == "manhattan":
            manhattan_distance = torch.sum(torch.abs(z_i_valid - anchors_valid), dim=1) + \
                               torch.sum(torch.abs(z_j_valid - anchors_valid), dim=1)
            manhattan_distance /= 2
            loss = torch.mean(manhattan_distance)

        loss *= self.lambda_value

        # Use distributed reduce to average the loss across all processes
        if dist.is_initialized():
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss /= dist.get_world_size()

        return loss
