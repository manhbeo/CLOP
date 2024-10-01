import torch.nn as nn
import torch
from lightly.utils import dist
import torch.nn.functional as F

class Supervised_NTXentLoss(nn.Module):
    def __init__(self, temperature: float = 0.5, label_fraction: float = 1.0, gather_distributed: bool = False):
        '''
        Parameters:
        - temperature (float): Temperature for the ntx-ent loss.
        - label_fraction (float): Fraction of labels to use for supervised contrastive loss.
        - gather_distributed (bool): Whether to gather data across multiple distributed processes for multi-GPU training.
        '''
        super(Supervised_NTXentLoss, self).__init__()
        self.temperature = temperature
        self.use_distributed = gather_distributed and dist.world_size() > 1
        self.label_fraction = label_fraction 

    def forward(self, out0: torch.Tensor, out1: torch.Tensor, labels: torch.Tensor):
        '''
        Parameters:
        - out0 (torch.Tensor): Embeddings from one view, shape (batch_size, embedding_dim).
        - out1 (torch.Tensor): Embeddings from another view, shape (batch_size, embedding_dim).
        - labels (torch.Tensor): Class labels corresponding to each embedding, shape (batch_size,).

        Returns:
        - torch.Tensor: The computed loss.
        '''
        out0 = F.normalize(out0, dim=1)
        out1 = F.normalize(out1, dim=1)

        if self.use_distributed:
            out0_large = torch.cat(dist.gather(out0), 0)
            out1_large = torch.cat(dist.gather(out1), 0)
            labels_large = torch.cat(dist.gather(labels), 0)
        else:
            out0_large = out0
            out1_large = out1
            labels_large = labels

        # Randomly mask out labels based on the label fraction
        if self.label_fraction < 1.0:
            num_samples = len(labels_large)
            mask = torch.rand(num_samples) < self.label_fraction
            labels_large = labels_large.clone()  
            labels_large[~mask] = -1  # Set unused labels to -1

        # Compute the cosine similarity matrix
        logits = torch.matmul(out0_large, out1_large.T) / self.temperature
        logits_exp = torch.exp(logits)

        positive_mask_labeled = (labels_large.unsqueeze(1) == labels_large.unsqueeze(0)) & (labels_large.unsqueeze(1) != -1)
        identity_mask = torch.eye(logits.size(0), device=logits.device).bool()  # Ensure self-pairs for same samples
        positive_mask_unlabeled = (labels_large.unsqueeze(1) == -1) & identity_mask
        positive_mask = positive_mask_labeled | positive_mask_unlabeled

        positive_logits = logits_exp * positive_mask
        all_logits_sum = logits_exp.sum(dim=1, keepdim=True) + 1e-8

        # Avoid divide-by-zero 
        positive_sum = positive_logits.sum(dim=1)
        valid_positive_mask = positive_sum > 0
        sup_contrastive_loss = -torch.log(positive_sum[valid_positive_mask] / all_logits_sum[valid_positive_mask].squeeze()).mean()

        return sup_contrastive_loss
