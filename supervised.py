import torch.nn as nn
import torch
from lightly.utils import dist
import torch.nn.functional as F

class Supervised_NTXentLoss(nn.Module):
    def __init__(self, temperature: float = 0.5, label_fraction: float = 1.0, gather_distributed: bool = False):
        super(Supervised_NTXentLoss, self).__init__()
        self.temperature = temperature
        self.use_distributed = gather_distributed and dist.world_size() > 1
        self.label_fraction = label_fraction  # Fraction of labels to use for supervised contrastive loss

    def forward(self, out0: torch.Tensor, out1: torch.Tensor, labels: torch.Tensor):
        out0 = F.normalize(out0, dim=1)
        out1 = F.normalize(out1, dim=1)

        if self.use_distributed:
            # Gather hidden representations and labels from other processes
            out0_large = torch.cat(dist.gather(out0), 0)
            out1_large = torch.cat(dist.gather(out1), 0)
            labels_large = torch.cat(dist.gather(labels), 0)
        else:
            # Use local data in a single process
            out0_large = out0
            out1_large = out1
            labels_large = labels

        # Randomly mask out labels based on the label fraction
        if self.label_fraction < 1.0:
            # Create a mask that randomly selects a fraction of the labels
            num_samples = len(labels_large)
            mask = torch.rand(num_samples) < self.label_fraction
            labels_large = labels_large.clone()  # Clone to avoid modifying the original tensor
            labels_large[~mask] = -1  # Set unused labels to -1

        # Compute the cosine similarity matrix
        logits = torch.matmul(out0_large, out1_large.T) / self.temperature
        logits_exp = torch.exp(logits)

        # Construct a mask for positive samples (label == label, excluding masked labels)
        positive_mask = (labels_large.unsqueeze(1) == labels_large.unsqueeze(0)) & (labels_large.unsqueeze(1) != -1)

        # Calculate the logits for positive samples
        positive_logits = logits_exp * positive_mask

        # Calculate the sum of the exponentiated logits for all samples (the denominator in the softmax)
        all_logits_sum = logits_exp.sum(dim=1, keepdim=True)

        # Avoid divide-by-zero by ensuring at least one positive sample per instance
        positive_sum = positive_logits.sum(dim=1)
        valid_positive_mask = positive_sum > 0
        sup_contrastive_loss = -torch.log(positive_sum[valid_positive_mask] / all_logits_sum[valid_positive_mask].squeeze()).mean()

        return sup_contrastive_loss
