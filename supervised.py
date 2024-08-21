import torch.nn as nn
import torch
from lightly.utils import dist
import torch.nn.functional as F

class Supervised_NTXentLoss(nn.Module):
    def __init__(self, temperature: float = 0.5, gather_distributed: bool = False):
        super(Supervised_NTXentLoss, self).__init__()
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