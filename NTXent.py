import torch
import torch.nn as nn
from lightly.utils import dist
import torch.nn.functional as F

class SupTreeConLoss(nn.Module):
    def __init__(self, temperature: float = 0.5, lambda_val: float = 0.5, use_distributed: bool = False, device: torch.device = torch.device('cuda')):
        super(SupTreeConLoss, self).__init__()
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
