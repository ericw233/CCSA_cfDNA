import torch
import torch.nn as nn
import torch.nn.functional as F

class CSALoss(nn.Module):
    def __init__(self, margin=1.0):
        super(CSALoss, self).__init__()
        self.margin = margin

    def forward(self, x1, x2, class_eq):
        dist = F.pairwise_distance(x1, x2)
        loss = class_eq * dist.pow(2)
        loss += (1 - class_eq) * (self.margin - dist).clamp(min=0).pow(2)
        return loss.mean()
    
def csa_loss(x1, x2, class_eq):
    margin = 1
    dist = F.pairwise_distance(x1, x2)
    loss = class_eq * dist.pow(2)
    loss += (1 - class_eq) * (margin - dist).clamp(min=0).pow(2)
    return loss.mean()