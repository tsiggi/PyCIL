import torch
from torch import nn
from torch.nn import functional as F

class OCCELoss(nn.Module):
    def __init__(self):
        super(OCCELoss, self).__init__()
 
    def forward(self, inputs, targets):
        N = inputs.shape[1]
        # multiply with N-1 for numerical stability, does not affect gradient
        ycomp = (N - 1) * F.softmax(-inputs, dim=1)
        y = torch.ones((targets.size(0), N), device=inputs.device)
        y.scatter_(1, targets.unsqueeze(1), 0.0)
        loss = - 1 / (N - 1) * torch.sum(y * torch.log(ycomp + 0.0000001), dim=1)
 
        return torch.mean(loss)