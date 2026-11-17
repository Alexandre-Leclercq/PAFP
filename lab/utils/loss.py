from typing import Optional
import torch
from torch.nn.modules.loss import _Loss
from torch.nn import functional as F

"""
Inspired by: https://github.com/owkin/FLamby/blob/main/flamby/datasets/fed_isic2019/loss.py
"""
class FocalLoss(_Loss):
    def __init__(self, alpha:Optional[torch.Tensor]=None, gamma:int|torch.Tensor=None):
        """

        :param alpha: the class_weight
        :param gamma: focusing parameter
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        pass

    def forward(self, inputs, targets):
        """Weighted focal loss function
        Parameters
        ----------
        inputs : torch.tensor of size 8, logits output by the model (pre-softmax)
        targets : torch.tensor of size 1, int between 0 and 7, groundtruth class
        """
        targets = targets.view(-1, 1).type_as(inputs)
        logpt = F.log_softmax(inputs, dim=1)
        logpt = logpt.gather(1, targets.long())
        logpt = logpt.view(-1)
        pt = logpt.exp()
        pt = torch.clamp(pt, 1e-7, 1.0 - 1e-7)
        if self.alpha is not None:
            self.alpha = self.alpha.to(targets.device)
            at = self.alpha.gather(0, targets.data.view(-1).long())
            logpt = logpt * at
        loss = -1 * (1 - pt) ** self.gamma * logpt

        return loss.mean()

    def __repr__(self):
        return f'FocalLoss(alpha = {self.alpha}, gamma = {self.gamma})'
