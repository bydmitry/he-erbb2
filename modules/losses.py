import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------- Focal Loss ----------------------------------
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=0, logits=True, batch_average=True):
        super(FocalLoss, self).__init__()
        self.alpha  = alpha
        self.gamma  = gamma
        self.logits = logits

        self.batch_average = batch_average

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')

        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.batch_average:
            return F_loss.mean()
        else:
            return F_loss.sum()

# ------------------------ Basic Losses ---------------------------------
def BCE_LogitsLoss():
    return nn.BCEWithLogitsLoss()

def CrossEntropy_Loss():
    return nn.CrossEntropyLoss()

def MSE_Loss():
    return nn.MSELoss()

def nll_loss(output, target):
    return F.nll_loss(output, target)