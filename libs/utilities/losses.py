# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from libs.utilities.utils import one_hot


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        n, ch, x, y, z = inputs.size()

        logpt = -self.criterion(inputs, targets.long())
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * logpt
        return loss.mean()


class tversky_loss(nn.Module):
    """
        Calculates the Tversky loss of the Foreground categories.
        if alpha == 1 --> Dice score
        alpha: controls the penalty for false positives.
        beta: controls the penalty for false negatives.
    """
    def __init__(self, alpha, eps=1e-5):
        super(tversky_loss, self).__init__()
        self.alpha = alpha
        self.beta = 2 - alpha
        self.eps = eps

    def forward(self, inputs, targets):
        # inputs.shape[1]: predicted categories
        targets = one_hot(targets, inputs.shape[1])
        inputs = F.softmax(inputs, dim=1)

        dims = tuple(range(2, targets.ndimension()))
        tps = torch.sum(inputs * targets, dims)
        fps = torch.sum(inputs * (1 - targets), dims) * self.alpha
        fns = torch.sum((1 - inputs) * targets, dims) * self.beta
        loss = (2 * tps) / (2 * tps + fps + fns + self.eps)
        loss = torch.mean(loss, dim=0)
        return 1 - (loss[1:]).mean()


class segmentation_loss(nn.Module):
    def __init__(self, alpha):
        super(segmentation_loss, self).__init__()
        self.dice = tversky_loss(alpha=alpha, eps=1e-5)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        dice = self.dice(inputs, targets.contiguous())
        ce = self.ce(inputs, targets)
        return dice + ce


class Dice_metric(nn.Module):
    def __init__(self, eps=1e-5):
        super(Dice_metric, self).__init__()
        self.eps = eps

    def forward(self, inputs, targets, logits=True):
        categories = inputs.shape[1]
        targets = targets.contiguous()
        targets = one_hot(targets, categories)
        if logits:
            inputs = torch.argmax(F.softmax(inputs, dim=1), dim=1)
        inputs = one_hot(inputs, categories)

        dims = tuple(range(2, targets.ndimension()))
        tps = torch.sum(inputs * targets, dims)
        fps = torch.sum(inputs * (1 - targets), dims)
        fns = torch.sum((1 - inputs) * targets, dims)
        loss = (2 * tps) / (2 * tps + fps + fns + self.eps)
        return loss[:, 1:].mean(dim=1)