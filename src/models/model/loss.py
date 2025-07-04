import torch.nn as nn
import torch as th
import torch.nn.functional as F
import torch
import numpy as np


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)
        N = inputs.size()[0]
        # flatten label and prediction tensors
        inputs = inputs.contiguous().view(N, -1)
        targets = targets.contiguous().view(N, -1)

        intersection = (inputs * targets).sum(1)
        dice = (2. * intersection + smooth) / (inputs.sum(1) + targets.sum(1) + smooth)

        return 1 - dice.sum() / N


class DiceLoss_v1(nn.Module):
    """docstring for DiceLoss_v1"""

    def __init__(self, weight=None, size_average=True, alpha=0.25):
        super(DiceLoss_v1, self).__init__()
        self.alpha = alpha

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (intersection + smooth) / (inputs.sum() * self.alpha + targets.sum() * (1 - self.alpha) + smooth)

        return 1 - dice


def dice_coef(y_true, y_pred):
    smooth = 1
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


class DiceLoss_LS(nn.Module):
    def __init__(self):
        super(DiceLoss_LS, self).__init__()

    def forward(self, input, target):
        N = target.size(0)
        smooth = 1

        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat
        Ls_loss = (input_flat - target_flat) * (input_flat - target_flat)

        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N + Ls_loss.sum() / N
        return loss


def dice_cal(input, target):
    N = target.size(0)
    smooth = 1

    input_flat = input.view(N, -1)
    target_flat = target.view(N, -1)

    intersection = input_flat * target_flat

    loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
    loss = 1 - loss.sum() / N + 1 / input_flat.mean(1)
    return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.15, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = th.exp(-BCE_loss)
        # print(pt.size())
        alpha = th.abs(targets - self.alpha)
        # print(pt)
        F_loss = alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            return th.mean(F_loss)
        else:
            return F_loss.sum()


class BCEFocalLoss(torch.nn.Module):
    """
    二分类的Focalloss alpha 固定
    """

    def __init__(self, gamma=2, alpha=0.25, reduction='elementwise_mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, _input, target):
        pt = torch.sigmoid(_input)
        # print(_input.size())
        # print(target.size())
        alpha = self.alpha
        loss = - alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - \
               (1 - alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss


class Dice_FocalLoss(nn.Module):
    """docstring  Dice_FocalLoss"""

    def __init__(self, alpha=0.15, gamma=2, logits=False, reduce=True, focal_loss_landa=0.5):
        super(Dice_FocalLoss, self).__init__()
        self.focal_loss_landa = focal_loss_landa
        self.dice = DiceLoss()
        self.fcl = FocalLoss()

    def forward(self, pre, true):
        loss = self.dice(pre, true) + self.focal_loss_landa * self.fcl(pre, true)
        return loss


class BCE_Dice(nn.Module):
    def __init__(self):
        super(BCE_Dice, self).__init__()
        self.dice = DiceLoss()

    def forward(self, pre, target, weight=None):
        return nn.BCEWithLogitsLoss(weight=weight)(pre, target) + self.dice(pre, target)


class BCE_Dice_v1(nn.Module):
    def __init__(self, alpha=0.1):
        super(BCE_Dice_v1, self).__init__()
        self.dice = DiceLoss_v1(alpha=alpha)

    def forward(self, pre, target, weight=None):
        return nn.BCEWithLogitsLoss(weight=weight)(pre, target) + self.dice(pre, target)

class Dist_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pre, true):
        pre = pre.view(-1)
        true = true.view(-1)
        N = true.size()[0]
        loss = torch.sum(torch.abs(pre ** 3 - true ** 3)) / N
        return loss
