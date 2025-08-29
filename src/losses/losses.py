from monai.losses import DiceLoss, DiceFocalLoss, DiceCELoss
from src.losses.cldice.compound_cldice_loss import DC_and_CE_and_CLDC_loss
from src.losses.cldice.cldice_loss import SoftclDiceLoss
from src.losses.skeleton_recall.compound_losses import DC_SkelREC_and_CE_loss
import torch
import warnings

class LossFactory:
    @staticmethod
    def create_loss(loss_name):
        if loss_name == "DiceLoss":
            return DiceLoss(to_onehot_y=True, softmax=True)
        elif loss_name == "DiceCELoss":
            return DiceCELoss(to_onehot_y=True, softmax=True)
        elif loss_name == "DiceFocalLoss":
            return DiceFocalLoss(to_onehot_y=True, softmax=True)
        elif loss_name == "SoftclDiceLoss":
            return SoftclDiceLoss()
        elif loss_name == "SoftDiceclDiceLoss":
            return DC_and_CE_and_CLDC_loss(
                soft_dice_kwargs=dict(batch_dice=True, smooth=1., do_bg=False, ddp=True),
                ce_kwargs=dict(weight=None, ignore_index=-100),
                cldc_kwargs=dict(iter_=10, smooth=1.),
                weight_ce=1,
                weight_dice=1,
                weight_cldice=1,
                ignore_label=None
            )
        elif loss_name == "SkeletonRecallLoss":
            return DC_SkelREC_and_CE_loss(
                soft_dice_kwargs=dict(batch_dice=True, smooth=1., do_bg=False),
                soft_skelrec_kwargs=dict(batch_dice=True, smooth=1., do_bg=False),
                ce_kwargs=dict(weight=None, ignore_index=-100),
                weight_ce=1,
                weight_dice=1,
                weight_srec=1,
                ignore_label=None
            )
        else:
            raise ValueError(f"Unsupported loss function: {loss_name}")