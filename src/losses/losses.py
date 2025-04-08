from monai.losses import DiceLoss, DiceFocalLoss, DiceCELoss

class LossFactory:
    @staticmethod
    def create_loss(loss_name):
        if loss_name == "DiceLoss":
            return DiceLoss(to_onehot_y=True, softmax=True)
        elif loss_name == "DiceCELoss":
            return DiceCELoss(to_onehot_y=True, softmax=True)
        elif loss_name == "DiceFocalLoss":
            return DiceFocalLoss(to_onehot_y=True, softmax=True)
        else:
            raise ValueError(f"Unsupported loss function: {loss_name}")