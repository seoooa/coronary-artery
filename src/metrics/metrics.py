from monai.metrics import (
    DiceMetric, 
    HausdorffDistanceMetric, 
    MeanIoU,
    ConfusionMatrixMetric,
)
from skimage.morphology import skeletonize
from skimage.measure import label
import torch
import numpy as np

class CLDice:
    """Centerline Dice (clDice) metric implementation"""
    def __init__(self, include_background=False, reduction="mean"):
        self.include_background = include_background
        self.reduction = reduction
        self.scores = []

    def __call__(self, y_pred, y):
        if isinstance(y_pred, list):
            y_pred = torch.stack(y_pred)
        if isinstance(y, list):
            y = torch.stack(y)
            
        if not self.include_background:
            y_pred = y_pred[:, 1:]  # Remove background
            y = y[:, 1:]  # Remove background
            
        batch_size = y_pred.shape[0]
        scores = []
        
        for i in range(batch_size):
            pred = y_pred[i].cpu().numpy()
            target = y[i].cpu().numpy()
            
            score = self.compute_cldice(pred, target)
            scores.append(score)
            
        self.scores.extend(scores)
        return torch.tensor(scores)

    def compute_cldice(self, pred, target):
        """
        Compute Centerline Dice score
        Args:
            pred: (C, D, H, W) or (C, H, W)
            target: (C, D, H, W) or (C, H, W)
        """
        def cl_score(v, s):
            return np.sum(v * s) / (np.sum(s) + np.finfo(float).eps)

        # Squeeze channel dimension if it exists
        if pred.shape[0] == 1:
            pred = pred.squeeze(0)
            target = target.squeeze(0)

        if len(pred.shape) == 2:
            tprec = cl_score(pred, skeletonize(target))
            tsens = cl_score(target, skeletonize(pred))
        elif len(pred.shape) == 3:
            tprec = cl_score(pred, skeletonize(target))
            tsens = cl_score(target, skeletonize(pred))
        else:
            raise ValueError(f"Invalid shape for cl_dice: {pred.shape}")

        return 2 * tprec * tsens / (tprec + tsens + np.finfo(float).eps)

    def aggregate(self):
        if not self.scores:
            return torch.tensor(0.0)
        if self.reduction == "mean":
            return torch.tensor(sum(self.scores) / len(self.scores))
        return torch.tensor(self.scores)

    def reset(self):
        self.scores = []

class BettiNumberError:
    """Betti number error metric implementation"""
    def __init__(self, include_background=False, reduction="mean"):
        self.include_background = include_background
        self.reduction = reduction
        self.betti_0_scores = []
        self.betti_1_scores = []

    def __call__(self, y_pred, y):
        if isinstance(y_pred, list):
            y_pred = torch.stack(y_pred)
        if isinstance(y, list):
            y = torch.stack(y)
            
        if not self.include_background:
            y_pred = y_pred[:, 1:]  # Remove background
            y = y[:, 1:]  # Remove background
            
        batch_size = y_pred.shape[0]
        betti_0_batch = []
        betti_1_batch = []
        
        for i in range(batch_size):
            pred = y_pred[i].cpu().numpy()
            target = y[i].cpu().numpy()
            
            betti_0, betti_1 = self.compute_betti_error(pred, target)
            betti_0_batch.append(betti_0)
            betti_1_batch.append(betti_1)
            
        self.betti_0_scores.extend(betti_0_batch)
        self.betti_1_scores.extend(betti_1_batch)
        return torch.tensor(betti_0_batch), torch.tensor(betti_1_batch)

    def extract_labels(self, gt_array, pred_array):
        """Extract unique labels from both arrays."""
        labels = set(np.unique(gt_array)) | set(np.unique(pred_array))
        return labels

    def betti_number(self, binary_array):
        """Calculate Betti numbers (β0, β1) for a binary array."""
        # β0: number of connected components
        labeled_array = label(binary_array)
        beta0 = len(np.unique(labeled_array)) - 1  # subtract 1 for background

        # β1: number of holes (2D) or tunnels (3D)
        if len(binary_array.shape) == 2:
            # For 2D: β1 = number of holes
            beta1 = len(np.unique(label(~binary_array))) - 1
        elif len(binary_array.shape) == 3:
            # For 3D: β1 = number of tunnels (simplified)
            skeleton = skeletonize(binary_array)
            beta1 = len(np.unique(label(skeleton))) - 1
        else:
            raise ValueError(f"Invalid shape for betti_number: {binary_array.shape}")

        return [beta0, beta1]

    def compute_betti_error(self, pred, target):
        """Compute Betti number errors."""
        # Ensure binary arrays and squeeze unnecessary dimensions
        pred = (pred > 0.5).astype(np.int32)
        target = (target > 0.5).astype(np.int32)
        
        # Squeeze channel dimension if it exists
        if pred.shape[0] == 1:
            pred = pred.squeeze(0)
            target = target.squeeze(0)

        # Extract labels and verify binary segmentation
        labels = self.extract_labels(target, pred)
        labels.remove(0)
        
        if len(labels) == 0:
            return 0, 0
        
        assert len(labels) == 1 and 1 in labels, "Invalid binary segmentation"

        gt_betti_numbers = self.betti_number(target)
        pred_betti_numbers = self.betti_number(pred)
        
        betti_0_error = abs(pred_betti_numbers[0] - gt_betti_numbers[0])
        betti_1_error = abs(pred_betti_numbers[1] - gt_betti_numbers[1])
        
        return betti_0_error, betti_1_error

    def aggregate(self):
        if not self.betti_0_scores:
            return torch.tensor(0.0), torch.tensor(0.0)
        if self.reduction == "mean":
            mean_betti_0 = sum(self.betti_0_scores) / len(self.betti_0_scores)
            mean_betti_1 = sum(self.betti_1_scores) / len(self.betti_1_scores)
            return torch.tensor(mean_betti_0), torch.tensor(mean_betti_1)
        return torch.tensor(self.betti_0_scores), torch.tensor(self.betti_1_scores)

    def reset(self):
        self.betti_0_scores = []
        self.betti_1_scores = []

class MetricFactory:
    @staticmethod
    def create_metrics():
        """Create all metrics and return them in a dictionary."""
        metrics = {
            "dice": DiceMetric(
                include_background=False, 
                reduction="mean", 
                get_not_nans=False
            ),
            "hausdorff": HausdorffDistanceMetric(
                include_background=False, 
                percentile=95, 
                reduction="mean"
            ),
            "iou": MeanIoU(
                include_background=False, 
                reduction="mean"
            ),
            "precision": ConfusionMatrixMetric(
                include_background=False,
                metric_name="precision",
                compute_sample=False,
                reduction="mean"
            ),
            "recall": ConfusionMatrixMetric(
                include_background=False,
                metric_name="recall",
                compute_sample=False,
                reduction="mean"
            ),
            "cldice": CLDice(
                include_background=False,
                reduction="mean"
            ),
            "betti": BettiNumberError(
                include_background=False,
                reduction="mean"
            )
        }
        return metrics

    @staticmethod
    def calculate_metrics(metrics_dict, outputs, labels):
        """Calculate metrics for a batch of predictions and labels."""
        for metric in metrics_dict.values():
            metric(y_pred=outputs, y=labels)

    @staticmethod
    def aggregate_metrics(metrics_dict):
        """Aggregate results for all metrics."""
        results = {}
        for name, metric in metrics_dict.items():
            if name == "betti":
                betti_0, betti_1 = metric.aggregate()
                results["betti_0"] = float(betti_0.item() if torch.is_tensor(betti_0) else betti_0)
                results["betti_1"] = float(betti_1.item() if torch.is_tensor(betti_1) else betti_1)
            else:
                result = metric.aggregate()
                if isinstance(result, (list, tuple)):
                    results[name] = float(result[0] if len(result) > 0 else 0.0)
                else:
                    results[name] = float(result.item() if torch.is_tensor(result) else result)
        return results

    @staticmethod
    def reset_metrics(metrics_dict):
        """Reset all metrics."""
        for metric in metrics_dict.values():
            metric.reset()