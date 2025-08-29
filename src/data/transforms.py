import torch
import numpy as np
from typing import Dict, Any
from monai.transforms import Transform
from skimage.morphology import skeletonize, dilation


class SkeletonTransform(Transform):
    """
    Calculates the skeleton of the segmentation (plus an optional 2 px tube around it)
    for skeleton recall loss training.
    """
    
    def __init__(self, do_tube: bool = True, keys=("label",)):
        """
        Args:
            do_tube: If True, applies dilation to create a 2-pixel tube around skeleton
            keys: Keys of the corresponding items to be transformed
        """
        super().__init__()
        self.do_tube = do_tube
        self.keys = keys
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        
        for key in self.keys:
            if key in d:
                seg = d[key]
                
                # Convert to numpy if tensor
                if isinstance(seg, torch.Tensor):
                    seg_np = seg.cpu().numpy()
                    was_tensor = True
                else:
                    seg_np = seg
                    was_tensor = False
                
                # Calculate skeleton
                skel = self._compute_skeleton(seg_np)
                
                # Convert back to tensor if input was tensor
                if was_tensor:
                    skel = torch.from_numpy(skel).to(seg.device)
                
                # Add skeleton to data dictionary
                skel_key = key.replace("label", "skeleton") if "label" in key else f"{key}_skeleton"
                d[skel_key] = skel
        
        return d
    
    def _compute_skeleton(self, seg: np.ndarray) -> np.ndarray:
        """
        Compute skeleton from segmentation mask
        
        Args:
            seg: Segmentation array, shape (C, H, W, D) or (H, W, D)
        
        Returns:
            Skeleton array with same shape as input
        """
        if seg.ndim == 4:  # (C, H, W, D)
            # Process only the foreground channel (assuming channel 1)
            if seg.shape[0] > 1:
                bin_seg = seg[1] > 0  # Take foreground channel
            else:
                bin_seg = seg[0] > 0
        elif seg.ndim == 3:  # (H, W, D) 
            bin_seg = seg > 0
        else:
            raise ValueError(f"Unsupported segmentation shape: {seg.shape}")
        
        # Initialize skeleton array
        skel_array = np.zeros_like(seg, dtype=np.int16)
        
        if bin_seg.sum() > 0:  # Only process if there's foreground
            # For 3D data, we need to skeletonize slice by slice or use 3D skeletonization
            if bin_seg.ndim == 3:  # 3D volume
                # Simple approach: skeletonize slice by slice
                skel_3d = np.zeros_like(bin_seg, dtype=bool)
                for z in range(bin_seg.shape[2]):
                    if bin_seg[:, :, z].sum() > 0:
                        skel_slice = skeletonize(bin_seg[:, :, z])
                        skel_3d[:, :, z] = skel_slice
                skel = skel_3d
            else:  # 2D
                skel = skeletonize(bin_seg)
            
            # Convert to int and apply tube dilation if requested
            skel = (skel > 0).astype(np.int16)
            
            if self.do_tube:
                # Apply double dilation for 2-pixel tube
                skel = dilation(dilation(skel))
            
            # Mask with original segmentation to preserve label values
            if seg.ndim == 4:
                if seg.shape[0] > 1:
                    skel = skel * seg[1].astype(np.int16)
                    skel_array[1] = skel  # Put in foreground channel
                else:
                    skel = skel * seg[0].astype(np.int16) 
                    skel_array[0] = skel
            else:
                skel = skel * seg.astype(np.int16)
                skel_array = skel
        
        return skel_array


class AddSkeletonToDatad(Transform):
    """
    Dictionary-based version of SkeletonTransform for MONAI data loading pipeline
    """
    
    def __init__(self, keys, do_tube: bool = True, allow_missing_keys: bool = False):
        super().__init__()
        self.keys = keys
        self.do_tube = do_tube
        self.allow_missing_keys = allow_missing_keys
        self.skeleton_transform = SkeletonTransform(do_tube=do_tube)
    
    def __call__(self, data):
        d = dict(data)
        
        for key in self.keys:
            if key in d:
                # Create skeleton key name
                skel_key = key.replace("label", "skeleton") if "label" in key else f"{key}_skeleton"
                
                # Apply skeleton transform
                temp_data = {key: d[key]}
                temp_result = self.skeleton_transform(temp_data)
                
                # Extract skeleton result
                for result_key in temp_result:
                    if "skeleton" in result_key:
                        d[skel_key] = temp_result[result_key]
                        break
            elif not self.allow_missing_keys:
                raise KeyError(f"Key '{key}' not found in data")
        
        return d 