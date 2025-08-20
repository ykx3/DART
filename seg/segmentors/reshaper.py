import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple
import sys
import os
sys.path.append(os.path.join(__file__[:-11], '../../../../DART'))
# Import custom modules and functions
from dart.tools import *
from dart.spn import *
# ===================================================================
# 1. DynamicAdaptiveImageReshaper (Enhanced to return boundary information)
# ===================================================================
import torchvision.models as models

class DynamicAdaptiveImageReshaper(nn.Module):
    """
    A dynamic image transformation module that computes adaptive shapes.
    
    This module strictly follows the user-specified SPN and PDF calculation logic,
    ensuring consistent input and output shapes. It performs dynamic image reshaping
    based on importance scores predicted by a Score Prediction Network (SPN).
    
    Attributes:
        spn (nn.Module): Pre-defined ScorePredNet instance
        patch_size (int): Size of each patch
        spn_input_size (tuple): Fixed input image size expected by SPN
        spn_grid_size (tuple): Grid size for SPN operations
    """
    
    def __init__(self, spn: nn.Module, patch_size: int = 16, spn_input_size: tuple = (224, 224)):
        """
        Initialize the DynamicAdaptiveImageReshaper module.
        
        Args:
            spn (nn.Module): Pre-defined ScorePredNet instance
            patch_size (int): Size of each patch
            spn_input_size (tuple): Fixed input image size expected by SPN
        """
        super().__init__()
        self.spn = spn
        self.patch_size = patch_size
        self.spn_input_size = to_2tuple(spn_input_size)
        
        self.spn_grid_size = (sz // patch_size for sz in self.spn_input_size)

    def forward(self, x: torch.Tensor, ret_dict: bool = False, vis: bool = False):
        """
        Forward pass for dynamic image reshaping.
        
        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W)
            ret_dict (bool): If True, return dictionary containing intermediate values
            vis (bool): If True, generate grid segmentation visualization for first image

        Returns:
            torch.Tensor or dict: Transformed image, or dictionary with intermediate results
        """
        B, C, H, W = x.shape
        ret = {}  # Dictionary for storing intermediate results

        # 1. Dynamically calculate actual input grid dimensions
        assert H % self.patch_size == 0 and W % self.patch_size == 0, \
            f"Input image dimensions ({H}x{W}) must be divisible by patch_size ({self.patch_size})."
        grid_h = H // self.patch_size
        grid_w = W // self.patch_size
        num_patches = grid_h * grid_w
        ret['shape'] = (grid_h, grid_w)
        
        # --------------------------------------------------------------------
        # 2. Predict importance scores (PDF)
        # --------------------------------------------------------------------
        score = self.spn(x, shape=ret['shape'])  # Score length is fixed (e.g., B, 196)
        
        # If SPN output length doesn't match actual patch count, align via interpolation
        if score.shape[1] != num_patches:
            spn_h, spn_w = self.spn_grid_size
            score = F.interpolate(
                score.view(B, 1, spn_h, spn_w).float(),  # Reshape to (B, 1, H_spn, W_spn)
                size=(grid_h, grid_w),  # Interpolate to actual grid size
                mode='bilinear',
                align_corners=False
            )
            score = score.view(B, num_patches)  # Flatten back to (B, num_patches)

        # Use specified normalization method
        pdf = score / score.sum(dim=-1, keepdim=True)
        ret['pdf'] = pdf
        ret['score'] = score
        
        # --------------------------------------------------------------------
        # 3. Calculate dynamic row heights and column widths
        # --------------------------------------------------------------------
        pdf_2d = pdf.view(B, grid_h, grid_w)
        
        # Calculate row heights
        pdf_rows = pdf_2d.sum(dim=2)  # Sum along width dimension to get row importance (B, grid_h)
        row_heights = pdf_to_row_heights(pdf_rows, total_height=H, target_h=grid_h, version='r')
        ret['row_heights'] = row_heights
        
        # Calculate column widths (via transpose)
        pdf_cols = pdf_2d.permute(0, 2, 1).sum(dim=2)  # Transpose then sum along old height dimension (B, grid_w)
        col_widths = pdf_to_row_heights(pdf_cols, total_height=W, target_h=grid_w, version='r')
        ret['col_heights'] = col_widths

        # 4. Build sampling boundaries (edges) based on column widths
        col_pos = torch.cumsum(col_widths, dim=1)
        new_edges = torch.cat([torch.ones_like(col_pos[:,:1])]+[col_pos+W*i for i in range(grid_h)], dim=1)
        # Scale edges to the image's dimensions
        new_edges = new_edges * x.size(2)*col_widths.size(1) / new_edges[0,-1].item()
        ret['new_edges'] = new_edges

        # 5. Dynamic image patch sampling
        patches = dynamic_image_patch_sample(x, row_heights, new_edges, 
                                             shape=(self.patch_size, self.patch_size))
        
        # 6. Reconstruct image from patches
        x_out = unpatchify(patches, self.patch_size, shape=(grid_h, grid_w))
        ret['x'] = x_out

        # Ensure output and input shapes are exactly the same
        assert x_out.shape == x.shape, \
            f"Output shape {x_out.shape} does not match input shape {x.shape}!"

        # ======================== Add visualization part ========================
        if vis and B > 0:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
            from matplotlib.patches import Rectangle
            
            # Ensure all tensors are on CPU
            device = x.device
            img = x[0].detach().to('cpu')  # Move to CPU
            
            # Denormalize
            if hasattr(self, 'img_norm_cfg'):
                mean = torch.tensor(self.img_norm_cfg['mean']).view(3, 1, 1)
                std = torch.tensor(self.img_norm_cfg['std']).view(3, 1, 1)
                img = img * std + mean
            else:
                img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + \
                       torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            img = torch.clamp(img, 0, 1)
            img_np = img.permute(1, 2, 0).numpy()
            
            # Prepare grid segmentation visualization
            row_heights_cpu = row_heights[0].detach().to('cpu')
            col_widths_cpu = col_widths[0].detach().to('cpu')
            
            cum_row = torch.cat([torch.zeros(1), torch.cumsum(row_heights_cpu, dim=0)])
            cum_col = torch.cat([torch.zeros(1), torch.cumsum(col_widths_cpu, dim=0)])
            
            # Create canvas (split into two subplots)
            fig = plt.figure(figsize=(16, 8))
            gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
            
            # Subplot 1: Dynamic grid visualization
            ax1 = plt.subplot(gs[0])
            ax1.imshow(img_np)
            ax1.set_title("Dynamic Grid Segmentation")
            
            # Draw grid lines
            for h in cum_row:
                ax1.axhline(h.item(), color='red', linestyle='-', linewidth=1)
            for w in cum_col:
                ax1.axvline(w.item(), color='blue', linestyle='-', linewidth=1)
            
            # Subplot 2: Importance score heatmap
            ax2 = plt.subplot(gs[1])
            # Extract scores for the first image and adjust dimensions
            score_map = score[0].view(grid_h, grid_w).detach().cpu().numpy()
            im = ax2.imshow(score_map, cmap='viridis', aspect='equal')
            
            # Add cell values and borders
            for i in range(grid_h):
                for j in range(grid_w):
                    val = score_map[i, j]
                    # Select text color based on brightness
                    text_color = 'white' if val < np.max(score_map)*0.5 else 'black'
                    ax2.text(j, i, f'{val:.2f}', 
                             ha='center', va='center', 
                             color=text_color, fontsize=8)
                    # Add cell border
                    rect = Rectangle((j-0.5, i-0.5), 1, 1, 
                                     fill=False, edgecolor='white', linewidth=0.5)
                    ax2.add_patch(rect)
            
            # Add colorbar
            plt.colorbar(im, ax=ax2, label='Importance Score')
            ax2.set_title("Importance Scores per Patch")
            ax2.set_xlabel("Column Index")
            ax2.set_ylabel("Row Index")
            
            # Add grid labels
            ax2.set_xticks(np.arange(grid_w))
            ax2.set_yticks(np.arange(grid_h))
            ax2.set_xticklabels([f'{w:.1f}' for w in col_widths_cpu], rotation=90)
            ax2.set_yticklabels([f'{h:.1f}' for h in row_heights_cpu])
            
            plt.tight_layout()
            plt.savefig("visualization_results.png", dpi=200, bbox_inches='tight')
            plt.close(fig)  # Close figure to avoid memory leaks
        
        # ======================== Visualization part ends ========================

        return x_out if not ret_dict else ret
# ===================================================================
# 2. ReshaperWithInverse wrapper class
# ===================================================================

class ReshaperWithInverse(nn.Module):
    """
    Wrapper class that provides both forward and inverse transformation capabilities.
    
    This class combines the DynamicAdaptiveImageReshaper with inverse transformation
    functionality, making it suitable for segmentation tasks where both forward
    transformation (for training) and inverse transformation (for inference) are needed.
    
    Attributes:
        reshaper (DynamicAdaptiveImageReshaper): The main reshaping module
        patch_size (int): Size of each patch
    """
    
    def __init__(self, patch_size=16):
        """
        Initialize the ReshaperWithInverse wrapper.
        
        Args:
            patch_size (int): Size of each patch. Default: 16
        """
        super().__init__()
        self.reshaper = DynamicAdaptiveImageReshaper(spn=EfficientNetB0Pred())
        self.patch_size = self.reshaper.patch_size

    def forward_transform(self, img, gt_mask=None):
        """
        Perform forward transformation on input image and optional ground truth mask.
        
        Args:
            img (torch.Tensor): Input image of shape (B, C, H, W)
            gt_mask (torch.Tensor, optional): Ground truth mask of shape (B, H, W) or (B, 1, H, W)
        
        Returns:
            tuple: (warped_img, warped_gt_mask, transform_params)
                - warped_img (torch.Tensor): Transformed image
                - warped_gt_mask (torch.Tensor or None): Transformed ground truth mask
                - transform_params (dict): Transformation parameters for inverse transform
        """
        B, _, H, W = img.shape
        ret_dict = self.reshaper(img, ret_dict=True)
        warped_img = ret_dict['x']
        
        transform_params = {
            'row_heights': ret_dict['row_heights'],
            'col_heights': ret_dict['col_heights'],
            'original_shape': (H, W),
            'target_shape': tuple(warped_img.shape[2:])
        }
        
        warped_gt_mask = None
        if gt_mask is not None:
            if gt_mask.ndim == 3:
                gt_mask = gt_mask.unsqueeze(1)
            
            # Mask is long type, cast to float for sampling
            patches = dynamic_image_patch_sample(
                gt_mask.float(), 
                transform_params['row_heights'], 
                ret_dict['new_edges'], 
                shape=(self.patch_size, self.patch_size),
                mode='nearest'  # Use nearest neighbor for masks
            )
            warped_gt_mask = unpatchify(patches, self.patch_size, shape=ret_dict['shape'])
            warped_gt_mask = warped_gt_mask.squeeze(1).long()  # Back to (B, H, W) and long type

        return warped_img, warped_gt_mask, transform_params

    def inverse_transform(self, warped_pred, transform_params):
        """
        Perform inverse transformation to restore predictions to original image space.
        
        Args:
            warped_pred (torch.Tensor): Predictions in warped space of shape (B, NumClasses, H_warped, W_warped)
            transform_params (dict): Transformation parameters from forward_transform
        
        Returns:
            torch.Tensor: Predictions in original image space of shape (B, NumClasses, H_orig, W_orig)
        """
        B, NumClasses, H_warped, W_warped = warped_pred.shape
        H_orig, W_orig = transform_params['original_shape']
        device = warped_pred.device
        
        row_heights = transform_params['row_heights']
        col_heights = transform_params['col_heights']
        
        num_rows, num_cols = row_heights.shape[1], col_heights.shape[1]

        # Create original image coordinate grid [H_orig, W_orig, 2]
        y_coords_orig = torch.linspace(0, H_orig-1, H_orig, device=device, dtype=torch.float32)
        x_coords_orig = torch.linspace(0, W_orig-1, W_orig, device=device, dtype=torch.float32)
        grid_y_orig, grid_x_orig = torch.meshgrid(y_coords_orig, x_coords_orig, indexing='ij')
        points_orig = torch.stack((grid_x_orig, grid_y_orig), dim=-1)  # [H_orig, W_orig, 2]
        
        # Initialize tensor to store original prediction results
        original_pred = torch.zeros(B, NumClasses, H_orig, W_orig, device=device)
        
        for b in range(B):
            # Get transformation parameters for current batch
            row_heights_b = row_heights[b]  # [num_rows]
            col_heights_b = col_heights[b]  # [num_cols]
            
            # Use transform_points to calculate warped image coordinates
            warped_points = transform_points(
                points_orig,
                row_heights_b,
                col_heights_b,
                original_shape=(H_orig, W_orig),
                target_grid_shape=(num_rows, num_cols),
                inverse=False  # Forward transformation: original coordinates → warped coordinates
            )  # [H_orig, W_orig, 2]
            
            # Normalize to [-1, 1] range
            norm_warped_x = (warped_points[..., 0] / (W_warped - 1)) * 2 - 1
            norm_warped_y = (warped_points[..., 1] / (H_warped - 1)) * 2 - 1
            inverse_grid = torch.stack((norm_warped_x, norm_warped_y), dim=-1)  # [H_orig, W_orig, 2]
            
            # Perform grid_sample
            original_pred_b = F.grid_sample(
                warped_pred[b:b+1], 
                inverse_grid.unsqueeze(0),
                mode='bilinear',
                padding_mode='border',
                align_corners=True
            )
            original_pred[b] = original_pred_b[0]
        
        return original_pred

