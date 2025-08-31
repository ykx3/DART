import torch
import os
import torch.nn.functional as F
import torch.nn as nn
import importlib.util
import math
import matplotlib.pyplot as plt
import numpy as np


class MLP(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) with GELU activation.
    
    This module implements a two-layer feedforward neural network with
    a GELU activation function between the layers.
    
    Args:
        input_dim (int): Dimension of input features. Default: 1
        hidden_dim (int): Dimension of hidden layer. Default: 16
        output_dim (int): Dimension of output features. Default: 1
    
    Attributes:
        net (nn.Sequential): Sequential container of linear layers and activation
    """
    
    def __init__(self, input_dim=1, hidden_dim=16, output_dim=1):
        """
        Initialize the MLP with specified dimensions.
        
        Args:
            input_dim (int): Input feature dimension
            hidden_dim (int): Hidden layer dimension
            output_dim (int): Output feature dimension
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        """
        Forward pass through the MLP.
        
        Args:
            x (torch.Tensor): Input tensor of shape (n, seqlen, input_dim)
        
        Returns:
            torch.Tensor: Output tensor of shape (n, seqlen, output_dim)
        """
        out = self.net(x)  # (n, seqlen, output_dim)
        return out

def dynamic_image_patch_sample(images, row_heights, new_edges, shape=(16,16), visualize=False, version='v2', mode='bilinear'):
    """
    Dynamically sample image patches based on variable row heights and edge positions.
    
    This function performs adaptive patch sampling from images by first resampling
    the image according to variable row heights, then extracting patches based on
    the provided edge positions. The sampling is done using grid sampling with
    bilinear interpolation.
    
    Args:
        images (torch.Tensor): Input images of shape (N, C, H, W)
        row_heights (torch.Tensor): Variable row heights of shape (N, num_rows)
        new_edges (torch.Tensor): Edge positions for patch sampling of shape (N, seqlen+1)
        shape (tuple): Target patch shape (height, width). Default: (16, 16)
        visualize (bool): Whether to save visualization of patches. Default: False
        version (str): Sampling version ('v1' or 'v2'). Default: 'v2'
        mode (str): Interpolation mode for grid sampling. Default: 'bilinear'
    
    Returns:
        torch.Tensor: Sampled patches of shape (N, seqlen, C, patch_h, patch_w)
    """
    seqlen = new_edges.size(1)-1
    device = images.device
    tar_h, tar_w = shape
    
    # Resample image according to variable row heights
    images_reshaped = resample_image_by_heights(images, row_heights, max(tar_h, 2), visualize=visualize, version=version)
    n, c, hh, ww = images_reshaped.shape  # hh=16
    
    # Calculate x-coordinates for patch sampling
    x_starts = new_edges[:, :-1]  # (n, seqlen)
    x_ends   = new_edges[:, 1:]   # (n, seqlen)
    
    # Generate sampling coordinates based on version
    if version == 'v1':
        t_lin = torch.linspace(0, 1, steps=tar_w, device=device).view(1,1,tar_w)   # =>(1,1,tar_w)
    else:
        t_lin = torch.arange(0,tar_w, device=images.device).view(1,1,tar_w)/tar_w   # =>(1,1,tar_w)
    t_lin = t_lin.expand(n, seqlen, tar_w)
    x_starts_ex = x_starts.unsqueeze(-1)  # (n,seqlen,1)
    x_ends_ex   = x_ends.unsqueeze(-1)    # (n,seqlen,1)
    
    # Interpolate x-coordinates across patch width
    x_coords_all = x_starts_ex + (x_ends_ex - x_starts_ex) * t_lin
    x_coords_all = x_coords_all.reshape(n, seqlen*tar_w)  # => (n,tar_w*seqlen)
    
    # Generate y-coordinates for patch sampling
    y_1d = torch.linspace(0, hh-1, steps=tar_h, device=device)  # =>(tar_h,)
    y_2d = y_1d.view(1, tar_h).expand(n, -1)  # => (n,tar_h)
    x_grid = x_coords_all.unsqueeze(1).expand(-1,tar_h,-1)  # =>(n,tar_h,tar_w*seqlen)
    y_grid = y_2d.unsqueeze(-1).expand(-1,-1,seqlen*tar_w)  # =>(n,tar_h,tar_w*seqlen)
    
    # Normalize coordinates to [-1, 1] for grid sampling
    x_grid_norm = 2.0 * (x_grid / (ww - 1)) - 1.0
    y_grid_norm = 2.0 * (y_grid / (hh - 1)) - 1.0
    
    # Perform grid sampling to extract patches
    grid = torch.stack([x_grid_norm, y_grid_norm], dim=-1)
    patches_wide = F.grid_sample(
        images_reshaped,  # (n,c,16,ww)
        grid,             # (n,16,16*seqlen,2)
        mode=mode,
        align_corners=True
    )
    
    # Reshape to final patch format
    patches_5d = patches_wide.reshape(n, c, tar_h, seqlen, tar_w)
    patches = patches_5d.permute(0, 3, 1, 2, 4)  # => (n,seqlen,c,16,16)
    
    # Visualization if requested
    if visualize and n>0:
        # Denormalize images for visualization
        mean = torch.tensor([0.4914, 0.4822, 0.4465], device=images.device).view(3, 1, 1)
        std = torch.tensor([0.2470, 0.2435, 0.2616], device=images.device).view(3, 1, 1)
        mean_images = patches * std + mean
        mean_images = mean_images.clamp(0, 1)  # [batch_size, 3, 32, 32]
        
        # Save each patch as an image
        for j in range(seqlen):
            patch_j = mean_images[0, j]  # => (3,16,16)
            patch_np = patch_j.permute(1,2,0).detach().cpu().numpy()
            plt.figure(figsize=(2,2))
            plt.imshow(patch_np)
            plt.title(f"Patch#{j+1} of Image#0")
            plt.axis("off")
            plt.savefig(f"patch_{j}.png")
            plt.close()
    
    return patches

def resample_image_by_heights(images, row_heights, final_row_height, visualize=False, version='v2', mode='bilinear'):
    """
    Resample images by applying variable row heights to create adaptive image layouts.
    
    This function redistributes image rows according to the provided row heights,
    creating a new image layout where each row can have different heights based
    on the importance or content distribution. The resampling is done using
    grid sampling with bilinear interpolation.
    
    Args:
        images (torch.Tensor): Input images of shape (N, C, H, W)
        row_heights (torch.Tensor): Variable row heights of shape (N, num_rows)
        final_row_height (int): Target height for each resampled row
        visualize (bool): Whether to save visualization of resampled image. Default: False
        version (str): Resampling version ('v1' or 'v2'). Default: 'v2'
        mode (str): Interpolation mode for grid sampling. Default: 'bilinear'
    
    Returns:
        torch.Tensor: Resampled images of shape (N, C, final_row_height, W*num_rows)
    """
    N, C, H, W = images.shape
    num_rows = row_heights.size(1)
    
    # Calculate cumulative heights for row boundaries
    cumsum_heights = row_heights.cumsum(dim=1)  # shape: (N, 14)

    row_chunks = []
    
    # Generate sampling grids based on version
    if version == 'v1':
        h_lin = torch.linspace(0, 1, steps=final_row_height, device=images.device)  
    else:
        h_lin = torch.arange(0, final_row_height, device=images.device)/(final_row_height)
    w_lin = torch.linspace(0, 1, steps=W, device=images.device)

    # Create 2D sampling grids
    h_grid, w_grid = torch.meshgrid(h_lin, w_lin, indexing='ij')  # shape(16,W), (16,W)
    
    # Process each row separately
    for i in range(num_rows):
        # Calculate row boundaries
        if i == 0:
            start_y = torch.zeros_like(cumsum_heights[:, i])           # (N,)
        else:
            start_y = cumsum_heights[:, i-1]                            # (N,)
        end_y = cumsum_heights[:, i]                                    # (N,)

        row_range = end_y - start_y  # (N,) 

        # Prepare for grid sampling
        start_y_ = start_y.view(N, 1, 1)
        row_range_ = row_range.view(N, 1, 1)
        
        # Expand grids to batch dimension
        h_grid_expanded = h_grid.unsqueeze(0).expand(N, -1, -1)
        w_grid_expanded = w_grid.unsqueeze(0).expand(N, -1, -1)
        
        # Calculate source coordinates for grid sampling
        source_y = (start_y_ + row_range_ * h_grid_expanded) / (H - 1) * 2.0 - 1.0
        source_x = w_grid_expanded * 2.0 - 1.0

        # Create sampling grid
        grid = torch.stack([source_x, source_y], dim=-1)  # (N,16,W,2)

        # Sample the row chunk using grid sampling
        row_chunk = F.grid_sample(
            images, 
            grid, 
            mode=mode, 
            padding_mode='border', 
            align_corners=True
        )
        
        row_chunks.append(row_chunk)
    
    # Concatenate all row chunks along width dimension
    rearranged_images = torch.cat(row_chunks, dim=3)

    # Visualization if requested
    if visualize:
        import numpy as np
        img_np = rearranged_images[0].permute(1, 2, 0).detach().cpu().numpy()
        img_np = np.clip(img_np, 0, 1)
        
        plt.figure(figsize=(12, 4))
        plt.title("Rearranged Image with Variable Row Heights")
        plt.imshow(img_np)
        plt.axis('off')
        plt.savefig('rearranged_variable_height.png')
        plt.close()

    return rearranged_images

def resample_tokens_by_heights(x, row_heights, org_h=14):
    """
    Resample tokens based on variable row heights using overlap-based interpolation.
    
    This function redistributes token sequences according to the provided row heights,
    creating new token sequences where each row can have different token counts based
    on the importance distribution. The resampling is done by calculating overlaps
    between old and new row boundaries and using weighted averaging.
    
    Args:
        x (torch.Tensor): Input tokens of shape (N, patch_count, D)
        row_heights (torch.Tensor): Variable row heights of shape (N, max_row_num)
        org_h (int): Original number of rows. Default: 14
    
    Returns:
        torch.Tensor: Resampled tokens of shape (N, new_patch_count, D)
    """
    N, patch_count, D = x.shape
    
    # Reshape tokens to 2D grid format
    old_tokens_2d = x.view(N, -1, org_h, D)  # (N, j=14, col=14, D)

    # Clamp negative heights to zero
    row_heights_clamped = row_heights.clone()
    row_heights_clamped[row_heights_clamped < 0] = 0

    # Calculate cumulative heights for new row boundaries
    cumsum_heights = row_heights_clamped.cumsum(dim=1)  # (N, max_row_num)
    new_starts = torch.cat([
        torch.zeros(N, 1, device=row_heights.device, dtype=row_heights.dtype),
        cumsum_heights[:, :-1]
    ], dim=1)  # (N, max_row_num)
    new_ends = cumsum_heights  # (N, max_row_num)

    # Calculate old row boundaries (assuming 16 tokens per row)
    old_starts = 16 * torch.arange(old_tokens_2d.size(1), device=x.device, dtype=row_heights.dtype)  # (14,)
    old_ends   = old_starts + 16  # (14,)

    # Expand dimensions for overlap calculation
    new_starts_expanded = new_starts.unsqueeze(-1)       # (N, max_row_num, 1)
    new_ends_expanded   = new_ends.unsqueeze(-1)         # (N, max_row_num, 1)
    old_starts_expanded = old_starts.view(1, 1, -1)      # (1, 1, 14)
    old_ends_expanded   = old_ends.view(1, 1, -1)        # (1, 1, 14)

    # Calculate overlap lengths between old and new row boundaries
    # overlap_length: (N, max_row_num, 14)
    overlap_length = (
        torch.min(new_ends_expanded, old_ends_expanded)
        - torch.max(new_starts_expanded, old_starts_expanded)
    ).clamp(min=0)
    overlap_ratio = overlap_length / 16.0

    # Normalize overlap ratios
    overlap_sum = overlap_ratio.sum(dim=2, keepdim=True).clamp(min=1e-6)
    overlap_ratio = overlap_ratio / overlap_sum

    # Apply weighted averaging using einsum
    # old_tokens_2d: (N, j=14, col=14, D)
    # overlap_ratio:    (N, i=max_row_num, j=14)
    new_tokens_2d = torch.einsum('b i j, b j c d -> b i c d', overlap_ratio, old_tokens_2d)

    # Reshape back to sequence format
    new_tokens = new_tokens_2d.view(N, -1, D)

    return new_tokens

def find_quantiles(p_values, pdf, eps=1e-8):
    """
    Find quantiles from a probability density function using cumulative distribution.
    
    This function computes quantiles (inverse CDF values) from a given probability
    density function. It uses the cumulative distribution function to find the
    positions where the cumulative probability equals the given p-values.
    
    Args:
        p_values (torch.Tensor): Probability values to find quantiles for, shape (num_quant,)
        pdf (torch.Tensor): Probability density function, shape (N, seqlen)
        eps (float): Small epsilon value to prevent division by zero. Default: 1e-8
    
    Returns:
        torch.Tensor: Quantile values of shape (N, num_quant)
    
    Example:
        >>> pdf = torch.softmax(torch.randn(2, 10), dim=1)
        >>> p_values = torch.tensor([0.25, 0.5, 0.75])
        >>> quantiles = find_quantiles(p_values, pdf)
        >>> print(quantiles.shape)  # torch.Size([2, 3])
    """
    # Handle empty p_values case
    if p_values.numel() == 0:
        return torch.empty(pdf.size(0), 0, device=pdf.device, dtype=pdf.dtype)
    
    n, seqlen = pdf.shape
    num_quant = p_values.shape[0]
    
    # Calculate cumulative distribution function
    cumsums = torch.cumsum(pdf, dim=1)  # (N, num_rows)
    
    # Create edge positions for interpolation
    edges = torch.linspace(
                0, seqlen, seqlen + 1, device=pdf.device, dtype=pdf.dtype
            ).unsqueeze(0).repeat(n, 1)

    # Expand dimensions for broadcasting
    p_values_expanded = p_values.view(1, 1, num_quant) 
    cumsums_expanded = cumsums.unsqueeze(-1)  # (n, seqlen, 1)
    
    # Find indices where cumulative sum first exceeds p_values
    mask = (cumsums_expanded >= p_values_expanded)  # (n, seqlen, num_quant)
    j_indices = torch.argmax(mask.int(), dim=1)     # (n, num_quant)

    # Handle cases where no cumulative sum exceeds p_values
    mask_sum = mask.sum(dim=1)  # (n, num_quant)
    no_true_mask = (mask_sum == 0)
    j_indices = torch.where(no_true_mask, torch.full_like(j_indices, seqlen - 1), j_indices)

    # Calculate previous indices and areas for linear interpolation
    prev_j = torch.clamp(j_indices - 1, 0, seqlen - 1)
    prev_area = torch.gather(cumsums, dim=1, index=prev_j)
    j_zero_mask = (j_indices == 0)
    prev_area = torch.where(j_zero_mask, torch.zeros_like(prev_area), prev_area)

    # Get PDF values and edge values at quantile indices
    pdf_val = torch.gather(pdf, dim=1, index=j_indices)
    edge_val = torch.gather(edges[:, :-1], dim=1, index=j_indices)

    # Perform linear interpolation to find exact quantile positions
    p_values_expanded_n = p_values.unsqueeze(0).expand(n, num_quant)
    quantiles = edge_val + (p_values_expanded_n - prev_area) / (pdf_val + eps)

    return quantiles

def pdf_to_row_heights(pdf, total_height=224, eps=1e-8, version='x', target_h=None):
    """
    Convert probability density function to row heights for adaptive sampling.
    
    This function transforms a probability density function into row heights that
    can be used for adaptive image or token sampling. It uses quantile-based
    approach to distribute the total height according to the importance weights
    in the PDF.
    
    Args:
        pdf (torch.Tensor): Probability density function of shape (N, num_patches)
        total_height (int): Total height to distribute. Default: 224
        eps (float): Small epsilon value to prevent division by zero. Default: 1e-8
        version (str): Processing version ('r' for raw, 'x' for 2D). Default: 'x'
        target_h (int, optional): Target number of rows. If None, uses num_rows. Default: None
    
    Returns:
        torch.Tensor: Row heights of shape (N, num_rows)
    
    Example:
        >>> pdf = torch.softmax(torch.randn(2, 196), dim=1)  # 14x14 patches
        >>> row_heights = pdf_to_row_heights(pdf, total_height=224)
        >>> print(row_heights.shape)  # torch.Size([2, 14])
    """
    N, num_patches = pdf.shape
    
    # Process PDF based on version
    if version == 'r':
        # Raw version: use PDF directly as row PDF
        row_pdf = pdf
        num_rows = pdf.size(1)
    else:
        # 2D version: reshape to grid and sum across columns
        num_cols = 14
        num_rows = num_patches // num_cols  

        pdf_2d = pdf.view(N, num_rows, num_cols)  # (N, num_rows, num_cols)
        row_pdf = pdf_2d.sum(dim=-1)  # (N, num_rows)
    
    # Set target height if not provided
    if not target_h:
        target_h = num_rows

    # Normalize row PDF
    row_sum = row_pdf.sum(dim=-1, keepdim=True) + eps
    row_pdf = row_pdf / row_sum  # (N, num_rows)

    # Generate quantile positions for adaptive sampling
    # [0, 1/num_rows, 2/num_rows,..., 1]
    quant_p = torch.linspace(0, 1, target_h + 1, device=pdf.device, dtype=pdf.dtype)[1:-1]
    edges = torch.linspace(0, num_rows, num_rows + 1, device=pdf.device, dtype=pdf.dtype)  # (num_rows + 1,)
    edges = edges.unsqueeze(0).expand(N, -1)  # (N, num_rows + 1)
    
    # Calculate new edges using quantiles
    if quant_p.numel() > 0:
        quantiles = find_quantiles(quant_p, row_pdf)  # (N, num_rows)
        new_edges = torch.cat([edges[:, :1], quantiles, edges[:, -1:]], dim=1)  # (N, num_rows + 1)
    else:
        new_edges = torch.cat([edges[:, :1], edges[:, -1:]], dim=1)  # (N, 2)

    # Calculate row heights from edge differences
    raw_row_heights = new_edges[:, 1:] - new_edges[:, :-1]  # (N, num_rows)
    
    # Scale to total height
    row_heights = raw_row_heights * (total_height / num_rows)  # (N, num_rows)

    return row_heights

def get_base_edges(seqlen, x: torch.Tensor):
    """
    Generate base edge positions for a given sequence length.
    
    This function creates evenly spaced edge positions from 0 to seqlen,
    which serve as the baseline for edge-based operations.
    
    Args:
        seqlen (int): Length of the sequence
        x (torch.Tensor): Reference tensor for device and dtype
    
    Returns:
        torch.Tensor: Base edges of shape (N, seqlen + 1)
    
    Example:
        >>> x = torch.randn(2, 10)
        >>> edges = get_base_edges(10, x)
        >>> print(edges.shape)  # torch.Size([2, 11])
    """
    edges = torch.linspace(0, seqlen, seqlen + 1, device=x.device, dtype=x.dtype)  # (num_rows + 1,)
    edges = edges.unsqueeze(0).expand(x.size(0), -1)  # (N, num_rows + 1)
    return edges

def get_edges_from_pdf(pdf, new_seqlen=None):
    """
    Generate adaptive edges from a probability density function.
    
    This function creates edge positions based on a PDF using quantile-based
    approach, allowing for adaptive sampling based on importance weights.
    
    Args:
        pdf (torch.Tensor): Probability density function of shape (N, seqlen)
        new_seqlen (int, optional): Target sequence length. If None, uses original seqlen. Default: None
    
    Returns:
        torch.Tensor: Adaptive edges of shape (N, new_seqlen + 1)
    
    Example:
        >>> pdf = torch.softmax(torch.randn(2, 10), dim=1)
        >>> edges = get_edges_from_pdf(pdf, new_seqlen=8)
        >>> print(edges.shape)  # torch.Size([2, 9])
    """
    seqlen = pdf.size(1)
    new_seqlen = new_seqlen or seqlen
    edges = get_base_edges(seqlen, pdf)
    quant_p = torch.linspace(0, 1, new_seqlen + 1, device=pdf.device, dtype=pdf.dtype)[1:-1]

    if quant_p.numel() > 0:
        _edges = get_base_edges(pdf.size(1), pdf)
        quantiles = find_quantiles(quant_p, pdf)
        new_edges = torch.cat([_edges[:, 0:1], quantiles, _edges[:, -1:]], dim=1)/pdf.size(1)*seqlen
    else:
        new_edges = torch.cat([edges[:, 0:1], edges[:, -1:]], dim=1)

    return new_edges

def resample_tokens_by_edges(tokens, edges):
    """
    Resample tokens based on edge positions using overlap-based interpolation.
    
    This function redistributes token sequences according to the provided edge
    positions, creating new token sequences with potentially different lengths.
    The resampling is done by calculating overlaps between old and new token
    boundaries and using weighted averaging.
    
    Args:
        tokens (torch.Tensor): Input tokens of shape (N, seqlen, dim)
        edges (torch.Tensor): Edge positions for resampling of shape (N, new_seqlen + 1)
    
    Returns:
        torch.Tensor: Resampled tokens of shape (N, new_seqlen, dim)
    """
    seqlen = tokens.size(1)
    old_edges = get_base_edges(seqlen, tokens)
    edges = edges / edges.max() * old_edges.max()

    # Calculate overlaps between old and new token boundaries
    # overlap: (n, new_seqlen, seqlen)
    new_start = edges[:, :-1].clone().unsqueeze(2)  # (n, new_seqlen, 1)
    new_end   = edges[:, 1:].clone().unsqueeze(2)   # (n, new_seqlen, 1)

    old_start = old_edges[:, :-1].clone().unsqueeze(1)      # (n, 1, seqlen)
    old_end   = old_edges[:, 1:].clone().unsqueeze(1)       # (n, 1, seqlen)

    overlap = torch.clamp(
        torch.min(new_end, old_end) - torch.max(new_start, old_start),
        min=0.0
    )  # (n, new_seqlen, seqlen)

    # Calculate weights for weighted averaging
    raw_weights = overlap  # (n, new_seqlen, seqlen)
    sum_weights = raw_weights.sum(dim=2, keepdim=True) + 1e-12
    weight = raw_weights / sum_weights  # (n, new_seqlen, seqlen)

    # Apply weighted averaging to resample tokens
    # weight: (n, new_seqlen, seqlen)
    # tokens: (n, seqlen, dim)
    new_tokens = torch.bmm(weight, tokens)  # (n, new_seqlen, dim)
    return new_tokens


def unpatchify(patches: torch.Tensor, patch_size: int, shape=None) -> torch.Tensor:
    """
    Reconstruct images from patches by reversing the patchification process.
    
    This function takes a tensor of image patches and reconstructs the original
    images by arranging the patches in a grid pattern. It's the inverse operation
    of patchification commonly used in vision transformers.
    
    Args:
        patches (torch.Tensor): Image patches of shape (N, num_patches, C, P, P)
        patch_size (int): Size of each patch (P)
        shape (tuple, optional): Target grid shape (h, w). If None, assumes square grid. Default: None
    
    Returns:
        torch.Tensor: Reconstructed images of shape (N, C, h*P, w*P)
    
    Raises:
        AssertionError: If patch size doesn't match or grid dimensions don't multiply to num_patches
    
    Example:
        >>> patches = torch.randn(2, 196, 3, 16, 16)  # 14x14 patches
        >>> images = unpatchify(patches, patch_size=16)
        >>> print(images.shape)  # torch.Size([2, 3, 224, 224])
    """
    N, num_patches, C, P, _ = patches.shape
    assert P == patch_size
    
    # Determine grid dimensions
    if shape is None:
        h = int(math.sqrt(num_patches))
        w = h
    else:
        h, w = shape
    assert h * w == num_patches
    
    # Reshape patches to grid format
    # (N, H*W, C, P, P) → (N, H, W, C, P, P)
    patches = patches.view(N, h, w, C, P, P)
    patches = patches.permute(0, 3, 1, 4, 2, 5)
    imgs = patches.reshape(N, C, h * P, w * P)
    return imgs


def transform_points(points, row_heights, col_heights, original_shape, target_grid_shape, inverse=False):
    """
    Transform points between original and warped coordinate systems.
    
    This function performs coordinate transformations between the original image
    coordinate system and a warped coordinate system based on variable row and
    column heights. It supports both forward (original to warped) and inverse
    (warped to original) transformations.
    
    Args:
        points (torch.Tensor): Points to transform of shape [..., 2], where last dimension is (x, y)
        row_heights (torch.Tensor): Variable row heights for the transformation
        col_heights (torch.Tensor): Variable column heights for the transformation
        original_shape (tuple): Original image shape (H, W)
        target_grid_shape (tuple): Target grid shape (target_h, target_w)
        inverse (bool): If True, perform inverse transformation (warped to original). Default: False
    
    Returns:
        torch.Tensor: Transformed points with same shape as input
    """
    H, W = original_shape
    target_h, target_w = target_grid_shape
    
    # Normalize heights/widths to match original dimensions
    row_heights = (row_heights / row_heights.sum()) * H
    col_heights = (col_heights / col_heights.sum()) * W

    # Calculate cumulative boundaries
    y_boundaries = torch.cumsum(row_heights, dim=0)
    x_boundaries = torch.cumsum(col_heights, dim=0)
    y_boundaries = torch.cat([torch.zeros(1, device=points.device), y_boundaries])
    x_boundaries = torch.cat([torch.zeros(1, device=points.device), x_boundaries])
    
    # Preserve original shape for reshaping
    input_shape = points.shape
    points = points.view(-1, 2)
    
    transformed_points = torch.zeros_like(points)
    
    # --- Core coordinate transformation logic ---
    if not inverse:  # Forward transformation: original -> warped
        warped_H, warped_W = H, W  # Assume warped image has same dimensions as original
        patch_size_h, patch_size_w = warped_H / target_h, warped_W / target_w
        
        # Transform Y coordinates
        y_indices = torch.searchsorted(y_boundaries, points[:, 1], right=True) - 1
        y_indices = y_indices.clamp(min=0, max=target_h - 1)
        y_start, y_end = y_boundaries[y_indices], y_boundaries[y_indices + 1]
        y_relative = (points[:, 1] - y_start) / (y_end - y_start + 1e-8)
        transformed_points[:, 1] = (y_indices + y_relative) * patch_size_h
        
        # Transform X coordinates
        x_indices = torch.searchsorted(x_boundaries, points[:, 0], right=True) - 1
        x_indices = x_indices.clamp(min=0, max=target_w - 1)
        x_start, x_end = x_boundaries[x_indices], x_boundaries[x_indices + 1]
        x_relative = (points[:, 0] - x_start) / (x_end - x_start + 1e-8)
        transformed_points[:, 0] = (x_indices + x_relative) * patch_size_w

    else:  # Inverse transformation: warped -> original
        warped_H, warped_W = H, W
        patch_size_h, patch_size_w = warped_H / target_h, warped_W / target_w

        # Transform Y coordinates
        y_indices = (points[:, 1] / patch_size_h).clamp(min=0, max=target_h - 1).long()
        y_relative = (points[:, 1] % patch_size_h) / patch_size_h
        y_start, y_end = y_boundaries[y_indices], y_boundaries[y_indices + 1]
        transformed_points[:, 1] = y_start + y_relative * (y_end - y_start)
        
        # Transform X coordinates
        x_indices = (points[:, 0] / patch_size_w).clamp(min=0, max=target_w - 1).long()
        x_relative = (points[:, 0] % patch_size_w) / patch_size_w
        x_start, x_end = x_boundaries[x_indices], x_boundaries[x_indices + 1]
        transformed_points[:, 0] = x_start + x_relative * (x_end - x_start)

    return transformed_points.view(input_shape)

def transform_bboxes(bboxes, *args, **kwargs):
    """
    Transform bounding boxes using the same transformation as points.
    
    This function applies coordinate transformation to bounding boxes by
    transforming their corner points (top-left and bottom-right) and
    reconstructing the bounding box from the transformed corners.
    
    Args:
        bboxes (torch.Tensor): Bounding boxes of shape (N, 4) with format (x1, y1, x2, y2)
        *args: Arguments passed to transform_points function
        **kwargs: Keyword arguments passed to transform_points function
    
    Returns:
        torch.Tensor: Transformed bounding boxes of shape (N, 4)
    """
    points1 = bboxes[:, :2]  # Top-left corners
    points2 = bboxes[:, 2:]  # Bottom-right corners
    new_points1 = transform_points(points1, *args, **kwargs)
    new_points2 = transform_points(points2, *args, **kwargs)
    return torch.cat([new_points1, new_points2], dim=1)

def transform_image_or_mask(img_tensor, row_heights, col_heights, inverse=False):
    """
    Transform images or masks using variable row and column heights.
    
    This function applies geometric transformation to images or masks based on
    variable row and column heights. It uses grid sampling to perform the
    transformation, automatically choosing the appropriate interpolation mode
    based on the input tensor type.
    
    Args:
        img_tensor (torch.Tensor): Input image or mask of shape (C, H, W)
        row_heights (torch.Tensor): Variable row heights for the transformation
        col_heights (torch.Tensor): Variable column heights for the transformation
        inverse (bool): If True, perform inverse transformation. Default: False
    
    Returns:
        torch.Tensor: Transformed image or mask of shape (C, H, W)
    """
    C, H, W = img_tensor.shape
    device = img_tensor.device
    
    # Create a coordinate grid covering the entire image
    y_coords = torch.linspace(-1, 1, H, device=device)
    x_coords = torch.linspace(-1, 1, W, device=device)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
    
    # Stack coordinates to create grid points
    # original_grid_points contains coordinates for each pixel in the target image
    original_grid_points = torch.stack((grid_x, grid_y), dim=-1)  # [H, W, 2]
    
    # Convert grid coordinates from [-1, 1] to pixel coordinates [0, W-1] and [0, H-1]
    points_to_transform = torch.zeros_like(original_grid_points)
    points_to_transform[..., 0] = (original_grid_points[..., 0] + 1) / 2 * (W - 1)
    points_to_transform[..., 1] = (original_grid_points[..., 1] + 1) / 2 * (H - 1)
    
    # Calculate sampling grid
    # inverse=not inverse: The logic here is a bit convoluted
    # If we want to generate a warped image (forward), we need to know which pixel (x,y) 
    # in the original image corresponds to each pixel (x',y') in the warped image
    # This is actually performing an inverse point transformation
    sampling_points_pixels = transform_points(
        points_to_transform, row_heights, col_heights, 
        (H, W), (len(row_heights), len(col_heights)), 
        inverse=not inverse
    )
    
    # Normalize sampling point coordinates to [-1, 1]
    sampling_grid = torch.zeros_like(sampling_points_pixels)
    sampling_grid[..., 0] = (sampling_points_pixels[..., 0] / (W - 1)) * 2 - 1
    sampling_grid[..., 1] = (sampling_points_pixels[..., 1] / (H - 1)) * 2 - 1

    # F.grid_sample requires [N, C, H, W] and [N, H, W, 2]
    # mode='nearest' for masks and binary images, 'bilinear' for regular images
    mode = 'nearest' if torch.all((img_tensor == 0) | (img_tensor == 1)) else 'bilinear'
    
    warped_img = F.grid_sample(
        img_tensor.unsqueeze(0), 
        sampling_grid.unsqueeze(0),
        mode=mode,
        padding_mode='border',
        align_corners=True
    )
    
    return warped_img.squeeze(0)