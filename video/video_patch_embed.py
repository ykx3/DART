import math
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from timm.models.layers import to_2tuple

import sys
sys.path.append('..')
from dart.tools import (
    pdf_to_row_heights,
    resample_tokens_by_heights,
    get_edges_from_pdf,
    resample_tokens_by_edges,
    dynamic_image_patch_sample
)


class DynamicVideoPatchEmbed(nn.Module):
    """
    A dynamic video patch embedding module that extracts patches based on content importance.

    This module first predicts an "intensity" score to identify key regions in videos (typically regions with more noticeable motion changes).
    It then uses these scores to construct a non-uniform grid, sampling more patches from important regions.
    Implementation-wise, video frames are stacked into a tall image, variable-height row splits are applied, and variable-width patch sampling is performed within these rows.
    Spatial and temporal position embeddings are resampled accordingly to match this new dynamic layout.
    """
    def __init__(self,
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_chans: int = 3,
                 embed_dim: int = 768,
                 num_frames: int = 8,
                 num_patches: int = 1568,  # e.g., 14*14*8
                 ):
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input channels.
            embed_dim (int): Embedding dimension.
            num_frames (int): Number of video frames.
            num_patches (int): Target number of output patches.
        """
        super().__init__()
        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.num_patches = num_patches

        # Target number of rows for the dynamic grid, controls vertical resolution of importance sampling
        self.target_h = int(math.sqrt(num_patches / num_frames) * num_frames)

        # Patch projection layer, applied after dynamic sampling
        self.proj = nn.Conv3d(in_chans, embed_dim,
                              kernel_size=(1, *self.patch_size), stride=(1, *self.patch_size))

        # Backbone network for predicting importance scores
        # Uses a MobileNet backbone to extract per-frame features
        self.score_backbone = MobileNetFeatureExtractor(pretrained=True)
        
        # Feature dimension output from the backbone (here 48)
        feature_dim = 48
        self.score_head1 = nn.Linear(feature_dim, feature_dim)
        self.score_head2 = nn.Linear(feature_dim, feature_dim)


    def forward(self, x: torch.Tensor, pos_embed: Optional[torch.Tensor] = None, 
                time_embed: Optional[torch.Tensor] = None, cls_token: Optional[torch.Tensor] = None, 
                ret_dict: bool = False) -> torch.Tensor | dict:
        """
        Args:
            x (torch.Tensor): Input video tensor of shape (B, C, T, H, W).
            pos_embed (torch.Tensor, optional): Spatial position embeddings of shape (1, N_spatial+1, D).
            time_embed (torch.Tensor, optional): Temporal position embeddings of shape (1, T, D).
            cls_token (torch.Tensor, optional): Classification token of shape (1, 1, D).
            ret_dict (bool): If True, return a dictionary containing intermediate values.

        Returns:
            torch.Tensor | dict: Patch embedding sequence or a dictionary with intermediate results.
        """
        B, C, T, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1] and T == self.num_frames, \
            f"Input video size ({T}x{H}x{W}) doesn't match model ({self.num_frames}x{self.img_size[0]}x{self.img_size[1]})."

        # Dictionary for storing intermediate results for analysis
        ret = {'x': x}

        # --- 1. Predict importance scores ---
        # Process frame-by-frame to extract features. Assume feature grid size is (14, 14).
        x_frames = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        _, features = self.score_backbone(x_frames, 8) # (B*T, 48, H_feat, W_feat)
        
        # Reshape features to compute temporal differences
        features = features.view(B, T, features.size(1), -1) # (B, T, 48, 196)
        
        # Compute temporal difference of features
        diff = features[:, 1:] - features[:, :-1] # (B, T-1, 48, 196)
        
        # Permute dimensions to feed into linear layers
        diff = diff.permute(0, 1, 3, 2) # (B, T-1, 196, 48)
        
        # Use global context as input to a sigmoid gate to normalize based on overall video motion
        global_context = features[:, 1:].permute(0, 1, 3, 2).mean(dim=(1, 2), keepdim=True) # (B, 1, 1, 48)

        # Compute strength scores from temporal differences
        strength = (F.relu(self.score_head1(diff)) * F.sigmoid(self.score_head2(global_context))).sum(-1)
        strength = strength.view(B, (T - 1) * 14 * 14)

        # Normalize scores per video
        strength = (strength - strength.mean(1, keepdim=True)) / (strength.std(1, keepdim=True) + 1e-8)
        strength = F.sigmoid((strength - 1) * 1.5)

        # Add a uniform score for the first frame (no previous frame to compute differences)
        first_frame_strength = torch.ones_like(strength[:, :14*14]) * 0.5
        score = torch.cat([first_frame_strength, strength], dim=1)

        # Convert scores to a probability density function (PDF)
        pdf = score / score.sum(dim=-1, keepdim=True)
        ret.update({'pdf': pdf, 'score': score})

        # --- 2. Reshape video and embeddings for sampling ---
        # Treat the video as a single tall image to apply 2D dynamic sampling
        x_tall = x.reshape(B, C, T * H, W)

        # Prepare spatial position embeddings for resampling
        pos_embed_spatial = None
        if pos_embed is not None:
            pos_embed_spatial = pos_embed[:, 1:, :]  # exclude CLS token
            pos_embed_spatial = pos_embed_spatial.repeat(B, T, 1) # (B, T*N_spatial, D)

        # Prepare temporal embeddings for resampling
        if time_embed is not None:
            time_embed = time_embed.repeat(B, 1, 1).unsqueeze(2).repeat(1, 1, 14*14, 1)
            time_embed = time_embed.view(B, T * 14 * 14, -1)

        # --- 3. Dynamic resampling based on the PDF ---
        # Compute variable row heights in the tall image
        row_heights = pdf_to_row_heights(pdf, T * H, target_h=self.target_h)
        ret['row_heights'] = row_heights
        
        # Resample PDF and embeddings according to the new variable row heights
        pdf_resampled = resample_tokens_by_heights(pdf.unsqueeze(-1), row_heights).squeeze(-1)
        pdf_resampled = pdf_resampled / pdf_resampled.sum(dim=-1, keepdim=True)  # renormalize
        ret['reshaped_pdf'] = pdf_resampled
        
        if pos_embed_spatial is not None:
            pos_embed_spatial = resample_tokens_by_heights(pos_embed_spatial, row_heights)
        if time_embed is not None:
            time_embed = resample_tokens_by_heights(time_embed, row_heights)
            
        # Compute new horizontal patch boundaries (edges) based on the resampled PDF
        new_edges = get_edges_from_pdf(pdf_resampled, new_seqlen=self.num_patches)
        
        # Resample embeddings again according to the new horizontal boundaries
        if pos_embed_spatial is not None:
            pos_embed_spatial = resample_tokens_by_edges(pos_embed_spatial, new_edges)
        if time_embed is not None:
            time_embed = resample_tokens_by_edges(time_embed, new_edges)

        # --- 4. Sample patches and finalize ---
        # Scale boundaries to the actual size of the tall image
        new_edges_scaled = new_edges * W * self.target_h / new_edges[0, -1].item()
        ret['new_edges'] = new_edges_scaled
        
        # Sample image patches using the dynamic grid
        patches = dynamic_image_patch_sample(x_tall, row_heights, new_edges_scaled, shape=self.patch_size)

        # Project patches to the embedding dimension
        patches = patches.reshape(B * self.num_patches, C, 1, self.patch_size[0], self.patch_size[1])
        x = self.proj(patches).view(B, self.num_patches, self.embed_dim)

        # Add resampled spatial and temporal embeddings
        if pos_embed_spatial is not None:
            x = x + pos_embed_spatial
        if time_embed is not None:
            x = x + time_embed

        # Add CLS token
        if cls_token is not None:
            cls_token = (cls_token + pos_embed[:, 0, :]).repeat(B, 1, 1)
            x = torch.cat([cls_token, x], dim=1)

        return x if not ret_dict else ret

class MobileNetFeatureExtractor(nn.Module):
    """
    A concise MobileNetV3 feature extractor.

    This module loads a pretrained MobileNetV3-Small model and freezes the parameters of its feature extraction layers.
    Its main function is to extract feature maps from an intermediate layer and the final feature layer.
    All extra, task-specific heads (e.g., MLPs, classification heads) are removed to keep the module general-purpose.
    """
    def __init__(self, pretrained: bool = True, freeze_features: bool = True):
        """
        Args:
            pretrained (bool): If True, load ImageNet-pretrained weights.
            freeze_features (bool): If True, freeze all feature extraction layer parameters so they are not updated during training.
        """
        super().__init__()
        # Load MobileNetV3-Small model
        mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
        self.features = mobilenet.features[:11]

        if freeze_features:
            for param in self.features.parameters():
                param.requires_grad = False
        
    def forward(self, x: torch.Tensor, intermediate_depth: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run forward pass to extract features.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).
            intermediate_depth (int): Defines the position ("depth") of the intermediate feature extraction.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple of (final_features, intermediate_features).
        """
        # Execute within no_grad since this is a pure feature extractor to save compute and memory
        with torch.no_grad():
            intermediate_features = self.features[:intermediate_depth](x)
            final_features = self.features[intermediate_depth:](intermediate_features)
            
        return final_features, intermediate_features