import torch
from torch import Tensor
from typing import Optional

from mmseg.registry import MODELS
from mmseg.models.segmentors import EncoderDecoder
from mmseg.structures import SegDataSample 

from .reshaper import ReshaperWithInverse


@MODELS.register_module() 
class DAIRSegmentor(EncoderDecoder):
    """
    A segmentation model integrated with dynamic image reshaping module.
    
    This segmentor performs dynamic adaptive image reshaping during training and
    inverse transformation during inference, allowing the model to focus on
    important regions while maintaining full resolution output.
    
    Attributes:
        reshaper_transform (ReshaperWithInverse): Dynamic image reshaping module
    """
    
    def __init__(self, reshaper_cfg, **kwargs):
        """
        Initialize the DAIRSegmentor.
        
        Args:
            reshaper_cfg (dict): Configuration for the reshaper module
            **kwargs: Additional arguments passed to EncoderDecoder
        """
        super().__init__(**kwargs)
        # Instantiate our transformation module
        self.reshaper_transform = ReshaperWithInverse(**reshaper_cfg)

    def loss(self, inputs: torch.Tensor, data_samples):
        """
        Calculate loss during training.
        
        This method performs the following steps:
        1. Apply forward transformation to input images and GT masks
        2. Pass transformed images through the network
        3. Calculate loss using transformed GT masks
        4. Handle auxiliary heads if present
        
        Args:
            inputs (torch.Tensor): Input images of shape (B, C, H, W)
            data_samples: List of SegDataSample containing ground truth information
        
        Returns:
            dict: Dictionary containing loss values
        """
        # Extract GT masks from data_samples
        gt_masks = self._stack_batch_gt(data_samples)

        # 1. Perform forward transformation
        warped_inputs, warped_gt_masks, _ = self.reshaper_transform.forward_transform(
            img=inputs, 
            gt_mask=gt_masks
        )
        
        # Get number of classes and ignore_index from decode_head
        num_classes = self.decode_head.num_classes
        # Usually ignore_index is defined in the loss function
        ignore_index = self.decode_head.ignore_index
        
        # Create a boolean mask marking all pixels that are neither valid classes nor ignore_index
        valid_class_mask = (warped_gt_masks >= 0) & (warped_gt_masks < num_classes)
        ignore_pixel_mask = (warped_gt_masks == ignore_index)
        
        # ~ represents logical NOT
        # Set all invalid pixel values to ignore_index
        warped_gt_masks[~(valid_class_mask | ignore_pixel_mask)] = ignore_index

        # 2. Update data_samples with transformed GT
        for i, sample in enumerate(data_samples):
            sample.gt_sem_seg.data = warped_gt_masks[i]
            
        # 3. Extract features, pass through decode head and calculate loss as usual
        #    Parent class loss method will handle all of this
        x = self.extract_feat(warped_inputs)
        losses = self.decode_head.loss(x, data_samples, self.train_cfg)
        
        # Check if auxiliary head exists and model is in training mode
        if self.with_auxiliary_head:
            # Call auxiliary head loss method to calculate loss
            loss_aux = self.auxiliary_head.loss(x, data_samples, self.train_cfg)
            # Add 'aux.' prefix to each key in auxiliary head loss dictionary, then merge into total loss
            # Use dictionary comprehension to manually add prefix to keys
            loss_aux_with_prefix = {f'aux.{k}': v for k, v in loss_aux.items()}
            losses.update(loss_aux_with_prefix)
        
        return losses

    def predict(self, inputs, data_samples, vis=False):
        """
        Perform prediction during inference.
        
        This method performs the following steps:
        1. Apply forward transformation to input images
        2. Pass transformed images through network to get warped_pred
        3. Apply inverse transformation to warped_pred to get final_pred
        
        Args:
            inputs (torch.Tensor): Input images of shape (B, C, H, W)
            data_samples: List of SegDataSample (can be None)
            vis (bool): If True, generate visualization of predictions
        
        Returns:
            List[SegDataSample]: List of segmentation results
        """
        # 1. Perform forward transformation, only transform images, and save transformation parameters
        warped_inputs, _, transform_params = self.reshaper_transform.forward_transform(img=inputs)
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]

        # 2. Normal inference to get warped_pred (logits)
        x = self.extract_feat(warped_inputs)
        seg_logits = self.decode_head.predict(x, batch_img_metas, self.test_cfg)
        
        if vis:
            try:
                import cv2
                import matplotlib.pyplot as plt
                import numpy as np
                
                # Get the first sample
                idx = 0
                img = warped_inputs[idx].detach().cpu().numpy()
                logit = seg_logits[idx].detach().cpu()
                
                # Denormalize image
                if hasattr(self, 'img_norm_cfg'):
                    mean = np.array(self.img_norm_cfg['mean']).reshape(3, 1, 1)
                    std = np.array(self.img_norm_cfg['std']).reshape(3, 1, 1)
                else:
                    mean = np.array([123.675, 116.28, 103.53]).reshape(3, 1, 1)
                    std = np.array([58.395, 57.12, 57.375]).reshape(3, 1, 1)
                
                img = img * std + mean
                img = np.clip(img, 0, 255).astype(np.uint8)
                img = img.transpose(1, 2, 0)  # CHW->HWC
                
                # Get prediction results
                pred_mask = logit.argmax(dim=0).numpy().astype(np.uint8)
                
                # Get ADE20K palette
                try:
                    from mmseg.utils import ade_palette
                    palette = np.array(ade_palette(), dtype=np.uint8)
                except ImportError:
                    print("Warning: ade_palette not found, using random palette")
                    # Create random palette
                    palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)
                    palette[0] = [0, 0, 0]  # Set background to black
                    
                # Create colored prediction image
                color_pred = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
                
                # Only color valid classes
                for class_id in np.unique(pred_mask):
                    if class_id < len(palette):  # Ensure within palette range
                        mask = (pred_mask == class_id)
                        color_pred[mask] = palette[class_id]
                
                # Create visualization image (original + prediction overlay)
                alpha = 0.5
                blended = cv2.addWeighted(img, 1 - alpha, color_pred, alpha, 0)
                
                # Add prediction contours
                if len(pred_mask.shape) == 2:  # Ensure 2D mask
                    # Find contours
                    contours, _ = cv2.findContours(
                        pred_mask.astype(np.uint8), 
                        cv2.RETR_EXTERNAL, 
                        cv2.CHAIN_APPROX_SIMPLE
                    )
                    # Draw contours on overlay image
                    cv2.drawContours(blended, contours, -1, (0, 255, 0), 1)
                
                # Create comparison image
                plt.figure(figsize=(10, 5))
                plt.subplot(121)
                plt.title("Original Transformed")
                plt.imshow(img)
                plt.axis('off')
                
                plt.subplot(122)
                plt.title("Prediction Overlay")
                plt.imshow(blended)
                plt.axis('off')
                
                plt.tight_layout()
                plt.savefig('warped_prediction_visualization.png', dpi=300)
                plt.close()
                print("Visualization results saved as: warped_prediction_visualization.png")
            
            except ImportError:
                print("Warning: Missing visualization dependencies (cv2/matplotlib), skipping visualization")
            except Exception as e:
                print(f"Error during visualization: {str(e)}")

        
        # 3. Execute inverse transformation
        final_pred_logits = self.reshaper_transform.inverse_transform(seg_logits, transform_params)

        # 4. Put the inverse transformed results back into data_samples
        #    MMSeg predict method requires returning a list of SegDataSample
        return self.postprocess_result(final_pred_logits, data_samples)
        
    def _stack_batch_gt(self, data_samples):
        """
        Stack ground truth masks from data samples into a batch tensor.
        
        Args:
            data_samples: List of SegDataSample containing ground truth information
        
        Returns:
            torch.Tensor: Stacked ground truth masks of shape (B, H, W)
        """
        gt_list = [
            sample.gt_sem_seg.data for sample in data_samples
        ]
        return torch.stack(gt_list, dim=0)
