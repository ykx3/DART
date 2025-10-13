## Video

### Overview
This folder contains the video module for dynamic, content-aware patch embedding. Frames are stacked into a tall image, a non-uniform grid is generated from predicted importance scores, and more patches are sampled from informative regions. The implementation is adapted from the VideoMamba project.

- Acknowledgement: adapted from [VideoMamba](https://github.com/OpenGVLab/VideoMamba)

### Installation
- Python 3.8+
- PyTorch, torchvision, timm
- The MobileNetV3 feature extractor uses torchvision weights by default. If running offline, set `pretrained=False` when constructing `MobileNetFeatureExtractor`.

Example (install core deps):
```bash
pip install torch torchvision timm
```

### Quick Start
```python
import torch
from video.video_patch_embed import DynamicVideoPatchEmbed

B, C, T, H, W = 2, 3, 8, 224, 224
x = torch.randn(B, C, T, H, W)

embed = DynamicVideoPatchEmbed(
    img_size=224,
    patch_size=16,
    in_chans=3,
    embed_dim=768,
    num_frames=T,
    num_patches=14 * 14 * T,
)

out = embed(x)  # shape: (B, num_patches, embed_dim) if no cls/pos/time given
print(out.shape)
```

### Training
Minimal distributed example (VideoMamba-style fine-tuning):
```bash
DATA_PATH=../../data/Kinetics-400
OUTPUT_DIR=tmp
NP=8
NC=2
B=64
python -m torch.distributed.launch --nproc_per_node=$NC --use_env main.py \
    --model dvideomamba_tiny \
    --data_path ${DATA_PATH} \
    --prefix ${DATA_PATH} \
    --data_set 'Kinetics_sparse' \
    --split ' ' \
    --nb_classes 400 \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size $B \
    --num_sample 2 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 100 \
    --num_frames 16 \
    --num_workers $NP \
    --warmup_epochs 5 \
    --tubelet_size 1 \
    --epochs 70 \
    --lr 2e-4 \
    --drop_path 0.1 \
    --aa rand-m5-n2-mstd0.25-inc1 \
    --opt adamw \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.1 \
    --test_num_segment 4 \
    --test_num_crop 3 \
    --dist_eval \
    --test_best
```
- Hyperparameters and recipes follow the original VideoMamba scripts; we add `--num_patches` to control the total number of dynamic patches (e.g., `14×14×T`). See [VideoMamba](https://github.com/OpenGVLab/VideoMamba) for more details.

Example with `--num_patches` explicitly set (e.g., 14×14 over 16 frames):
```bash
python -m torch.distributed.launch --nproc_per_node=$NC --use_env main.py \
    --model dvideomamba_tiny \
    --data_path ${DATA_PATH} \
    --prefix ${DATA_PATH} \
    --data_set 'Kinetics_sparse' \
    --split ' ' \
    --nb_classes 400 \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size $B \
    --num_sample 2 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 100 \
    --num_frames 16 \
    --num_patches $((14*14*16)) \
    --num_workers $NP \
    --warmup_epochs 5 \
    --tubelet_size 1 \
    --epochs 70 \
    --lr 2e-4 \
    --drop_path 0.1 \
    --aa rand-m5-n2-mstd0.25-inc1 \
    --opt adamw \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.1 \
    --test_num_segment 4 \
    --test_num_crop 3 \
    --dist_eval \
    --test_best
```

### Notes
- Position/time embeddings are optional. If provided, they will be dynamically resampled to match the non-uniform grid.
- For reproducibility, ensure that the input spatial size matches `img_size` and the temporal length matches `num_frames`.

### Acknowledgement
- This module is adapted from `VideoMamba: State Space Model for Efficient Video Understanding` [link](https://github.com/OpenGVLab/VideoMamba). 