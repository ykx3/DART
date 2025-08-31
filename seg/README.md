## Segmentation (seg)

### Overview
This folder provides segmentation-related components/configs that integrate with the MMSegmentation framework. To run anything here, install MMSegmentation first and then place/copy the files in this folder into the appropriate locations of your MMSegmentation project.

- Prerequisite: install MMSegmentation by following the official docs: [MMSegmentation Documentation](https://mmsegmentation.readthedocs.io/en/main/)

### Setup
Copy the contents of `seg/` into your local MMSegmentation repo with matching structure. Typical placements:
- Configs: `your_mmseg_repo/configs/dart/` (or another subfolder under `configs/`)
- Custom datasets/transforms: `your_mmseg_repo/mmseg/datasets/` (and `.../datasets/transforms/` if provided)
- Custom models/heads/losses/necks: `your_mmseg_repo/mmseg/models/` (e.g., `.../backbones/`, `.../decode_heads/`, `.../losses/`)
- Utilities: `your_mmseg_repo/mmseg/utils/` or the closest matching module

If a file already exists upstream, either replace it or merge changes as needed.

### Run
Once files are placed, use the standard MMSegmentation training/testing entry points.

Train:
```bash
python tools/train.py configs/dart/your_config.py
```

Test/evaluate:
```bash
python tools/test.py configs/dart/your_config.py work_dirs/your_config/latest.pth --eval mIoU
```

Inference demo (example):
```bash
python demo/image_demo.py demo/demo.png configs/dart/your_config.py work_dirs/your_config/latest.pth --device cuda:0
```

### Acknowledgement
- Built on top of [MMSegmentation](https://mmsegmentation.readthedocs.io/en/main/). 