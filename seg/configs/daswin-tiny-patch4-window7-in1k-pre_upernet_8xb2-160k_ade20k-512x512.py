_base_ = [
    '../_base_/models/upernet_swin.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
#custom_imports = dict(imports=['mmseg.models', 'mmseg.datasets', 'mmseg.visualization'], allow_failed_imports=False)
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
checkpoint_file = '/path/to/ckpt_transed.pth'  # noqa
model = dict(
    type='DAIRSegmentor',
    reshaper_cfg=dict(
        patch_size=16,
    ),
    data_preprocessor=data_preprocessor,
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        use_abs_pos_embed=False,
        drop_path_rate=0.3,
        patch_norm=True),
    decode_head=dict(in_channels=[96, 192, 384, 768], num_classes=150),
    auxiliary_head=dict(in_channels=384, num_classes=150),
    )
# test_cfg=dict(mode='slide', crop_size=(769, 769), stride=(513, 513))
# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False,
    )
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='Resize',
        scale=(2048, 512),
        keep_ratio=True), 
    dict(
        type='ResizeToMultiple',
    size_divisor=16, 
    ),
    dict(type='LoadAnnotations', reduce_zero_label=True), 
    dict(type='PackSegInputs')
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ], [dict(type='ResizeToMultiple',size_divisor=16),],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
        ])
]

# By default, models are trained on 8 GPUs with 2 images per GPU
train_dataloader = dict(batch_size=4)
val_dataloader = dict(batch_size=1, dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

auto_scale_lr = dict(base_batch_size=16, enable=True)