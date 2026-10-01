custom_imports = {"imports": ["opencd_custom"], "allow_failed_imports": False}
default_scope = "opencd"
crop_size = (512, 512)
data_root = "data/WaterCD"
dataset_type = "FSSENetDataset"
model = {
    "asymetric_input": True,
    "data_preprocessor": {
        "bgr_to_rgb": True,
        "mean": [122.7709, 116.746, 104.0937, 122.7709, 116.746, 104.0937],
        "pad_val": 0,
        "seg_pad_val": 255,
        "std": [68.5005, 66.6322, 70.3232, 68.5005, 66.6322, 70.3232],
        "test_cfg": {"size": (512, 512)},
        "type": "DualInputSegDataPreProcessor",
        "size": (512, 512),
    },
    "decode_head": {
        "ban_cfg": {
            "clip_channels": 1024,
            "fusion_index": [3],
            "side_enc_cfg": {
                "attn_drop_rate": 0.0,
                "drop_path_rate": 0.1,
                "drop_rate": 0.0,
                "embed_dims": 32,
                "in_channels": 3,
                "init_cfg": {
                    "checkpoint": "pretrained/mit_b0_20220624-7e0fe6dd.pth",
                    "type": "Pretrained",
                },
                "mlp_ratio": 4,
                "num_heads": [1, 2, 5, 8],
                "num_layers": [2, 2, 2, 2],
                "num_stages": 4,
                "out_indices": (0, 1, 2, 3),
                "patch_sizes": [7, 3, 3, 3],
                "qkv_bias": True,
                "sr_ratios": [8, 4, 2, 1],
                "type": "mmseg.MixVisionTransformer",
            },
        },
        "ban_dec_cfg": {
            "align_corners": False,
            "channels": 128,
            "dropout_ratio": 0.1,
            "in_channels": [32, 64, 160, 256],
            "norm_cfg": {"requires_grad": True, "type": "SyncBN"},
            "num_classes": 2,
            "type": "BAN_MLPDecoder",
        },
        "loss_decode": {
            "loss_weight": 1.0,
            "type": "mmseg.CrossEntropyLoss",
            "use_sigmoid": False,
        },
        "type": "BitemporalAdapterHead",
    },
    "encoder_resolution": {"mode": "bilinear", "size": (336, 336)},
    "image_encoder": {
        "act_cfg": {"type": "mmseg.QuickGELU"},
        "attn_drop_rate": 0.0,
        "drop_path_rate": 0.0,
        "drop_rate": 0.0,
        "embed_dims": 1024,
        "frozen_exclude": [],
        "img_size": (336, 336),
        "in_channels": 3,
        "interpolate_mode": "bicubic",
        "mlp_ratio": 4,
        "norm_cfg": {"eps": 1e-05, "type": "LN"},
        "norm_eval": False,
        "num_heads": 16,
        "num_layers": 18,
        "out_indices": (5, 11, 17),
        "out_origin": False,
        "output_cls_token": True,
        "patch_bias": False,
        "patch_pad": 0,
        "patch_size": 14,
        "pre_norm": True,
        "qkv_bias": True,
        "type": "mmseg.VisionTransformer",
        "with_cls_token": True,
    },
    "pretrained": "pretrained/clip_vit-large-patch14-336_3rdparty-0b5df9cb.pth",
    "test_cfg": {"crop_size": (512, 512), "mode": "slide", "stride": (256, 256)},
    "train_cfg": {},
    "type": "DualSiamEncoderDecoder",
}
train_pipeline = [
    {"type": "MultiImgLoadImageFromFile"},
    {"type": "MultiImgLoadAnnotations"},
    {"degree": 180, "prob": 0.5, "type": "MultiImgRandomRotate"},
    {"cat_max_ratio": 0.75, "crop_size": (512, 512), "type": "MultiImgRandomCrop"},
    {"direction": "horizontal", "prob": 0.5, "type": "MultiImgRandomFlip"},
    {"direction": "vertical", "prob": 0.5, "type": "MultiImgRandomFlip"},
    {
        "brightness_delta": 10,
        "contrast_range": (0.8, 1.2),
        "hue_delta": 10,
        "saturation_range": (0.8, 1.2),
        "type": "MultiImgPhotoMetricDistortion",
    },
    {"type": "MultiImgPackSegInputs"},
]
test_pipeline = [
    {"type": "MultiImgLoadImageFromFile"},
    {"keep_ratio": True, "scale": (512, 512), "type": "MultiImgResize"},
    {"type": "MultiImgLoadAnnotations"},
    {"type": "MultiImgPackSegInputs"},
]
train_dataloader = {
    "batch_size": 8,
    "dataset": {
        "data_prefix": {
            "img_path_from": "train/A",
            "img_path_to": "train/B",
            "seg_map_path": "train/label",
        },
        "data_root": data_root,
        "pipeline": train_pipeline,
        "type": dataset_type,
    },
    "num_workers": 8,
    "persistent_workers": True,
    "sampler": {"shuffle": True, "type": "InfiniteSampler"},
}
val_dataloader = {
    "batch_size": 1,
    "dataset": {
        "data_prefix": {
            "img_path_from": "test/A",
            "img_path_to": "test/B",
            "seg_map_path": "test/label",
        },
        "data_root": data_root,
        "pipeline": test_pipeline,
        "type": dataset_type,
        "test_mode": True,
    },
    "num_workers": 1,
    "persistent_workers": True,
    "sampler": {"shuffle": False, "type": "DefaultSampler"},
}
test_dataloader = {
    "batch_size": 1,
    "dataset": {
        "data_prefix": {
            "img_path_from": "test/A",
            "img_path_to": "test/B",
            "seg_map_path": "test/label",
        },
        "data_root": data_root,
        "pipeline": test_pipeline,
        "type": dataset_type,
        "test_mode": True,
    },
    "num_workers": 4,
    "persistent_workers": True,
    "sampler": {"shuffle": False, "type": "DefaultSampler"},
}
train_cfg = {"max_iters": 40000, "type": "IterBasedTrainLoop", "val_interval": 4000}
val_cfg = {"type": "ValLoop"}
test_cfg = {"type": "TestLoop"}
val_evaluator = {"iou_metrics": ["mFscore", "mIoU"], "type": "mmseg.IoUMetric"}
test_evaluator = {"iou_metrics": ["mFscore", "mIoU"], "type": "mmseg.IoUMetric"}
optim_wrapper = {
    "clip_grad": {"max_norm": 0.01, "norm_type": 2},
    "loss_scale": "dynamic",
    "optimizer": {
        "betas": (0.9, 0.999),
        "lr": 0.0001,
        "type": "AdamW",
        "weight_decay": 0.0001,
    },
    "paramwise_cfg": {
        "custom_keys": {
            "img_encoder": {"decay_mult": 1.0, "lr_mult": 0.1},
            "mask_decoder": {"lr_mult": 10.0},
            "norm": {"decay_mult": 0.0},
        }
    },
    "type": "AmpOptimWrapper",
}
param_scheduler = [
    {
        "begin": 0,
        "by_epoch": False,
        "end": 1000,
        "start_factor": 1e-06,
        "type": "LinearLR",
    },
    {
        "begin": 1000,
        "by_epoch": False,
        "end": 40000,
        "eta_min": 0.0,
        "power": 1.0,
        "type": "PolyLR",
    },
]
default_hooks = {
    "checkpoint": {
        "by_epoch": False,
        "interval": 4000,
        "save_best": "mIoU",
        "type": "CheckpointHook",
    },
    "logger": {"interval": 50, "log_metric_by_epoch": False, "type": "LoggerHook"},
    "param_scheduler": {"type": "ParamSchedulerHook"},
    "sampler_seed": {"type": "DistSamplerSeedHook"},
    "timer": {"type": "IterTimerHook"},
    "visualization": {"img_shape": None, "interval": 1, "type": "CDVisualizationHook"},
}
env_cfg = {
    "cudnn_benchmark": True,
    "dist_cfg": {"backend": "nccl"},
    "mp_cfg": {"mp_start_method": "fork", "opencv_num_threads": 0},
}
visualizer = {
    "alpha": 1.0,
    "name": "visualizer",
    "type": "CDLocalVisualizer",
    "vis_backends": [{"type": "CDLocalVisBackend"}],
}
log_processor = {"by_epoch": False}
log_level = "INFO"
load_from = None
resume = False
