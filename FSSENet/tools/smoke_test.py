import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from mmengine.registry import init_default_scope
from mmengine.structures import PixelData
from mmseg.structures import SegDataSample
from opencd.registry import MODELS

from opencd_custom.runtime import clear_initialization, load_config


def main():
    parser = argparse.ArgumentParser(
        description="Check full-size FSSENet forward, backward and prediction without weights."
    )
    parser.add_argument("--config")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.manual_seed(123)
    torch.set_num_threads(4)
    cfg = load_config(args.config)
    clear_initialization(cfg.model)
    init_default_scope("opencd")
    model = MODELS.build(cfg.model).to(args.device)
    model.train()
    inputs = torch.randn(1, 6, 512, 512, device=args.device)
    sample = SegDataSample(
        metainfo=dict(
            ori_shape=(512, 512),
            img_shape=(512, 512),
            pad_shape=(512, 512),
            padding_size=(0, 0, 0, 0),
        )
    )
    sample.gt_sem_seg = PixelData(
        data=torch.randint(0, 2, (1, 512, 512), device=args.device)
    )
    losses = model(inputs, [sample], mode="loss")
    total = sum(v for k, v in losses.items() if "loss" in k)
    if not torch.isfinite(total):
        raise RuntimeError("Nonfinite loss")
    total.backward()
    gradients = [
        p.grad for p in model.parameters() if p.requires_grad and p.grad is not None
    ]
    if not gradients or not all(torch.isfinite(g).all() for g in gradients):
        raise RuntimeError("Missing or nonfinite gradients")
    if not any(torch.count_nonzero(g).item() for g in gradients):
        raise RuntimeError("All gradients are zero")
    for branch in (
        "side_adapter_network.side_encoder",
        "side_adapter_network.clip_attns",
        "mask_decoder.SMSA_RGAs_add22_PCSA_layer1",
        "mask_decoder.conv_seg",
    ):
        params = [
            p for n, p in model.decode_head.named_parameters() if n.startswith(branch)
        ]
        if not params or not any(
            p.grad is not None and torch.count_nonzero(p.grad).item() for p in params
        ):
            raise RuntimeError(f"No gradient reached {branch}")
    if any(
        p.requires_grad or p.grad is not None for p in model.image_encoder.parameters()
    ):
        raise RuntimeError(
            "The supplied configuration must freeze the foundation encoder"
        )
    model.zero_grad(set_to_none=True)
    model.eval()
    with torch.inference_mode():
        result = model(inputs, [sample], mode="predict")[0]
    mask = result.pred_sem_seg.data
    if tuple(mask.shape) != (1, 512, 512) or not set(mask.unique().tolist()) <= {0, 1}:
        raise RuntimeError("Invalid prediction shape or labels")
    print(
        json.dumps(
            dict(
                status="PASS",
                loss=float(total.detach()),
                gradient_tensors=len(gradients),
                parameters=sum(p.numel() for p in model.parameters()),
                prediction_shape=list(mask.shape),
                note="Random parameters: this is a runtime check, not an accuracy measurement.",
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
