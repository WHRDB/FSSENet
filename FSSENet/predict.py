import argparse
from pathlib import Path

import cv2
import torch
from mmengine.dataset import Compose, pseudo_collate
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from opencd.registry import MODELS

from opencd_custom.runtime import clear_initialization, load_config, require_file


def main():
    parser = argparse.ArgumentParser(
        description="Predict a binary change mask for one aligned image pair."
    )
    parser.add_argument("image_a")
    parser.add_argument("image_b")
    parser.add_argument("checkpoint")
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--output",
        required=True,
        help="Output PNG; pixels are 0 (unchanged) or 255 (changed).",
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    image_a, image_b, checkpoint = (
        require_file(x, label)
        for x, label in [
            (args.image_a, "Image A"),
            (args.image_b, "Image B"),
            (args.checkpoint, "Checkpoint"),
        ]
    )
    shape_a = cv2.imread(image_a)
    shape_b = cv2.imread(image_b)
    if shape_a is None or shape_b is None or shape_a.shape != shape_b.shape:
        raise ValueError(
            "The two images must be readable and have identical dimensions."
        )
    output = Path(args.output).expanduser().resolve()
    if output.suffix.lower() != ".png":
        parser.error("--output must be a PNG path.")
    if output in (Path(image_a), Path(image_b), Path(checkpoint)):
        parser.error("--output must differ from the input paths.")
    cfg = load_config(args.config)
    clear_initialization(cfg.model)
    init_default_scope("opencd")
    model = MODELS.build(cfg.model)
    load_checkpoint(model, checkpoint, map_location="cpu", strict=True)
    model.to(args.device).eval()
    pipeline = Compose(
        [
            step
            for step in cfg.test_pipeline
            if step["type"] != "MultiImgLoadAnnotations"
        ]
    )
    batch = pseudo_collate([pipeline(dict(img_path=[image_a, image_b]))])
    with torch.inference_mode():
        result = model.test_step(batch)[0]
    mask = result.pred_sem_seg.data.squeeze(0).cpu().numpy().astype("uint8") * 255
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), mask):
        raise OSError(f"Could not write mask to {output}")
    print(f"Saved {mask.shape[1]} x {mask.shape[0]} mask: {output}")


if __name__ == "__main__":
    main()
