import argparse
import os
from pathlib import Path

from mmengine.config import DictAction
from mmengine.runner import Runner

from opencd_custom.runtime import (
    clear_initialization,
    configure_pretrained,
    load_config,
    require_file,
)


def main():
    parser = argparse.ArgumentParser(
        description="Train FSSENet on paired change detection images."
    )
    parser.add_argument("config", nargs="?", default=None)
    parser.add_argument("--config", dest="config_option")
    parser.add_argument("--data-root")
    parser.add_argument(
        "--pretrained", help="Converted CLIP ViT-L/14 initialization file."
    )
    parser.add_argument("--side-pretrained", help="MiT-B0 initialization file.")
    parser.add_argument("--work-dir")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        help="Resume a checkpoint or the latest run.",
    )
    parser.add_argument("--cfg-options", nargs="+", action=DictAction)
    parser.add_argument(
        "--launcher", choices=["none", "pytorch", "slurm", "mpi"], default="none"
    )
    parser.add_argument("--local-rank", "--local_rank", type=int, default=0)
    args = parser.parse_args()
    os.environ.setdefault("LOCAL_RANK", str(args.local_rank))
    cfg = load_config(
        args.config_option or args.config,
        args.cfg_options,
        args.data_root,
        args.work_dir,
    )
    cfg.launcher = args.launcher
    cfg.resume = bool(args.resume)
    if args.resume:
        if args.resume != "auto":
            cfg.load_from = require_file(args.resume)
        elif not (Path(cfg.work_dir) / "last_checkpoint").is_file():
            parser.error(
                "--resume found no last_checkpoint file in --work-dir; specify a checkpoint path."
            )
        clear_initialization(cfg.model)
    elif cfg.get("load_from"):
        cfg.load_from = require_file(cfg.load_from)
        clear_initialization(cfg.model)
    else:
        configure_pretrained(cfg, args.pretrained, args.side_pretrained)
    Runner.from_cfg(cfg).train()


if __name__ == "__main__":
    main()
