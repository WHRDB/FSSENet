import argparse
import os

from mmengine.config import DictAction
from mmengine.runner import Runner, load_checkpoint

from opencd_custom.runtime import clear_initialization, load_config, require_file


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained FSSENet checkpoint."
    )
    parser.add_argument("config", nargs="?", default=None)
    parser.add_argument("checkpoint", nargs="?", default=None)
    parser.add_argument("--config", dest="config_option")
    parser.add_argument("--checkpoint", dest="checkpoint_option")
    parser.add_argument("--data-root")
    parser.add_argument("--work-dir")
    parser.add_argument(
        "--show-dir", help="Directory for predicted mask visualizations."
    )
    parser.add_argument("--cfg-options", nargs="+", action=DictAction)
    parser.add_argument(
        "--launcher", choices=["none", "pytorch", "slurm", "mpi"], default="none"
    )
    parser.add_argument("--local-rank", "--local_rank", type=int, default=0)
    args = parser.parse_args()
    os.environ.setdefault("LOCAL_RANK", str(args.local_rank))
    checkpoint = args.checkpoint_option or args.checkpoint
    if checkpoint is None:
        parser.error(
            "A trained checkpoint is required: supply a positional path or --checkpoint."
        )
    checkpoint = require_file(checkpoint)
    cfg = load_config(
        args.config_option or args.config,
        args.cfg_options,
        args.data_root,
        args.work_dir,
    )
    cfg.launcher = args.launcher
    cfg.load_from = None
    cfg.resume = False
    clear_initialization(cfg.model)
    if args.show_dir:
        cfg.default_hooks.visualization.draw = True
        cfg.visualizer.save_dir = os.path.abspath(args.show_dir)
    runner = Runner.from_cfg(cfg)
    load_checkpoint(runner.model, checkpoint, map_location="cpu", strict=True)
    runner.test()


if __name__ == "__main__":
    main()
