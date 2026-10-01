from pathlib import Path

from mmengine.config import Config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(value):
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else (PROJECT_ROOT / path).resolve())


def input_path(value):
    return str(Path(value).expanduser().resolve())


def clear_initialization(model_cfg):
    """A full checkpoint supplies both encoders and the decoder."""
    model_cfg.pretrained = None
    model_cfg.image_encoder.init_cfg = None
    model_cfg.decode_head.init_cfg = None
    model_cfg.decode_head.ban_cfg.side_enc_cfg.init_cfg = None


def load_config(config=None, options=None, data_root=None, work_dir=None):
    path = Path(config or "configs/fssenet_watercd.py").expanduser()
    if not path.is_absolute():
        path = path if path.is_file() else PROJECT_ROOT / path
    cfg = Config.fromfile(str(path.resolve()))
    if options:
        cfg.merge_from_dict(options)
    override = data_root or ((options or {}).get("data_root"))
    if override is not None:
        cfg.data_root = input_path(override)
    else:
        cfg.data_root = project_path(cfg.data_root)
    for name in ("train_dataloader", "val_dataloader", "test_dataloader"):
        loader = cfg.get(name)
        if loader is None:
            continue
        dataset = loader.dataset
        dataset.data_root = (
            cfg.data_root if override is not None else project_path(dataset.data_root)
        )
        if loader.get("num_workers", 0) == 0:
            loader.persistent_workers = False
    cfg.work_dir = (
        input_path(work_dir)
        if work_dir
        else project_path(cfg.get("work_dir") or f"work_dirs/{path.stem}")
    )
    return cfg


def configure_pretrained(cfg, pretrained=None, side_pretrained=None):
    if pretrained:
        cfg.model.pretrained = input_path(pretrained)
    if side_pretrained:
        cfg.model.decode_head.ban_cfg.side_enc_cfg.init_cfg = dict(
            type="Pretrained", checkpoint=input_path(side_pretrained)
        )
    if cfg.model.get("pretrained"):
        cfg.model.pretrained = project_path(cfg.model.pretrained)
        require_file(cfg.model.pretrained, "Foundation initialization (--pretrained)")
    elif cfg.model.image_encoder.get("frozen_exclude") == []:
        raise ValueError(
            "The foundation encoder is frozen: provide --pretrained. "
            "Use tools/smoke_test.py for a test with random parameters."
        )
    init_cfg = cfg.model.decode_head.ban_cfg.side_enc_cfg.get("init_cfg")
    if init_cfg and init_cfg.get("checkpoint"):
        init_cfg.checkpoint = project_path(init_cfg.checkpoint)
        require_file(
            init_cfg.checkpoint, "Side encoder initialization (--side-pretrained)"
        )


def require_file(filename, description="Checkpoint"):
    path = Path(filename).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{description} was not found: {path}")
    return str(path)
