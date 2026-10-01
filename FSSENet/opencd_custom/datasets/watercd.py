from opencd.datasets.basecddataset import _BaseCDDataset
from opencd.registry import DATASETS


@DATASETS.register_module()
class FSSENetDataset(_BaseCDDataset):
    """Paired PNG images with binary labels: 0 unchanged, 255 changed."""

    METAINFO = dict(
        classes=("unchanged", "changed"), palette=[[0, 0, 0], [255, 255, 255]]
    )

    def __init__(
        self,
        img_suffix=".png",
        seg_map_suffix=".png",
        format_seg_map="to_binary",
        **kwargs
    ):
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            format_seg_map=format_seg_map,
            **kwargs
        )
