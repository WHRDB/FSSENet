import argparse
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Check paired PNG files, image sizes and binary labels."
    )
    parser.add_argument("data_root")
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument(
        "--labels-01", action="store_true", help="Expect 0/1 labels instead of 0/255."
    )
    args = parser.parse_args()
    root = Path(args.data_root).expanduser().resolve()
    allowed = {0, 1} if args.labels_01 else set(range(256))
    for split in args.splits:
        folders = [root / split / name for name in ("A", "B", "label")]
        names = [
            {p.relative_to(folder).as_posix() for p in folder.rglob("*.png")}
            for folder in folders
        ]
        if not names[0] or names[0] != names[1] or names[0] != names[2]:
            raise ValueError(
                f"{split}: A, B and label must contain the same nonempty set of PNG names; counts={list(map(len, names))}"
            )
        values = set()
        sizes = set()
        thresholded_files = 0
        for name in sorted(names[0]):
            a, b = (
                cv2.imread(str(folder / name), cv2.IMREAD_COLOR)
                for folder in folders[:2]
            )
            label = cv2.imread(str(folders[2] / name), cv2.IMREAD_UNCHANGED)
            if a is None or b is None or label is None:
                raise ValueError(f"{split}/{name}: unreadable image")
            if label.ndim != 2 or a.shape != b.shape or a.shape[:2] != label.shape:
                raise ValueError(
                    f"{split}/{name}: dimensions differ or label is not single-channel"
                )
            unique = set(np.unique(label).tolist())
            if not unique <= allowed:
                raise ValueError(
                    f"{split}/{name}: unexpected label values {unique}; expected {allowed}"
                )
            if not args.labels_01 and not unique <= {0, 255}:
                thresholded_files += 1
            values.update(unique)
            sizes.add(a.shape[:2])
        if not args.labels_01 and values == {0, 1}:
            raise ValueError(
                f"{split}: labels appear to be 0/1; use --labels-01 and set format_seg_map=None in the dataset config."
            )
        print(
            f"{split}: {len(names[0])} pairs OK; dimensions={sorted(sizes)}; labels={sorted(values)}; "
            f"non-0/255 masks converted at threshold 128={thresholded_files}"
        )


if __name__ == "__main__":
    main()
