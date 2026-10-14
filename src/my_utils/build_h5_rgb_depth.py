import argparse
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm


def _is_relative_to(path, parent):
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def image_files(folder, exclude_dirs=None):
    exclude_dirs = [Path(path) for path in (exclude_dirs or [])]
    suffixes = {".png", ".jpg", ".jpeg", ".bmp"}
    paths = []
    for path in Path(folder).rglob("*"):
        if path.suffix.lower() not in suffixes:
            continue
        if any(_is_relative_to(path, exclude_dir) for exclude_dir in exclude_dirs):
            continue
        paths.append(path)
    return sorted(paths)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb_dir", required=True)
    parser.add_argument("--depth_dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rgb_paths = image_files(args.rgb_dir, exclude_dirs=[args.depth_dir])
    depth_paths = {path.stem: path for path in image_files(args.depth_dir)}
    missing = [path.name for path in rgb_paths if path.stem not in depth_paths]
    if missing:
        raise FileNotFoundError(f"Missing depth files for {len(missing)} RGB images, first missing: {missing[0]}")

    with h5py.File(args.output, "w") as h5_file:
        for idx, rgb_path in enumerate(tqdm(rgb_paths, desc="Building RGB-depth H5")):
            depth_path = depth_paths[rgb_path.stem]
            image = Image.open(rgb_path).convert("RGB")
            depth = Image.open(depth_path).convert("RGB")

            group = h5_file.create_group(str(idx))
            group.create_dataset("image", data=np.array(image), dtype=np.uint8)
            group.create_dataset("depth", data=np.array(depth), dtype=np.uint8)
            group.attrs["filename"] = rgb_path.name


if __name__ == "__main__":
    main()
