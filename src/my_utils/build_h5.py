import argparse
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def image_files(folder: Path) -> list[Path]:
    return sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def build_h5(train_dir: Path, output_file: Path) -> None:
    files = image_files(train_dir)
    if not files:
        raise FileNotFoundError(f"No training images found in {train_dir}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with h5py.File(output_file, "w") as handle:
        for path in tqdm(files, desc="images"):
            try:
                image = Image.open(path).convert("RGB")
                handle.create_dataset(str(written), data=np.asarray(image), dtype=np.uint8)
                written += 1
            except Exception as exc:
                print(f"[WARN] skipped {path.name}: {exc}")

    print(f"[DONE] wrote {written} images to {output_file}")


def parse_args():
    parser = argparse.ArgumentParser(description="Build the RGB training HDF5 file.")
    parser.add_argument("--train_dir", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_h5(args.train_dir, args.output_file)
