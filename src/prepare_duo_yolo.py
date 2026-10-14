import argparse
import json
import os
import shutil
from pathlib import Path


def link_or_copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(dst):
        dst.unlink(missing_ok=True)
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def load_coco(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def convert_split(duo_root, output_root, split, categories):
    coco_path = duo_root / "annotations" / f"instances_{split}.json"
    image_dir = duo_root / "images" / split
    coco = load_coco(coco_path)

    cat_to_yolo = {cat["id"]: idx for idx, cat in enumerate(categories)}
    images = {image["id"]: image for image in coco["images"]}
    labels = {image_id: [] for image_id in images}

    for ann in coco["annotations"]:
        image = images[ann["image_id"]]
        x, y, w, h = ann["bbox"]
        if w <= 0 or h <= 0:
            continue
        x_center = (x + 0.5 * w) / image["width"]
        y_center = (y + 0.5 * h) / image["height"]
        width = w / image["width"]
        height = h / image["height"]
        cls = cat_to_yolo[ann["category_id"]]
        labels[ann["image_id"]].append(
            f"{cls} {x_center:.8f} {y_center:.8f} {width:.8f} {height:.8f}"
        )

    for image in images.values():
        src_image = image_dir / image["file_name"]
        dst_image = output_root / "images" / split / image["file_name"]
        link_or_copy(src_image, dst_image)

        label_path = output_root / "labels" / split / f"{Path(image['file_name']).stem}.txt"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("\n".join(labels[image["id"]]) + ("\n" if labels[image["id"]] else ""), encoding="utf-8")


def write_yaml(output_root, categories):
    names = [cat["name"] for cat in categories]
    yaml_path = output_root / "data.yaml"
    with yaml_path.open("w", encoding="utf-8") as f:
        f.write(f"path: {output_root}\n")
        f.write("train: images/train\n")
        f.write("val: images/test\n")
        f.write("test: images/test\n")
        f.write("names:\n")
        for idx, name in enumerate(names):
            f.write(f"  {idx}: {name}\n")
    return yaml_path


def main(args):
    duo_root = Path(args.duo_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    train_coco = load_coco(duo_root / "annotations" / "instances_train.json")
    categories = sorted(train_coco["categories"], key=lambda cat: cat["id"])

    convert_split(duo_root, output_root, "train", categories)
    convert_split(duo_root, output_root, "test", categories)
    yaml_path = write_yaml(output_root, categories)
    print(f"[DUO] YOLO dataset written to {output_root}")
    print(f"[DUO] data.yaml: {yaml_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duo_root", required=True)
    parser.add_argument("--output_root", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
