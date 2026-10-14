import argparse
import json
import os
import shutil
from pathlib import Path

from ultralytics import YOLO


def image_files(folder):
    suffixes = {".png", ".jpg", ".jpeg", ".bmp"}
    return sorted([path for path in Path(folder).glob("*") if path.suffix.lower() in suffixes])


def link_or_copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def infer_names(labels_dir):
    max_cls = -1
    for label_path in Path(labels_dir).glob("*.txt"):
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            max_cls = max(max_cls, int(float(parts[0])))
    if max_cls < 0:
        return ["object"]
    return [str(idx) for idx in range(max_cls + 1)]


def load_names(args):
    if args.names:
        return [item.strip() for item in args.names.split(",") if item.strip()]

    if args.source_data_yaml:
        import yaml

        with open(args.source_data_yaml, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        names = data.get("names")
        if isinstance(names, dict):
            return [names[key] for key in sorted(names.keys(), key=lambda x: int(x))]
        if isinstance(names, list):
            return names

    return infer_names(args.labels_dir)


def write_data_yaml(path, dataset_root, names):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"path: {dataset_root}\n")
        f.write("train: images/val\n")
        f.write("val: images/val\n")
        f.write(f"names: {json.dumps(names, ensure_ascii=False)}\n")


def prepare_dataset(recon_dir, labels_dir, output_dir, names, max_images=None):
    dataset_root = Path(output_dir) / "yolo_eval_dataset"
    if dataset_root.exists():
        shutil.rmtree(dataset_root)
    image_out = dataset_root / "images" / "val"
    label_out = dataset_root / "labels" / "val"
    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)

    labels_by_stem = {path.stem: path for path in Path(labels_dir).glob("*.txt")}
    images = image_files(recon_dir)
    if max_images is not None:
        images = images[: max_images]
    for image_path in images:
        label_path = labels_by_stem.get(image_path.stem)
        if label_path is None:
            raise FileNotFoundError(f"Missing YOLO label for {image_path.name}")
        link_or_copy(image_path, image_out / image_path.name)
        link_or_copy(label_path, label_out / label_path.name)

    data_yaml = dataset_root / "data.yaml"
    write_data_yaml(data_yaml, dataset_root, names)
    return data_yaml


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    names = load_names(args)
    data_yaml = prepare_dataset(args.recon_dir, args.labels_dir, output_dir, names, args.max_images)

    model = YOLO(args.model_path)
    metrics = model.val(
        data=str(data_yaml),
        split="val",
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(output_dir),
        name="yolo_val",
        exist_ok=True,
    )

    summary = {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
    }
    for key, value in summary.items():
        print(f"{key}: {value:.6f}")

    with (output_dir / "detection_metrics.txt").open("w", encoding="utf-8") as f:
        for key, value in summary.items():
            f.write(f"{key}: {value:.6f}\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--recon_dir", required=True)
    parser.add_argument("--labels_dir", required=True, help="YOLO txt labels matched to recon images by filename stem.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--source_data_yaml", default=None, help="Optional original YOLO data.yaml, used only for class names.")
    parser.add_argument("--names", default=None, help="Optional comma-separated class names.")
    parser.add_argument("--imgsz", default=640, type=int)
    parser.add_argument("--batch", default=1, type=int)
    parser.add_argument("--max_images", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
