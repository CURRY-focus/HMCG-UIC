import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image


CLASS_ORDER = ["RO", "FV", "HD", "RI", "WR"]


def image_files(folder):
    suffixes = {".png", ".jpg", ".jpeg", ".bmp"}
    return sorted([path for path in Path(folder).glob("*") if path.suffix.lower() in suffixes])


def patch_legacy_keras_api(tf):
    """Make the old SUIM Keras-1/2 style imports work with tf.keras."""
    import types
    import keras
    import keras.layers
    import keras.models
    import keras.optimizers

    keras.models.Input = keras.layers.Input

    def model_compat(*args, **kwargs):
        if "input" in kwargs and "inputs" not in kwargs:
            kwargs["inputs"] = kwargs.pop("input")
        if "output" in kwargs and "outputs" not in kwargs:
            kwargs["outputs"] = kwargs.pop("output")
        return tf.keras.Model(*args, **kwargs)

    keras.models.Model = model_compat

    conv_module = types.ModuleType("keras.layers.convolutional")
    conv_module.Conv2D = keras.layers.Conv2D
    conv_module.UpSampling2D = keras.layers.UpSampling2D
    sys.modules["keras.layers.convolutional"] = conv_module

    class AdamCompat(tf.keras.optimizers.Adam):
        def __init__(self, *args, **kwargs):
            if "lr" in kwargs and "learning_rate" not in kwargs:
                kwargs["learning_rate"] = kwargs.pop("lr")
            super().__init__(*args, **kwargs)

    class SGDCompat(tf.keras.optimizers.SGD):
        def __init__(self, *args, **kwargs):
            if "lr" in kwargs and "learning_rate" not in kwargs:
                kwargs["learning_rate"] = kwargs.pop("lr")
            super().__init__(*args, **kwargs)

    keras.optimizers.Adam = AdamCompat
    keras.optimizers.SGD = SGDCompat


def find_suim_code_dir(root):
    root = Path(root)
    if (root / "models" / "suim_net.py").is_file():
        return root
    for path in root.rglob("suim_net.py"):
        if path.parent.name == "models":
            return path.parent.parent
    raise FileNotFoundError(f"Could not find models/suim_net.py under {root}")


def load_gt_mask(mask_dir, class_name, stem, size):
    class_dir = Path(mask_dir) / class_name
    candidates = []
    for suffix in (".bmp", ".png", ".jpg", ".jpeg"):
        candidates.append(class_dir / f"{stem}{suffix}")
    mask_path = next((path for path in candidates if path.is_file()), None)
    if mask_path is None:
        raise FileNotFoundError(f"Missing SUIM {class_name} mask for {stem}")
    mask = Image.open(mask_path).convert("L").resize(size, Image.NEAREST)
    arr = np.asarray(mask, dtype=np.float32) / 255.0
    return arr >= 0.5


def load_image(path, size):
    image = Image.open(path).convert("RGB").resize(size, Image.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def binary_metrics(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = np.logical_and(pred, gt).sum(dtype=np.float64)
    fp = np.logical_and(pred, np.logical_not(gt)).sum(dtype=np.float64)
    fn = np.logical_and(np.logical_not(pred), gt).sum(dtype=np.float64)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    return precision, recall, f1, iou


def build_maps(pred_probs, gt_masks, threshold):
    pred_max = pred_probs.max(axis=-1)
    pred_cls = pred_probs.argmax(axis=-1) + 1
    pred_map = np.where(pred_max >= threshold, pred_cls, 0)

    gt_stack = np.stack(gt_masks, axis=-1)
    gt_has_class = gt_stack.any(axis=-1)
    gt_cls = gt_stack.argmax(axis=-1) + 1
    gt_map = np.where(gt_has_class, gt_cls, 0)
    return pred_map, gt_map


def main(args):
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError("TensorFlow/Keras is required for SUIM .hdf5 checkpoints.") from exc

    patch_legacy_keras_api(tf)

    code_dir = find_suim_code_dir(args.suim_code_dir)
    sys.path.insert(0, str(code_dir))
    from models.suim_net import SUIM_Net

    if args.base.upper() == "RSB":
        im_size = (320, 240)
    elif args.base.upper() == "VGG":
        im_size = (320, 256)
    else:
        raise ValueError("--base must be RSB or VGG")

    tf.get_logger().setLevel("ERROR")
    model = SUIM_Net(base=args.base.upper(), im_res=(im_size[0], im_size[1], 3), n_classes=5).model
    model.load_weights(args.checkpoint)

    pred_root = Path(args.pred_dir) if args.pred_dir else None
    if pred_root:
        for class_name in CLASS_ORDER:
            (pred_root / class_name).mkdir(parents=True, exist_ok=True)

    totals = {class_name: {"f1": [], "iou": []} for class_name in CLASS_ORDER}
    pa_correct = 0
    pa_total = 0
    rows = []

    for image_path in image_files(args.recon_dir):
        image = load_image(image_path, im_size)
        pred = model.predict(np.expand_dims(image, axis=0), verbose=0)[0]
        pred_binary = pred >= args.threshold

        gt_masks = [
            load_gt_mask(args.mask_dir, class_name, image_path.stem, im_size)
            for class_name in CLASS_ORDER
        ]

        row = {"image": image_path.name}
        for idx, class_name in enumerate(CLASS_ORDER):
            gt = gt_masks[idx]
            pred_cls = pred_binary[:, :, idx]
            if gt.sum() > 0:
                _, _, f1, iou = binary_metrics(pred_cls, gt)
                totals[class_name]["f1"].append(float(f1))
                totals[class_name]["iou"].append(float(iou))
                row[f"{class_name}_f1"] = float(f1)
                row[f"{class_name}_iou"] = float(iou)
            else:
                row[f"{class_name}_f1"] = ""
                row[f"{class_name}_iou"] = ""

            if pred_root:
                out = (pred_cls.astype(np.uint8) * 255)
                Image.fromarray(out).save(pred_root / class_name / f"{image_path.stem}.png")

        pred_map, gt_map = build_maps(pred, gt_masks, args.threshold)
        pa_correct += np.equal(pred_map, gt_map).sum(dtype=np.float64)
        pa_total += gt_map.size
        rows.append(row)

    summary = {}
    for class_name in CLASS_ORDER:
        summary[f"{class_name}_F1"] = float(np.mean(totals[class_name]["f1"])) if totals[class_name]["f1"] else 0.0
        summary[f"{class_name}_IoU"] = float(np.mean(totals[class_name]["iou"])) if totals[class_name]["iou"] else 0.0
    summary["mean_F1"] = float(np.mean([summary[f"{class_name}_F1"] for class_name in CLASS_ORDER]))
    summary["mIoU"] = float(np.mean([summary[f"{class_name}_IoU"] for class_name in CLASS_ORDER]))
    summary["PA"] = float(pa_correct / max(pa_total, 1))

    print("[SUIM Semantic Segmentation]")
    for key, value in summary.items():
        print(f"{key}: {value:.6f}")

    if args.output_csv:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["image"]
        for class_name in CLASS_ORDER:
            fieldnames.extend([f"{class_name}_f1", f"{class_name}_iou"])
        for class_name in CLASS_ORDER:
            fieldnames.extend([f"{class_name}_F1", f"{class_name}_IoU"])
        fieldnames.extend(["mean_F1", "mIoU", "PA"])
        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            writer.writerow({"image": "summary", **summary})


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recon_dir", required=True)
    parser.add_argument("--mask_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--suim_code_dir", required=True)
    parser.add_argument("--base", default="RSB", choices=["RSB", "VGG"])
    parser.add_argument("--threshold", default=0.5, type=float)
    parser.add_argument("--pred_dir", default=None)
    parser.add_argument("--output_csv", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
