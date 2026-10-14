"""Evaluate reconstructed images with the manuscript's independent U2-Net protocol."""

from __future__ import annotations

import argparse
import csv
import importlib
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def image_files(folder: str | Path) -> list[Path]:
    return sorted(
        path for path in Path(folder).iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def load_mask(path: Path) -> np.ndarray:
    mask = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    return np.clip(mask, 0.0, 1.0)


def flatten_tensors(output) -> list[torch.Tensor]:
    if torch.is_tensor(output):
        return [output]
    if isinstance(output, dict):
        tensors = []
        for value in output.values():
            tensors.extend(flatten_tensors(value))
        return tensors
    if isinstance(output, (list, tuple)):
        tensors = []
        for value in output:
            tensors.extend(flatten_tensors(value))
        return tensors
    return []


def pick_saliency(output) -> torch.Tensor:
    tensors = [tensor for tensor in flatten_tensors(output) if tensor.ndim == 4]
    if not tensors:
        raise RuntimeError("U2-Net did not return a four-dimensional saliency map.")
    one_channel = [tensor for tensor in tensors if tensor.shape[1] == 1]
    prediction = one_channel[0] if one_channel else tensors[0].mean(dim=1, keepdim=True)
    if prediction.min() < 0.0 or prediction.max() > 1.0:
        prediction = torch.sigmoid(prediction)
    return prediction


def normalize_prediction(prediction: torch.Tensor, size_hw: tuple[int, int]) -> np.ndarray:
    prediction = F.interpolate(
        prediction,
        size=size_hw,
        mode="bilinear",
        align_corners=False,
    )
    array = prediction[0, 0].detach().cpu().numpy().astype(np.float32)
    array = np.clip(array, 0.0, 1.0)
    low, high = float(array.min()), float(array.max())
    if high > low:
        array = (array - low) / (high - low)
    return array


def object_score(prediction: np.ndarray) -> float:
    if prediction.size == 0:
        return 0.0
    mean = float(prediction.mean())
    std = float(prediction.std())
    return 2.0 * mean / (mean * mean + 1.0 + std + 1e-8)


def ssim_score(prediction: np.ndarray, target: np.ndarray) -> float:
    if prediction.size == 0:
        return 0.0
    x = prediction.astype(np.float64)
    y = target.astype(np.float64)
    mean_x, mean_y = x.mean(), y.mean()
    var_x, var_y = x.var(), y.var()
    covariance = ((x - mean_x) * (y - mean_y)).mean()
    c1, c2 = 0.01, 0.03
    score = ((2 * mean_x * mean_y + c1) * (2 * covariance + c2)) / (
        (mean_x**2 + mean_y**2 + c1) * (var_x + var_y + c2)
    )
    return float(np.clip(score, -1.0, 1.0))


def centroid(target: np.ndarray) -> tuple[int, int]:
    height, width = target.shape
    area = target.sum()
    if area <= 0:
        return width // 2, height // 2
    ys, xs = np.indices(target.shape)
    center_x = int(np.round((xs * target).sum() / area))
    center_y = int(np.round((ys * target).sum() / area))
    return min(max(center_x, 1), width - 1), min(max(center_y, 1), height - 1)


def region_score(prediction: np.ndarray, target: np.ndarray) -> float:
    height, width = target.shape
    center_x, center_y = centroid(target)
    regions = [
        (slice(0, center_y), slice(0, center_x)),
        (slice(0, center_y), slice(center_x, width)),
        (slice(center_y, height), slice(0, center_x)),
        (slice(center_y, height), slice(center_x, width)),
    ]
    score = 0.0
    area = float(height * width)
    for ys, xs in regions:
        weight = ((ys.stop - ys.start) * (xs.stop - xs.start)) / area
        score += weight * ssim_score(prediction[ys, xs], target[ys, xs])
    return score


def s_measure(prediction: np.ndarray, target: np.ndarray) -> float:
    target_mean = float(target.mean())
    if target_mean <= 1e-8:
        return 1.0 - float(prediction.mean())
    if target_mean >= 1.0 - 1e-8:
        return float(prediction.mean())
    foreground = object_score(prediction[target > 0.5])
    background = object_score(1.0 - prediction[target <= 0.5])
    object_part = target_mean * foreground + (1.0 - target_mean) * background
    return 0.5 * object_part + 0.5 * region_score(prediction, target)


def max_f_measure(prediction: np.ndarray, target: np.ndarray, beta2: float = 0.3) -> float:
    target_positive = target > 0.5
    target_count = float(target_positive.sum())
    best = 0.0
    for threshold in np.linspace(0.0, 1.0, 256):
        prediction_positive = prediction >= threshold
        true_positive = float(np.logical_and(prediction_positive, target_positive).sum())
        precision = true_positive / (float(prediction_positive.sum()) + 1e-8)
        recall = true_positive / (target_count + 1e-8)
        score = (1.0 + beta2) * precision * recall / (
            beta2 * precision + recall + 1e-8
        )
        best = max(best, score)
    return best


def load_u2net(args, device: torch.device) -> torch.nn.Module:
    repo_root = Path(args.repo_root).resolve()
    if not repo_root.is_dir():
        raise FileNotFoundError(f"U2-Net repository not found: {repo_root}")
    sys.path.insert(0, str(repo_root))
    module = importlib.import_module(args.model_module)
    model_class = getattr(module, args.model_class)
    try:
        model = model_class(3, 1)
    except TypeError:
        model = model_class()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise TypeError("U2-Net checkpoint must contain a state dictionary.")
    checkpoint = {
        key.removeprefix("module."): value for key, value in checkpoint.items()
    }
    model.load_state_dict(checkpoint, strict=True)
    return model.to(device).eval()


def evaluate(args) -> None:
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = load_u2net(args, device)
    transform = transforms.Compose(
        [
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    reconstructions = image_files(args.recon_dir)
    masks = {path.stem: path for path in image_files(args.mask_dir)}
    if not reconstructions:
        raise FileNotFoundError(f"No reconstructed images found in {args.recon_dir}")

    prediction_dir = Path(args.pred_dir) if args.pred_dir else None
    if prediction_dir:
        prediction_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with torch.inference_mode():
        for reconstruction_path in reconstructions:
            mask_path = masks.get(reconstruction_path.stem)
            if mask_path is None:
                raise FileNotFoundError(
                    f"Missing saliency mask for {reconstruction_path.name}"
                )
            mask = load_mask(mask_path)
            target = (mask > 0.5).astype(np.float32)
            image = transform(Image.open(reconstruction_path).convert("RGB"))
            output = model(image.unsqueeze(0).to(device))
            prediction = normalize_prediction(pick_saliency(output), mask.shape)

            if prediction_dir:
                Image.fromarray((prediction * 255.0).astype(np.uint8)).save(
                    prediction_dir / f"{reconstruction_path.stem}.png"
                )

            row = {
                "image": reconstruction_path.name,
                "Smeasure": s_measure(prediction, target),
                "Fmeasure": max_f_measure(prediction, target),
                "MAE": float(np.mean(np.abs(prediction - mask))),
            }
            rows.append(row)
            print(
                f"{row['image']}: S={row['Smeasure']:.6f}, "
                f"F={row['Fmeasure']:.6f}, MAE={row['MAE']:.6f}"
            )

    summary = {
        "Smeasure": float(np.mean([row["Smeasure"] for row in rows])),
        "Fmeasure": float(np.mean([row["Fmeasure"] for row in rows])),
        "MAE": float(np.mean([row["MAE"] for row in rows])),
    }
    print("\n[U2-Net saliency]")
    for name, value in summary.items():
        print(f"{name}: {value:.6f}")

    if args.output_csv:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["image", "Smeasure", "Fmeasure", "MAE"],
            )
            writer.writeheader()
            writer.writerows(rows)
            writer.writerow({"image": "average", **summary})


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate reconstructions with the manuscript U2-Net protocol."
    )
    parser.add_argument("--recon_dir", required=True)
    parser.add_argument("--mask_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--repo_root", required=True)
    parser.add_argument("--model_module", default="model.u2net")
    parser.add_argument("--model_class", default="U2NET")
    parser.add_argument("--input_size", type=int, default=320)
    parser.add_argument("--device", default=None)
    parser.add_argument("--pred_dir", default=None)
    parser.add_argument("--output_csv", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
