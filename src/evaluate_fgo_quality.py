import argparse
import csv
from pathlib import Path

import numpy as np
import pyiqa
import torch
import tqdm
from PIL import Image
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity
from torchvision.transforms import ToTensor


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def image_files(root):
    root = Path(root)
    return sorted(
        [path for path in root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES],
        key=lambda path: path.stem,
    )


def image_map(root):
    return {path.name: path for path in image_files(root)}


def load_rgb(path):
    with Image.open(path) as image:
        return image.convert("RGB")


def load_mask(path, size, threshold):
    with Image.open(path) as image:
        mask = image.convert("L")
    if mask.size != size:
        mask = mask.resize(size, Image.NEAREST)
    mask_np = np.asarray(mask, dtype=np.float32) / 255.0
    return (mask_np > threshold).astype(np.float32)


def apply_foreground_mask(image, mask_np):
    image_np = np.asarray(image, dtype=np.float32) / 255.0
    image_np = image_np * mask_np[:, :, None]
    image_np = np.clip(image_np * 255.0, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(image_np, mode="RGB")


def evaluate(args):
    recon_dir = Path(args.recon_dir)
    gt_dir = Path(args.gt_dir)
    mask_dir = Path(args.mask_dir)
    output_csv = Path(args.output_csv)

    recon_paths = image_files(recon_dir)
    gt_paths = image_map(gt_dir)
    mask_paths = image_map(mask_dir)
    if args.max_images is not None:
        recon_paths = recon_paths[: args.max_images]

    missing = []
    for path in recon_paths:
        if path.name not in gt_paths:
            missing.append(f"GT:{path.name}")
        if path.name not in mask_paths:
            missing.append(f"MASK:{path.name}")
    if missing:
        raise FileNotFoundError("Missing paired files: " + ", ".join(missing[:20]))

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    to_tensor = ToTensor()
    psnr_metric = pyiqa.create_metric("psnr").to(device)
    ms_ssim_metric = pyiqa.create_metric("ms_ssim").to(device)
    dists_metric = pyiqa.create_metric("dists").to(device)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)

    rows = []
    sums = {
        "fgo_psnr": 0.0,
        "fgo_ms_ssim": 0.0,
        "fgo_dists": 0.0,
        "fgo_lpips": 0.0,
        "foreground_ratio": 0.0,
    }

    print(f"Find {len(recon_paths)} images in {recon_dir}")
    for recon_path in tqdm.tqdm(recon_paths):
        gt_path = gt_paths[recon_path.name]
        mask_path = mask_paths[recon_path.name]

        recon = load_rgb(recon_path)
        gt = load_rgb(gt_path)
        if recon.size != gt.size:
            raise ValueError(f"Size mismatch for {recon_path.name}: recon={recon.size}, gt={gt.size}")

        mask_np = load_mask(mask_path, gt.size, args.mask_threshold)
        masked_recon = apply_foreground_mask(recon, mask_np)
        masked_gt = apply_foreground_mask(gt, mask_np)

        recon_tensor = to_tensor(masked_recon).unsqueeze(0).to(device)
        gt_tensor = to_tensor(masked_gt).unsqueeze(0).to(device)

        with torch.no_grad():
            fgo_psnr = float(psnr_metric(recon_tensor, gt_tensor).item())
            fgo_ms_ssim = float(ms_ssim_metric(recon_tensor, gt_tensor).item())
            fgo_dists = float(dists_metric(recon_tensor, gt_tensor).item())
            fgo_lpips = float(lpips_metric(recon_tensor, gt_tensor).item())

        foreground_ratio = float(mask_np.mean())
        row = {
            "image": recon_path.name,
            "fgo_psnr": f"{fgo_psnr:.6f}",
            "fgo_ms_ssim": f"{fgo_ms_ssim:.6f}",
            "fgo_dists": f"{fgo_dists:.6f}",
            "fgo_lpips": f"{fgo_lpips:.6f}",
            "foreground_ratio": f"{foreground_ratio:.8f}",
        }
        rows.append(row)
        for key in sums:
            sums[key] += float(row[key])

    count = max(len(rows), 1)
    averages = {key: sums[key] / count for key in sums}

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["image", "fgo_psnr", "fgo_ms_ssim", "fgo_dists", "fgo_lpips", "foreground_ratio"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow(
            {
                "image": "average",
                "fgo_psnr": f"{averages['fgo_psnr']:.6f}",
                "fgo_ms_ssim": f"{averages['fgo_ms_ssim']:.6f}",
                "fgo_dists": f"{averages['fgo_dists']:.6f}",
                "fgo_lpips": f"{averages['fgo_lpips']:.6f}",
                "foreground_ratio": f"{averages['foreground_ratio']:.8f}",
            }
        )

    print(f"fgo_psnr: {averages['fgo_psnr']:.6f}")
    print(f"fgo_ms_ssim: {averages['fgo_ms_ssim']:.6f}")
    print(f"fgo_dists: {averages['fgo_dists']:.6f}")
    print(f"fgo_lpips: {averages['fgo_lpips']:.6f}")
    print(f"foreground_ratio: {averages['foreground_ratio']:.8f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate quality on USOD foreground-object regions.")
    parser.add_argument("--recon_dir", required=True)
    parser.add_argument("--gt_dir", required=True, help="Original RGB image directory.")
    parser.add_argument("--mask_dir", required=True, help="USOD foreground mask directory.")
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--mask_threshold", type=float, default=0.5)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main():
    evaluate(parse_args())


if __name__ == "__main__":
    main()
