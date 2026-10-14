import os
import sys
import math
import glob
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.abspath(os.path.dirname(__file__))
for path in (ROOT_DIR, SRC_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import diffusers
from diffusers.utils.import_utils import is_xformers_available

from my_utils.testing_utils import parse_args_testing
from hmcg_uic import HMCGUIC
from my_utils.compress_utils import *


def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image)
    return image_tensor


def compress_one_image(net, bin_path, ori_h, ori_w, img_name, x):
    with torch.no_grad():
        output_dict = net.compress(x)
    shape = output_dict["shape"]
    if not os.path.exists(bin_path): os.makedirs(bin_path)
    output = os.path.join(bin_path, img_name)
    with Path(output).open("wb") as f:
        write_body(f, shape, output_dict["strings"])
    size = filesize(output)
    bpp = float(size) * 8 / (ori_h * ori_w)
    names = output_dict.get("stream_names", [f"stream_{idx}" for idx in range(len(output_dict["strings"]))])
    payload_bytes = {
        name: len(stream[0])
        for name, stream in zip(names, output_dict["strings"])
    }
    main_bytes = payload_bytes.pop("main_latent", 0) + payload_bytes.pop("main_hyperprior", 0)
    container_bytes = max(0, size - main_bytes - sum(payload_bytes.values()))
    pixels = float(ori_h * ori_w)
    payload_bpp = (main_bytes + sum(payload_bytes.values())) * 8.0 / pixels
    actual_bpp = bpp
    stream_bpp = {
        "main_bpp": main_bytes * 8.0 / pixels,
        "structure_bpp": payload_bytes.get("structure", 0) * 8.0 / pixels,
        "color_bpp": payload_bytes.get("color", 0) * 8.0 / pixels,
        "semantic_bpp": payload_bytes.get("semantic", 0) * 8.0 / pixels,
        "payload_bpp": payload_bpp,
        "container_bpp": container_bytes * 8.0 / pixels,
        # The manuscript reports actual bpp from the complete on-disk
        # bitstream.  Keep total_bpp as a compatibility alias only.
        "actual_bpp": actual_bpp,
        "total_bpp": actual_bpp,
    }
    reconstructed_bpp = stream_bpp["payload_bpp"] + stream_bpp["container_bpp"]
    if not math.isclose(reconstructed_bpp, actual_bpp, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(
            f"Bit accounting mismatch: payload+container={reconstructed_bpp}, actual={actual_bpp}."
        )
    return actual_bpp, stream_bpp


def decompress_one_image(net, bin_path, ori_h, ori_w, img_name, prompt, return_aux=False):
    output = os.path.join(bin_path, img_name)
    with Path(output).open("rb") as f:
        strings, shape = read_body(f)
    with torch.no_grad():
        decoded = net.decompress(strings, shape, prompt, return_aux=return_aux)
    if return_aux:
        out_img, aux_outputs = decoded
    else:
        out_img = decoded
        aux_outputs = None
    out_img = out_img[:, :, 0: ori_h, 0: ori_w]
    out_img = (out_img * 0.5 + 0.5).float().cpu().detach()
    if aux_outputs is not None:
        machine_info = aux_outputs.get("machine")
        if machine_info is None or machine_info.get("feature") is None:
            raise RuntimeError(
                "F_hat_m was requested but D_M is inactive. Enable the structure branch."
            )
        machine_feature = machine_info["feature"][:, :, 0:ori_h, 0:ori_w]
        aux_outputs["machine"]["feature"] = (machine_feature * 0.5 + 0.5).float().cpu().detach()
    return out_img, aux_outputs


def main(args):
    if args.sd_path is None:
        from huggingface_hub import snapshot_download
        sd_path = snapshot_download(repo_id="stabilityai/sd-turbo")
    else:
        sd_path = args.sd_path

    if args.seed is not None:
        set_seed(args.seed)

    net = HMCGUIC(sd_path=sd_path, args=args)
    net.cuda().eval()
    net.codec.update(force=True)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            net.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    bpp = []
    stream_bpp_rows = []
    pos_tag_prompt = [1]
    images = []
    for pattern in ("*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.bmp", "*.BMP"):
        images.extend(glob.glob(os.path.join(args.img_path, pattern)))
    images = sorted(images)
    if args.max_images is not None:
        images = images[: args.max_images]
    print(f'\nFind {str(len(images))} images in {args.img_path}\n')

    for img_path in images:

        print('[Processing]', img_path)
        (path, name) = os.path.split(img_path)
        fname, ext = os.path.splitext(name)
        outf = os.path.join(args.rec_path, fname + '.png')

        img = preprocess_image(img_path, transform).cuda().unsqueeze(0)
        ori_h, ori_w = img.shape[2:]

        pad_h = (math.ceil(ori_h / 256)) * 256 - ori_h
        pad_w = (math.ceil(ori_w / 256)) * 256 - ori_w
        img_padded = F.pad(img, pad=(0, pad_w, 0, pad_h), mode='reflect')

        with torch.no_grad():
            try:
                rate, stream_rates = compress_one_image(net, args.bin_path, ori_h, ori_w, fname, img_padded)
                out_img, aux_outputs = decompress_one_image(
                    net,
                    args.bin_path,
                    ori_h,
                    ori_w,
                    fname,
                    pos_tag_prompt,
                    return_aux=args.machine_path is not None,
                )
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(str(name))
                    print("CUDA out of memory. Continuing to next image.")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise

        output_pil = transforms.ToPILImage()(out_img[0].clamp(0.0, 1.0))

        bpp.append(rate)
        stream_bpp_rows.append({"image": name, **stream_rates})
        print('[BPP]', rate)

        output_pil.save(outf)
        if args.machine_path is not None:
            os.makedirs(args.machine_path, exist_ok=True)
            machine_pil = transforms.ToPILImage()(aux_outputs["machine"]["feature"][0].clamp(0.0, 1.0))
            machine_pil.save(os.path.join(args.machine_path, fname + '.png'))

    print('\n[Average BPP]', np.mean(bpp))
    if args.stream_bpp_csv is not None and stream_bpp_rows:
        output_csv = Path(args.stream_bpp_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "image", "main_bpp", "structure_bpp", "color_bpp",
            "semantic_bpp", "payload_bpp", "container_bpp", "actual_bpp", "total_bpp",
        ]
        average = {"image": "average"}
        for key in fieldnames[1:]:
            average[key] = float(np.mean([row[key] for row in stream_bpp_rows]))
        with output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(stream_bpp_rows)
            writer.writerow(average)
        print('[Stream BPP CSV]', output_csv)


if __name__ == "__main__":
    args = parse_args_testing()
    if not os.path.exists(args.rec_path): os.makedirs(args.rec_path)
    if not os.path.exists(args.bin_path): os.makedirs(args.bin_path)
    main(args)
