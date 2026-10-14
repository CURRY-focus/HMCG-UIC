import h5py
import torch
import argparse
import json
import os
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.utils.data.dataset import Dataset
from transformers import CLIPVisionModelWithProjection
from pathlib import Path
from PIL import Image


def apply_machine_task_preset(args):
    if not args.machine_task_preset:
        return args

    if args.machine_task_preset != "underwater_yolo_tcusod":
        raise ValueError(f"Unsupported machine task preset: {args.machine_task_preset}")

    required_environment = {
        "FATHOMNET_YOLO_PATH": os.environ.get("FATHOMNET_YOLO_PATH"),
        "TCUSOD_CHECKPOINT": os.environ.get("TCUSOD_CHECKPOINT"),
        "TCUSOD_REPO_ROOT": os.environ.get("TCUSOD_REPO_ROOT"),
    }
    missing = [name for name, value in required_environment.items() if not value]
    if missing:
        raise ValueError(
            "The underwater_yolo_tcusod preset requires these environment variables: "
            + ", ".join(missing)
            + ". Copy configs/paper.env.example and set local paths."
        )

    args.use_machine_task_loss = True
    args.machine_task_cv_type = "underwater_det+underwater_saliency"
    args.machine_task_output_type = "feature+feature"
    args.machine_task_model_classes = "ultralytics+tc_usod"
    args.machine_task_model_paths = (
        required_environment["FATHOMNET_YOLO_PATH"]
        + "+"
        + required_environment["TCUSOD_CHECKPOINT"]
    )
    args.machine_task_input_sizes = "640+224"
    args.machine_task_checkpoint_keys = "+"
    yolo_kwargs = {
        "feature_layers": [16, 19, 22],
        "feature_channels": [384, 768, 768],
        "feature_strides": [8, 16, 32],
    }
    tcusod_kwargs = {
        "repo_root": required_environment["TCUSOD_REPO_ROOT"],
        "feature_layers": ["relu5d_2", "relu4d_2", "relu3d_2", "relu2d_2", "relu1d_m"],
        "feature_channels": [128, 128, 128, 64, 16],
        "feature_strides": [16, 8, 4, 2, 1],
    }
    args.machine_task_model_kwargs = json.dumps(yolo_kwargs) + "+" + json.dumps(tcusod_kwargs)
    args.machine_task_normalize_types = "none+imagenet"
    args.machine_task_loss_type = "l1"
    args.machine_task_layer_weights = "1,1,1,1,1,1,1,1"
    return args


def parse_args_training(input_args=None):
    parser = argparse.ArgumentParser()

    # pretrained weights
    parser.add_argument("--sd_path", required=True, help="Path to SD-Turbo")
    parser.add_argument("--elic_path", required=True, help="Path to pretrained ELIC model")
    parser.add_argument("--codec_path", help="Path to a pretrained HMCG-UIC checkpoint", default=None)

    # dataset
    parser.add_argument("--train_dataset", required=True, help="Path to training dataset (hdf5)")
    parser.add_argument("--test_dataset", required=True, help="Path to test dataset (Kodak)")

    # loss function
    parser.add_argument("--gan_loss_type", default="multilevel_sigmoid_s")
    parser.add_argument("--lambda_gan", default=0.1, type=float)
    parser.add_argument("--lambda_clip", default=0.1, type=float)
    parser.add_argument("--lambda_lpips", default=1.0, type=float)
    parser.add_argument("--lambda_l2", default=2.0, type=float)
    parser.add_argument("--lambda_rate", required=True, default=0.5, type=float)
    parser.add_argument(
        "--lambda_entropy_aux",
        default=1.0,
        type=float,
        help="Weight for entropy-bottleneck CDF/quantile fitting losses.",
    )
    parser.add_argument("--lambda_structure", default=1.0, type=float)
    parser.add_argument("--lambda_structure_sketch", default=1.0, type=float)
    parser.add_argument("--lambda_structure_edge", default=0.2, type=float)
    parser.add_argument("--lambda_structure_lap", default=0.1, type=float)
    parser.add_argument("--structure_residual_scale", default=0.1, type=float)
    parser.add_argument("--lambda_color", default=0.5, type=float)
    parser.add_argument("--lambda_color_chroma", default=1.0, type=float)
    parser.add_argument("--lambda_color_mean", default=0.2, type=float)
    parser.add_argument("--lambda_color_vq", default=0.1, type=float)
    parser.add_argument("--color_codebook_size", default=256, type=int)
    parser.add_argument("--color_code_dim", default=64, type=int)
    parser.add_argument("--color_residual_scale", default=0.05, type=float)
    parser.add_argument(
        "--color_global_tokens",
        default=4,
        type=int,
        help="Use K global color tokens. The paper configuration uses 4; set 0 only for legacy dense spatial mode.",
    )
    parser.add_argument("--lambda_semantic", default=1.0, type=float)
    parser.add_argument("--lambda_semantic_attention", default=1.0, type=float)
    parser.add_argument("--lambda_semantic_boundary", default=0.2, type=float)
    parser.add_argument("--lambda_semantic_vq", default=0.1, type=float)
    parser.add_argument("--semantic_codebook_size", default=256, type=int)
    parser.add_argument("--semantic_code_dim", default=64, type=int)
    parser.add_argument("--semantic_residual_scale", default=0.05, type=float)
    parser.add_argument(
        "--semantic_global_tokens",
        default=4,
        type=int,
        help="Use K global semantic tokens. 4 keeps the semantic side stream near 0.0001 bpp for 640x480.",
    )
    parser.add_argument(
        "--use_machine_task_loss",
        "--use_task_loss",
        action="store_true",
        dest="use_machine_task_loss",
        help="Enable frozen downstream-task backbone feature loss.",
    )
    parser.add_argument(
        "--lambda_machine_task",
        "--lambda_task",
        default=1.0,
        type=float,
        dest="lambda_machine_task",
        help="Weight for the downstream-task feature loss.",
    )
    parser.add_argument(
        "--machine_task_preset",
        default=None,
        help="Optional machine-task config preset. Supported: underwater_yolo_tcusod.",
    )
    parser.add_argument(
        "--machine_task_cv_type",
        "--task_cv_type",
        default="underwater_det+underwater_saliency",
        dest="machine_task_cv_type",
        help="Paper task backbones: underwater_det+underwater_saliency.",
    )
    parser.add_argument(
        "--machine_task_output_type",
        "--task_output_type",
        default="feature",
        dest="machine_task_output_type",
        help="Backbone output mode. Examples: feature, pool, conv, conv_multi_level.",
    )
    parser.add_argument(
        "--machine_task_loss_type",
        "--task_loss_type",
        default="l1",
        choices=["l1", "mse", "cosine"],
        dest="machine_task_loss_type",
        help="Distance used between reconstructed and original task features.",
    )
    parser.add_argument(
        "--machine_task_layer_weights",
        "--task_layer_weights",
        default=None,
        dest="machine_task_layer_weights",
        help="Optional comma-separated weights for task feature tensors.",
    )
    parser.add_argument(
        "--machine_task_model_paths",
        default=None,
        help="Optional +-separated local checkpoint paths for underwater task models.",
    )
    parser.add_argument(
        "--machine_task_model_classes",
        default=None,
        help="Paper model loaders: ultralytics+tc_usod.",
    )
    parser.add_argument(
        "--machine_task_input_sizes",
        default=None,
        help="Optional +-separated input sizes for underwater task models.",
    )
    parser.add_argument(
        "--machine_task_checkpoint_keys",
        default=None,
        help="Optional +-separated checkpoint keys such as state_dict or model.",
    )
    parser.add_argument(
        "--machine_task_model_kwargs",
        default=None,
        help="Optional +-separated JSON kwargs for package.module:ClassName task models.",
    )
    parser.add_argument(
        "--machine_task_normalize_types",
        default=None,
        help="Optional +-separated input normalization modes: imagenet, none, or zero_one.",
    )
    parser.add_argument(
        "--machine_task_strict_load",
        action="store_true",
        help="Use strict=True when loading package.module:ClassName task model checkpoints.",
    )
    parser.add_argument(
        "--disable_machine_task_feature_norm",
        action="store_true",
        help="Disable channel-wise feature normalization before computing task loss.",
    )
    parser.add_argument(
        "--train_depth_dataset",
        default=None,
        help="Optional depth HDF5 paired with train_dataset, using the same sample keys.",
    )
    parser.add_argument(
        "--test_depth_dir",
        default=None,
        help="Optional validation/test depth directory matched to test_dataset by filename stem.",
    )

    # model details
    parser.add_argument("--lora_rank_unet", default=32, type=int)
    parser.add_argument("--lora_rank_vae", default=16, type=int)
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=160)
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024)
    parser.add_argument("--latent_tiled_size", type=int, default=96)
    parser.add_argument("--latent_tiled_overlap", type=int, default=32)
    parser.add_argument("--pos_prompt", type=str,
                        default="A high-resolution, 8K, ultra-realistic image with sharp focus, vibrant colors, and natural lighting.")

    # training details
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--train_patch_size", type=int, default=512,
                        help="Input image size after resizing (no cropping)")
    parser.add_argument("--train_batch_size", type=int, default=1,
                        help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate for optimizer")
    parser.add_argument("--max_train_steps", type=int, default=120000)
    parser.add_argument("--checkpointing_steps", type=int, default=10000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500,
                        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--dataloader_num_workers", type=int, default=8)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--report_to", type=str, default=None)
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"], )
    parser.add_argument("--enable_xformers_memory_efficient_attention", default=True,
                        help="Whether or not to use xformers.")
    parser.add_argument("--set_grads_to_none", action="store_true")
    parser.add_argument("--eval_freq", default=100, type=int)
    parser.add_argument("--save_val", default=True)
    parser.add_argument("--save_num", type=int, default=10, help="Number of visual samples to save")


    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    args = apply_machine_task_preset(args)

    return args


class ImageDepthTrainTransform:
    def __init__(self, image_size):
        self.image_size = image_size

    def __call__(self, image, depth):
        image = TF.to_pil_image(image).convert("RGB")
        depth = _to_depth_pil(depth)
        image = TF.resize(image, (self.image_size, self.image_size))
        depth = TF.resize(depth, (self.image_size, self.image_size))
        if torch.rand(()) < 0.5:
            image = TF.hflip(image)
            depth = TF.hflip(depth)
        if torch.rand(()) < 0.5:
            image = TF.vflip(image)
            depth = TF.vflip(depth)
        return _normalize_rgb(image), _normalize_rgb(depth)


class ImageDepthEvalTransform:
    def __init__(self, image_size):
        self.image_size = image_size

    def __call__(self, image, depth):
        if not isinstance(image, Image.Image):
            image = TF.to_pil_image(image)
        image = image.convert("RGB")
        depth = _to_depth_pil(depth)
        image = TF.resize(image, (self.image_size, self.image_size))
        depth = TF.resize(depth, (self.image_size, self.image_size))
        return _normalize_rgb(image), _normalize_rgb(depth)


def _normalize_rgb(image):
    tensor = TF.to_tensor(image)
    return TF.normalize(tensor, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])


def _to_depth_pil(depth):
    if isinstance(depth, Image.Image):
        return depth.convert("RGB")
    if depth.ndim == 2:
        depth = depth[:, :, None]
    if depth.shape[-1] == 1:
        depth = depth.repeat(3, axis=-1)
    return TF.to_pil_image(depth).convert("RGB")


class H5Dataset(Dataset):
    def __init__(self, path, transform=None, depth_path=None, depth_transform=None, paired_transform=None):
        self.file_path = path
        self.dataset = h5py.File(self.file_path, 'r')
        self.depth_dataset = h5py.File(depth_path, 'r') if depth_path is not None else None
        with h5py.File(self.file_path, 'r') as file:
            self.dataset_len = len(file)
        self.transform = transform
        self.depth_transform = depth_transform
        self.paired_transform = paired_transform
        # Preserve every sample; spatial normalization is handled by the transform.
        self.keys = list(self.dataset.keys())

    def __getitem__(self, index):
        key = self.keys[index]
        item = self.dataset[key]
        if isinstance(item, h5py.Group):
            image = item["image"][:]
            depth = item["depth"][:] if "depth" in item else None
        else:
            image = item[:]
            depth = self.depth_dataset[key][:] if self.depth_dataset is not None else None

        if depth is not None and self.paired_transform is not None:
            return self.paired_transform(image, depth)
        if self.transform:
            image = self.transform(image)
        if depth is not None:
            if self.depth_transform:
                depth = self.depth_transform(depth)
            return image, depth
        return image

    def __len__(self):
        return len(self.keys)


class ImageDepthFolder(Dataset):
    def __init__(self, image_dir, depth_dir, paired_transform):
        self.image_paths = _image_files(image_dir, exclude_dirs=[depth_dir])
        depth_paths = _image_files(depth_dir)
        self.depth_by_stem = {path.stem: path for path in depth_paths}
        self.paired_transform = paired_transform

        missing = [path.name for path in self.image_paths if path.stem not in self.depth_by_stem]
        if missing:
            raise FileNotFoundError(f"Missing depth files for {len(missing)} images, first missing: {missing[0]}")

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        depth_path = self.depth_by_stem[image_path.stem]
        image = Image.open(image_path).convert("RGB")
        depth = Image.open(depth_path).convert("RGB")
        return self.paired_transform(image, depth)

    def __len__(self):
        return len(self.image_paths)


def _is_relative_to(path, parent):
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _image_files(folder, exclude_dirs=None):
    folder = Path(folder)
    exclude_dirs = [Path(path) for path in (exclude_dirs or [])]
    suffixes = {".png", ".jpg", ".jpeg", ".bmp"}
    paths = []
    for path in folder.rglob("*"):
        if path.suffix.lower() not in suffixes:
            continue
        if any(_is_relative_to(path, exclude_dir) for exclude_dir in exclude_dirs):
            continue
        paths.append(path)
    return sorted(paths)


class CLIPLoss(torch.nn.Module):

    def __init__(self, clip_model_name="openai/clip-vit-base-patch32"):
        super().__init__()

        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_model_name).eval()
        self.image_encoder.requires_grad_(False)

        self.transform_for_clip = transforms.Compose([
            transforms.Lambda(lambda x: (x + 1) / 2.0),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ])

    def forward(self, rec, gt):
        rec_inputs = self.transform_for_clip(rec)
        gt_inputs = self.transform_for_clip(gt)

        rec_features = self.image_encoder(rec_inputs).image_embeds
        gt_features = self.image_encoder(gt_inputs).image_embeds

        rec_features = rec_features / rec_features.norm(p=2, dim=-1, keepdim=True)
        gt_features = gt_features / gt_features.norm(p=2, dim=-1, keepdim=True)

        loss = torch.norm(gt_features - rec_features, p=2, dim=-1).mean()
        return loss
