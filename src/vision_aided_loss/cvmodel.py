"""Frozen DINOv2 backbone used by the paper's vision-aided discriminator."""

import os

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .DiffAugment_pytorch import DiffAugment


class DINOv2Reg(nn.Module):
    def __init__(self, output_type="conv_multi_level"):
        super().__init__()
        if output_type != "conv_multi_level":
            raise ValueError("The paper discriminator requires conv_multi_level DINOv2 features.")

        repo_path = os.environ.get("DINOV2_REPO_PATH")
        if not repo_path:
            raise RuntimeError(
                "Set DINOV2_REPO_PATH to a local facebookresearch/dinov2 checkout "
                "before Stage-2 fine-tuning."
            )
        self.model = torch.hub.load(
            repo_path,
            "dinov2_vitb14_reg",
            source="local",
            trust_repo=True,
        )
        self.model.eval()
        self.model.requires_grad_(False)
        self.register_buffer(
            "image_mean",
            torch.tensor([0.485, 0.456, 0.406]),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.229, 0.224, 0.225]),
            persistent=False,
        )
        self.model.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
            self.model.pos_embed.data,
            [16, 16],
        )

    def train(self, mode=True):
        super().train(False)
        self.model.eval()
        return self

    def forward(self, images):
        images = F.interpolate(images * 0.5 + 0.5, size=(224, 224), mode="area")
        images = (images - self.image_mean[:, None, None]) / self.image_std[:, None, None]
        features = self.model.get_intermediate_layers(
            images,
            n=8,
            reshape=True,
            return_class_token=True,
        )
        selected = [features[index] for index in (0, 4, -1)]
        return [selected[0][0], selected[1][0], selected[2][1]]


class CVBackbone(nn.Module):
    """The single frozen backbone specified by the released Stage-2 recipe."""

    def __init__(self, cv_type, output_type, diffaug=False, device="cpu"):
        super().__init__()
        if cv_type != "dinov2_reg":
            raise ValueError("The paper configuration uses cv_type='dinov2_reg'.")
        self.model = DINOv2Reg(output_type=output_type).to(device)
        self.models = nn.ModuleList([self.model])
        self.policy = "color,translation,cutout" if diffaug else ""

    def train(self, mode=True):
        super().train(False)
        self.model.eval()
        return self

    def forward(self, images):
        return [self.model(DiffAugment(images, policy=self.policy))]
