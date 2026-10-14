"""Vision-aided adversarial head used in Stage 2."""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

from .blurpool import BlurPool
from .cv_losses import losses_list
from .cvmodel import CVBackbone


class MultiLevelDViT(nn.Module):
    def __init__(self, activation=None):
        super().__init__()
        activation = activation or nn.LeakyReLU(0.2, inplace=True)
        self.spatial_heads = nn.ModuleList(
            [
                nn.Sequential(
                    BlurPool(768, pad_type="zero", stride=1, pad_off=1),
                    spectral_norm(nn.Conv2d(768, 128, kernel_size=3, stride=2)),
                    activation,
                    BlurPool(128, pad_type="zero", stride=1),
                    spectral_norm(nn.Conv2d(128, 1, kernel_size=1, stride=2)),
                )
                for _ in range(2)
            ]
        )
        self.token_head = nn.Sequential(
            spectral_norm(nn.Linear(768, 128)),
            activation,
            spectral_norm(nn.Linear(128, 1)),
        )

    def forward(self, features):
        predictions = [
            head(feature).squeeze(1)
            for head, feature in zip(self.spatial_heads, features[:2])
        ]
        predictions.append(self.token_head(features[2].float()))
        return predictions


class Discriminator(nn.Module):
    def __init__(
        self,
        cv_type="dinov2_reg",
        output_type="conv_multi_level",
        loss_type="multilevel_sigmoid_s",
        diffaug=True,
        device="cpu",
        **_unused,
    ):
        super().__init__()
        if output_type != "conv_multi_level":
            raise ValueError("The paper discriminator requires conv_multi_level features.")
        self.cv_ensemble = CVBackbone(
            cv_type=cv_type,
            output_type=output_type,
            diffaug=diffaug,
            device=device,
        )
        self.decoder = nn.ModuleList([MultiLevelDViT()])
        self.loss_type = losses_list(loss_type=loss_type) if loss_type else None

    def train(self, mode=True):
        super().train(mode)
        self.cv_ensemble.train(False)
        self.decoder.train(mode)
        return self

    def forward(self, images, detach=False, **loss_kwargs):
        if detach:
            with torch.no_grad():
                feature_sets = self.cv_ensemble(images)
        else:
            feature_sets = self.cv_ensemble(images)
        predictions = [
            decoder(features)
            for decoder, features in zip(self.decoder, feature_sets)
        ]
        if self.loss_type is not None:
            return self.loss_type(predictions, **loss_kwargs)
        return predictions
